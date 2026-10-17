# Why Maya reaches 2,300+ prefill TPS, and what to take back to Strata

Source audit: 17 October 2026. No inference was run for this document. Existing measurements are distinguished from source evidence and proposed experiments.

## Main conclusion

Maya's main advantage over a decode-like prompt implementation is **amortizing weight movement and kernel work over thousands of known input tokens**. It walks a large chunk layer by layer, groups its token assignments by expert, streams each needed expert for many rows, and uses matrix kernels. It borrows decode-cache memory to make that chunk fit, pipelines transfers with computation, and uses tensor-core prompt attention.

However, **current Qwen Strata already implements most of this architecture**. It is not accurate to explain the difference as “Maya batches; Strata does not.” Current Strata's own source records 2,485–2,660 input TPS on particular 32K Q2_0 workloads. Those are upstream source-recorded measurements, not measurements reproduced by this audit and not forecasts for our IQ3_XXS model.

There are two different porting targets:

1. **Current Qwen Strata:** identify an actual fallback, allocation, quant-format or scheduling bottleneck before copying anything. Several attractive Maya mechanisms already have equivalents or newer alternatives here.
2. **Our shelved Strata GLM port:** a much clearer gap exists between its opt-in, partly batched execution and Maya's integrated whole-chunk pipeline. Importing that architecture would be a substantial redesign, not a switch or a few kernels.

Do not attribute Maya's high prefill to our later decode transfer campaign. Stock Maya already achieved this speed class.

## Source checkpoints and how to use the references

Paths and line numbers below refer to these local snapshots; use the named functions when lines move.

| Alias | Local repository | Audited revision |
|---|---|---|
| Maya | `project-maya` | `6625fa355d395a4314a4e39492dd06ba448f21a0` |
| Strata | `Strata-upstream-20261017` | v0.1.45, `477ac13cbe8e582c20484871ce459d20d82df09d` |
| Old GLM | `Strata-glm` | `v0.1.42-93-gc323dbae` |

These paths are relative to the common LocalLLMs directory. This audit focuses on single-GPU CUDA. HIP, EXL3, multi-GPU and vision branches exist but are not explanations for our measured CUDA K-quant result.

## What the 2,300+ figure actually establishes

| Evidence | What was measured | What it does not establish |
|---|---|---|
| `docs/evidence/maya-reproduction/RESULTS.md` | Unmodified Maya v1.1.3, Maya-M: 1,141 input TPS at 2K; 2,314 at 8K | A gain caused by our changes, or a same-model comparison with Qwen |
| `docs/evidence/maya-residency-overlap/report/RESULTS.md` | GSQ-RCO Q4-attention, controlled GPU-only 8K input: 8,192 / 3.5606291 s = **2,300.7 TPS** | A statistically established prefill improvement over the single 2,188.3 TPS control observation |
| Corresponding `gpu8k-candidate/engine.log` | Tensor-core `mma.sync` prompt attention; CPU expert lane off; auto maximum chunk 32,768; reported mixer sub-batch 2,048 at setup; 22 GB-labelled expert pool | That a 32K chunk was executed for an 8K prompt, or that the setup's maximum allocation was all borrowed at runtime |
| Strata `src/prefill/prefill.cpp:247` | Source comments report fused Q2_0, 8K chunks, 32K prompts at 2,485/2,660 TPS with different rings | Our local IQ3_XXS throughput or a controlled Maya-versus-Strata speed ratio |

The Maya reproduction recipe warms prefill and its 8K sample uses an offset of 997 tokens; its corpus is generated from repository documentation. The later fixed 8K case is separate evidence. Do not blend their timing definitions, model formats or cache states.

Input TPS is not decode TPS. All prompt tokens are known. A 2,300 TPS prompt is about 0.435 ms **amortized per input token**, not a 0.435 ms sequential next-token loop. It says nothing directly about concurrent output TPS during another request's prefill. The old monitor's `WithPrefill` field measured output throughput during prefill, not prompt ingestion speed.

## The complete primary prefill path

### 1. Request dispatch, reuse and fallback

Maya `src/program/generate.cpp:1595` separates reused prompt prefixes from new work and invokes `model.prefill(part, ..., next_token)`. It normally leaves the last prompt token for the token path, which supplies logits. Checkpoint boundaries can split the work further.

`Glm5Model::prefill`, `src/core/glm_prefill.cu:3147`, declines tiny inputs (default below 32 tokens), missing prompt state, or unavailable lent memory. The caller then handles the remaining tokens through the ordinary path. A successful engine load does not prove batched prefill engaged.

**Strata lesson:** log processed new tokens, actual chunks and fallback reason. Prefix reuse, checkpoint fragmentation and fallback can dominate an apparent TPS comparison.

### 2. Allocate for prefill rather than for decode

`prefill_setup` (line 537), `chunk_bytes` (905), `prefill_carve` (near 970), `prefill_lend` (1014), and `prefill_return` (1244) form one memory protocol.

- Size the chunk against memory the expert pool can lend, rather than requiring all activations to coexist with a maximum decode cache.
- Auto sizing searches a 256-token grid, up to 32,768 on one GPU. The default lend cap is 90%; settlement can reduce it to 85% when less than 90% of expert bytes are held pinned. This is a budget cap, not a promise to borrow that fraction for every request.
- Carve only what the actual prompt needs. Evict/move affected tail residents and update ownership before reusing the region. Return the tail afterward; subsequent decode may refill it.
- Alias mutually exclusive KDA, DSA, dense-FFN and MoE scratch. Reuse temporary activation storage.
- `moe_window_rows` (216) limits live expanded MoE output rows instead of retaining all `T × top_k` rows. Partial results are combined as windows fill.

For this GLM geometry, one full FP32 routed-output buffer at 8K is `8192 × 8 × 4096 × 4 = 1 GiB`. A window of 8,192 rows reduces that buffer to 128 MiB. There are other buffers, including a separate CPU-output window: this is a component saving, not total VRAM saved.

**Why this matters:** larger chunks reduce expert reuploads. Saving scratch memory may improve speed more through a larger chunk than through reduced device-memory traffic itself.

### 3. Chunk scheduling and embeddings

`prefill_run` (3290) prepares embedding rows on the host, uploads them, and broadcasts the residual streams. On one GPU it processes full chunks plus a remainder. A multi-GPU pipeline uses a different balancing policy; do not transplant that policy to one GPU by accident.

The bulk MoE chunk and the mixer sub-batch are separate. `mixer_sub` (281) grows the latter from 256 to as much as 4,096 when it fits the already reserved scratch union. Quantized dense matrices are then reconstructed fewer times, and larger GEMMs replace many small ones. The shared expert has its own sub-batch choice.

### 4. Hyper-connections, projections and attention

`prefill_half` (1328) walks every layer over the chunk:

- Hyper-connection state updates and gating run over token rows. BF16 router/hyper-connection weights are widened for FP32 SGEMM rather than sending every projection through a low-precision shortcut.
- `hgemm_q` (1356) reconstructs quantized dense weights into bounded FP16 scratch, then calls cuBLAS GEMM with FP32 accumulation. Reconstruction is reused across the sub-batch's rows.
- KDA layers batch projections and causal convolution; recurrence advances the sequence inside the kernel while retaining state. This removes per-token host orchestration, but does **not** make causal recurrence independent across time.
- DSA layers publish latent/indexer cache entries, pool/index/select causal positions, absorb queries into latent space, attend to selected positions, and project the context back. Causal selection still applies even when a sub-batch's cache entries are written together.
- CUDA `mla_attn` in `src/kernels/cuda/glm_batch.cu:1913` selects the `mma.sync` implementation on suitable Ampere-and-newer devices, with older-device and FP32 fallbacks. Our saved run explicitly logged this path.

Important nuance: **not every attention operation is tensor-core GEMM**. The audited CUDA q-abs and output latent products use strided batched SGEMM; the central selected-latent attention has a tensor-core path. Avoid a blanket “everything is FP16/tensor cores” explanation.

KDA/MLA/DSA equations cannot be copied into Qwen's GDN/QSA. The transferable ideas are batched projections, local causal-state processing, memory reuse and attention tiling; the math must remain the target model's.

### 5. Route the whole chunk, then plan by expert

`glm_prefill.cu:1968–2229` computes router scores/top-k, obtains per-expert row counts, and constructs row/bounds maps. It partitions work into resident, prestaged, staged RAM/disk and optionally CPU groups. It also updates usage statistics for future cache placement.

This is **not** one host round trip per token per expert. It is a layer-level host planning boundary over a chunk. There is still synchronization and planning cost; source inspection alone does not quantify it.

For intuition, with uniform routing and 8,192 tokens:

- GLM: `8192 × 8 / 288 ≈ 228` assignments per expert.
- Default Qwen geometry: `8192 × 10 / 512 = 160` assignments per expert.

Actual routing is skewed, and pruning changes expert count. These are occupancy illustrations, not measured distributions. They explain why large prompt chunks offer more weight reuse than decode.

### 6. Compute resident experts and stream the rest

`run_set` (2395) packs activation rows, performs grouped native-quant gate/up MMQ, SwiGLU, activation quantization and grouped down MMQ. Supported types fuse SwiGLU into the gate/up epilogue (`STRATA_GLM_MMQ_GLU`). Gate/up output aliases output scratch until the down projection replaces it.

- Main resident experts are addressed directly in the layer's contiguous pool partition: no unconditional materialization of every resident weight into a second dense buffer.
- Light resident groups have a separate small-row kernel path (default threshold 32 rows); large groups take MMQ.
- Non-residents use **three staging groups of four expert slots** (`PrefillState::NG/GE`, line 358).
- Each group waits for its previous consumer before reuse. DMA runs on the copy stream; computation waits on that group's ready event, runs its matrix work, then records the group-free event (`2717–2778`).
- RAM-backed weights bypass disk landing. Disk-only weights use a pinned landing ring and reader queue. The RAM groups run first so disk work can catch up.

The unit of readiness is a staged expert group. Maya need not wait for all experts in the model, nor all layers, before computing. It still needs each layer's actual activations and routes; “stream layers in execution order” does not remove that dependency.

### 7. Get ahead during the next mixer

`prestage` (around 1590) predicts next-layer expert demand using persistent usage and the last chunk's row counts, excluding existing residents. It copies candidates into a separate buffer during useful mixer work, then only computes candidates actually selected by real routing.

Default capacity is up to 160 experts for sufficiently large chunks, subject to memory. This spends VRAM and potentially wasted transfer bytes to avoid later waits. The optional adaptive controller is not synonymous with the base prestage path.

This is expert prediction, not speculative token generation or expert skipping. Its benefit must be checked with useful/unused bytes and exposed waits, not the number of predicted experts alone.

### 8. Optional CPU share, combine and finish

The CPU share is gated by measured costs (`cpu_lane`, around 1660), then selects lightly routed RAM experts while predicted CPU completion fits within the remaining transfer lane. All required outputs are returned before combine. **The saved 2,300.7 TPS run had this lane off**, so CPU execution cannot explain that result.

Windowed `moe_combine_add`, the hyper-connection write, and layer iteration complete the chunk. Optional NextN cache fill and DFlash capture add their own work when enabled. The single-device terminal layer can omit outputs with no reader while still doing required cache updates (`tail_skip`, around 1412); it is gated off when those outputs have consumers. This is dead-work elimination, not dropping a required expert contribution.

Finally the engine completes the prompt's last-token path and returns borrowed memory. Include these boundaries in user-visible latency; do not count only the fastest interior chunk.

## Why the speed can be so much higher

For chunk length `T`, let `b_e` be expert bytes and `u_e` indicate a needed nonresident expert. Approximate compulsory uploads per input token are:

`sum_over_layers(sum_e u_e × b_e) / T`

Additional prestage waste, repeated chunks, spills and ownership moves increase this. In token-by-token processing the same expert may have to be accessed or transferred repeatedly. In prompt processing, one uploaded expert serves many rows before its staging slot is recycled.

Separately, matrix kernels reuse weights in the GPU and exploit tensor cores. Transfer amortization and compute amortization multiply in their effect on a poorly batched path; neither is explained by GPU cache hit rate alone.

The model difference is not simply “GLM does less work.” With default geometries, routed FFN matrix FLOPs per MoE token scale as `6 × hidden × expert_width × top_k`: GLM's `4096,2048,8` is about **4.10×** Qwen's `2560,640,10` per MoE layer. Layer counts, dense/shared layers, attention and actual formats differ. This is not an end-to-end FLOP ratio, but it rules out simplistic reasoning from model size or expert count.

The exact share of the 2,300 TPS explained by each mechanism is **not measured here**. A source audit establishes the mechanisms and saved logs establish selected engagement; only controlled ablations can assign speedups.

## Current Strata already has much of the answer

| Mechanism | Maya | Current Qwen Strata | Action |
|---|---|---|---|
| Large layer-wise prompt batches | Integrated GLM path | `Prefill::run_impl`, line 2372 | Verify actual chunk/fallback; do not reimplement |
| Borrow expert-cache VRAM | `prefill_lend` and carve | Existing prefill memory planning/lending | Compare actual budgets and eligibility |
| Expert grouping + MMQ | `run_set` | `prefill.cpp:3802–4347` | Inspect actual quant dispatch, gather cost and rows/group |
| Pinned DMA + staged overlap | Three 4-expert groups plus prestage | `stage_one` at 4164 and `Stager`; streamed rings | Compare schedules, not presence of async APIs |
| Cross-layer lookahead | Usage-based selective prestage | Long chunks stream nonresidents ahead, `146–175`, `2715` | Selective policy is useful only if it beats current streaming |
| Memory-scaled ring | Geometry-sized buffers | Byte-based ring budgets, `229–328` | Do not copy Maya's slot counts |
| Fused native experts | MMQ GLU plus type-specific paths | Fused Q2/native IQ paths, `3364–3391` | Verify our actual IQ3 format is eligible |
| Avoid resident weight gather | Contiguous pool/stride | Pointer-table/direct modes plus gather fallbacks | Audit selected mode before proposing a port |
| CPU share for sparse groups | Cost-gated | Existing automatic share, `184–224` | Model-specific cost checks; not a guaranteed win |
| Expanded output memory reduction | Windowed outputs | Fused paths already reduce buffers | Investigate only remaining MMQ/fallback footprint |
| Detailed phase counters | GLM PROF/TRACE | `STRATA_PREFILL_TIMING`, `2285–2325` | Use existing tools first |

Strata also has its own GDN/QSA, PLE, residual and cache obligations. A missing Maya analogue is not proof that Strata's work is unnecessary.

## The older Strata GLM port is a different case

`Strata-glm/src/glm/engine.cpp:481–698` exposes separate opt-ins and dependencies for prompt projections, HC, MLA, routing, shared FFN and staged experts. The implementation was not entirely serial: claiming so would erase real prior work.

But `src/glm/prefill_experts.cpp:465–543` executes each expert's assignments through repeated tiles capped at 32 columns (`include/strata/glm/prefill_experts.hpp:67`), launches gate/up and down multi-row helpers per tile, scatters outputs, and fences outstanding slots at the helper's completion. That staging strategy alone does not provide Maya's whole-pipeline MMQ, large-chunk memory planning and batched dense/attention execution.

For reviving that port, first unify the prompt executor and ownership model, then add a large-row tensor-core expert backend and batched mixer/dense work. Do not merely increase the 32-column cap: the kernel, scratch, launch geometry and quantization layout must support the new shape.

## Ranked plan for improving current Qwen Strata

### P0 — Establish which speed is actually missing

Use the intended Qwen model on v0.1.45 and retain the exact pack/config identity. One 8K prompt is sufficient for first diagnosis; optionally add a 32K prompt to expose repeated-upload cost. Count newly processed tokens, cache reuse and real occupied context separately. Do not compare output-during-prefill TPS with input TPS.

Record actual chunk lengths, ring bytes, pinned share, selected MMQ/fused/direct paths, tail size, host staging and device phase timing. If input TPS is already near 2K+, the problem may be concurrent service latency rather than standalone ingestion.

### P1 — Fix a demonstrated fallback or memory bottleneck first

If chunk size is small, map every large allocation's lifetime. Assess windowed MoE outputs and scratch aliasing only where the existing fused path does not already solve it. Keep planner accounting and execution allocation identical. Prefer a larger **effective** chunk only when saved transfers outweigh reduced residency.

Confirmation: fewer expert bytes uploaded per input token or fewer chunks, no extra memory spill, lower prompt wall time. Reject if only an interior kernel improves or decode recovery becomes materially worse.

### P2 — Improve the actual quant-format execution path

If gather/dequant/expert compute dominates, inspect IQ3_XXS eligibility for current fused/native and direct-pointer paths. Compare that implementation against Maya's direct resident MMQ and fused epilogue design. Do not convert weights to a new quantization to manufacture a win.

Confirmation: target kernel engaged, fewer gather/temporary bytes or launches, improved full-layer and prompt time, acceptable predeclared numerical results. Unsupported mixed layers must retain correctly sized fallbacks.

### P3 — Spend attention time on useful transfers

Only if `wait copy` is exposed, compare current stream-all policy with selective next-layer admission informed by routing and bytes. Large chunks may touch nearly every expert: in that case streaming all is rational and prediction only adds bookkeeping. Smaller/mixed chunks may benefit from avoiding low-probability uploads.

Confirmation: useful prestaged bytes, unused bytes, queue lead time, wait reduction and total latency. Preserve ring readiness and pinned-source lifetime. Test a task transition so an old profile cannot quietly overfit the prompt.

### P4 — Remove repeated projection work and expensive small dispatches

Inspect dequantization count and GEMM shapes per dense/mixer matrix. Maya's independent mixer sub-batching is a design pattern, not an assumption that Strata lacks full-chunk GEMMs. Increase reuse only where traces prove repeated reconstruction or inefficient small products. Consider lightweight kernels for the sparse tail of resident expert groups if MMQ overhead is visible.

### P5 — Target model-specific attention/recurrence only with evidence

If QSA or GDN dominates after the above, inspect its existing batched kernels and numerical contract. Borrow tiling, state locality and scratch ideas; preserve Qwen's equations. A GLM MLA kernel is not a QSA replacement. This is a distinct project, not a prerequisite for P1–P4.

## Minimal profiling and validation package for the next session

- Maya: `STRATA_GLM_PREFILL_PROF=1` records layer phase events; `STRATA_GLM_PREFILL_TRACE=1` adds lane completion evidence. Presence enables these diagnostic flags, so unset them for unprofiled timing. Inspect counters for RAM/disk/prestaged/CPU/resident rows.
- Strata: `STRATA_PREFILL_TIMING=1` reports GPU phases plus host setup/staging/synchronization. Its source explicitly notes that waits are charged to the phase whose interval contains them; “GEMM phase time” is not necessarily pure GEMM arithmetic.
- Nsight Systems: capture CUDA kernels, API calls, copies and NVTX across one representative complete chunk. Add/inspect layer and group IDs. Use interval unions to distinguish compute-only, transfer-only, overlap and uncovered gaps; do not sum overlapping streams.
- Attribute bytes: distinguish source reads, H2D payload, VRAM gather copies, unused prestage and additional chunk uploads. PCIe generation alone is not an explanation for achieved bandwidth.
- Separate instrumentation from throughput. Pin model, request text, settings and cache conditions. Startup logs describe maximum capacity; capture runtime chunk decisions too.
- Validate scheduling changes with fixed tokens and controlled backends. Check logits plus recurrent/KV state across chunk boundaries, partial tails and prefix reuse. For arithmetic changes, investigate differences against the existing declared reference rather than asserting bit-exact equivalence to singleton decode.
- Exercise cancellation and lend/return restoration before retaining memory changes. Validate mixed quant layers and fallback buffers. Check first decode latency after prefill as well as prompt TPS.
- Keep the first experiment to one or two useful prompts. Broaden only when a mechanism engages and shows a meaningful gain.

## What to carry forward

The highest-value lesson is **a complete prompt pipeline that makes large batches affordable and keeps them on efficient kernels**, not an isolated cache trick. For current Strata, first establish why our desired model/config is not reaching its already-developed fast path. For the shelved GLM port, Maya demonstrates the integrated architecture it was missing. Neither path justifies promising 2,300 TPS for a different model without measurement.
