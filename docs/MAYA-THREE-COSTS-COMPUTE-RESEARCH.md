# Maya compute research for the three-cost campaign

2026-10-13. Read-only investigation of `4f2f6e8ef93893ad4c18e2ab6ec702233f3fb7b5` and the saved latest decode evidence. Source locations below refer to that frozen revision; concurrent parent implementation can move line numbers. This investigation ran no inference, builds, benchmarks, GPU tools, or source modifications. The only authored file is this note. Offline SQLite aggregation and GGUF **header** inspection add source-backed bounds, not new performance measurements.

## Decision

Implement one small exact Q4_K experiment first: two output rows per warp, sharing the activation loads and activation-only minimum correction already represented by `DecW<12>::X`. Preserve every row's original lane-strided accumulation and warp reduction. This removes proven repeated integer work from the 3.211 ms dense Q4_K family, without pretending to remove required model weights.

The next exact candidate is the ordinary router's stable top-eight selection and cooperative mapped-request publication. The source performs eight rounds of CTA synchronization and a serial scan even though all 288 scores fit nine registers per lane of one warp. Its whole kernel is 0.976 ms/token, including other necessary work. A 0.1–0.3 ms useful improvement is a hypothesis, not an established result.

No source or saved evidence establishes several milliseconds of removable GPU arithmetic. Dense Q4_K is already close to the memory ceiling; attention already shares latent reads, expert gate/up already shares activations, and HC already has three generations of fusion. Changes should be judged by whole-token time with retained CPU/transfer concurrency. A faster kernel can shorten useful overlap or expose CPU waits.

## Measured accounting and hard limits

Baseline: `docs/evidence/maya-decode-latest/RESULTS.md`, `latest/deep-kernel-means.csv`, `latest/deep-summary.json`, and `latest/manifest.json`. One frozen CSV continuation, 30 prompt tokens plus 128 forced decode tokens, two sequential requests; short occupied context. GSQ-RCO 3.0bit Q4-attention, c=1, 22 GiB expert pool, transition residency, captured-only 32-block down overlap, KDA graphs. These are profiler measurements, separate from the earlier 45.20 TPS unprofiled result.

| Warm family | ms/token | Launches/token | Interpretation |
|---|---:|---:|---|
| Typed Q4_K dense GEMV | 3.2114 | 150 | Required attention/shared gate-up/dense FFN/head weight streams |
| Generic dense GEMV | 0.5164 | 98 | Primarily the BF16 auxiliary/router projections |
| Routed gate/up + fused shared down | 1.2452 | 42 | Includes every quant class and shared down |
| Routed down | 0.7921 | 42 | Includes every quant class, ordered expert reduction |
| HC3 + final HC post | 0.8120 | 91 | Residual write, mixing, norms, Q8_1 |
| Routing + shared activation + publication | 0.9759 | 42 | Fused kernel; not a measured pure top-k cost |
| DSA/MLA | 1.1837 | 77 | Includes headwise GEMVs, selection, score, split attention/combine |
| KDA prep/recurrence | 0.4613 | 68 | State update already reads state once and writes once |

GPU arithmetic interval union is 9.223 ms, including 0.594 ms overlapping fetch. Fetch is 10.088 ms for 383.000 MB, with 9.494 ms exposed in the disjoint activity partition. CPU-answer wait is 2.956 ms; other recorded GPU waits/copies 0.042 ms; no recorded GPU activity 1.156 ms. Whole token is 22.871 ms. Neither summed kernel durations nor “CPU wait” are automatically recoverable critical-path time.

The 7.257 ms `moe_component_kernel` entry is **transfer**, not arithmetic. It comprises 4.974 ms GU-prefix and 2.284 ms down copies. Its 62 launches are two nodes for each of 31 captured MoE layers. Eleven uncaptured MoE layers still use whole-expert pulls. Prefix-ready→GU and all-down-dependencies-ready→down gaps average only 0.857 and 0.881 microseconds/layer.

### Newly derived dense weight bound

Header-only parsing of the exact GGUF named in `latest/manifest.json` confirms the dense Q4_K tensors and their shapes. The source resolves those quantized tensors in `fast_setup` (`src/core/glm_fast_path.cu:651`) and launches them in `fast_layers`, `fast_dsa`, `fast_moe`, and the head. The pack index's placeholder quant type is not sufficient to infer the actual native dense format; use the source GGUF header and resolved `WSlot` types.

| Required Q4_K payload per token | Bytes |
|---|---:|
| Attention projections/output | 3,176,792,064 |
| Leading dense FFN gate/up/down | 254,803,968 |
| Shared expert gate/up | 396,361,728 |
| Output head | 356,843,520 |
| **Typed dense Q4_K family total** | **4,184,801,280** |

Shared down contributes another 198,180,864 Q4_K bytes, but rides the expert gate/up launch and is excluded from the dense-family denominator. No MTP, embedding-row reads, or routed-expert blobs are counted here. These are useful logical weight bytes, not hardware-counter DRAM traffic; caching and metadata re-reads can change physical transactions.

4.185 GB / 3.211 ms = approximately 1.303 TB/s of useful Q4_K weight payload. Using the RTX 5090's nominal 1,792 GB/s memory specification gives a 2.335 ms ideal streaming floor, hence **at most 0.876 ms dense-family improvement from bandwidth alone**, before activation/output traffic and instruction limits. This is an optimistic counterfactual, not measured available headroom. The head alone already streams 356.84 MB in 0.2318 ms (about 1.54 TB/s); its nominal bandwidth-only headroom is only about 0.033 ms.

An offline rank-to-source mapping of the 150 Q4_K launches in each of the saved 128 warm tokens gives the following. Every token has exactly the expected launch sequence (34 recurrent layers, eleven DSA layers, three dense FFNs, 42 shared gate/up batches, one head), and the sum reconciles to `deep-kernel-means.csv`. This mapping uses source order, not additional NVTX phase timestamps.

| Typed Q4_K substage | ms/token | Launches/token |
|---|---:|---:|
| KDA Q/K/V | 1.38093 | 34 |
| KDA output | 0.53851 | 34 |
| DSA Q-a | 0.07627 | 11 |
| DSA Q-b | 0.12150 | 11 |
| DSA output | 0.33064 | 11 |
| Shared expert gate/up | 0.35250 | 42 |
| Dense FFN gate/up | 0.11301 | 3 |
| Dense FFN down | 0.06626 | 3 |
| Head | 0.23176 | 1 |

## C1 — exact two-row Q4_K dense GEMV (first implementation)

**Proven redundancy.** `dot_q4_K` (`src/kernels/cuda/glm_fast.cu:117`) performs four DP4A instructions for weight-dependent products and four more for the activation-only sum used by the minimum correction, for every row and call. `mv_kernel_t<12>` (`:1224`) assigns one row to each warp. It repeats that activation load and integer sum across all output rows.

`DecW<12>` (`:269`) already separates weight decode from activation load and computes the same two eight-value integer sums in `X::su[2]`. `DecW<12>::dot` preserves the original integer DP4A nesting, `sumf_d`/`sumf_m` float update order, and final `dm.x * sumf_d - dm.y * sumf_m`. Reusing this helper avoids introducing a new quant decoder. Do not use Q8_1's whole-block floating sum in place of these integer eight-value sums: that is a different quantity and rounding path.

Two rows sharing one `X` require twelve DP4A operations instead of sixteen per call, a **25% reduction in this instruction category**. Four rows would reduce it by 37.5%, but require more live row accumulators/weight state and a smaller grid; R=2 is the bounded first experiment. Activation loads halve within a warp; required weight rows remain unchanged. L1 already helps activation reuse, so this is not a claim to halve external memory traffic.

### Kernel pseudocode

Insert adjacent to `mv_kernel_t`; retain `MvBatch` job mapping, `g_skip`, `mv_out`, and 256 threads. For R=2, eight warps produce sixteen rows/block.

```cpp
template<int R>
__global__ void mv_q4_rows_kernel(MvBatch b) {
    if (b.skip && *b.skip) return;
    int ji = job_for_block(b.blk_end, b.n, blockIdx.x);  // existing scan
    const MvJob& J = b.j[ji];
    int first = ji ? b.blk_end[ji - 1] : 0;
    int lane = threadIdx.x & 31;
    int row = ((blockIdx.x - first) * 8 + (threadIdx.x >> 5)) * R;
    if (row >= J.n_out) return; // warp-uniform
    float acc[R] = {};
    size_t rb = rbytes<12>(J.n_in);
    using Fm = F<12>;          // qk=256, ipb=16, step=2
    for (int k = lane; k < (J.n_in / Fm::qk) * Fm::ipb; k += 32) {
        int kbx = k / Fm::ipb;
        int iqs = Fm::step * (k % Fm::ipb);
        auto x = DecW<12>::load_x(
            (const block_q8_1*) J.xq + kbx * (Fm::qk / 32), iqs);
        #pragma unroll
        for (int r = 0; r < R; ++r) if (row + r < J.n_out) {
            DecW<12> w;
            w.load((const uint8_t*) J.w + (row + r) * rb, kbx, iqs);
            acc[r] += w.dot(x);
        }
    }
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        float s = warp_sum(acc[r]); // XOR offsets16,8,4,2,1, as today
        if (lane == 0 && row + r < J.n_out)
            J.y[row + r] = mv_out(J.alpha, s,
                J.bias ? J.bias[row + r] : 0.0f);
    }
}
```

Every row uses the original `row_dot<12>` (`:529`) k sequence. Do not split K, add a parallel float reduction, combine rows' floats, replace the correction with a block sum, use approximate tensor-core accumulation, or alter FMA/fast-math flags. Integer correction sharing is exact; full-kernel exactness still requires the ordinary parity gate because compiler reassociation/inlining can matter.

### Complete dispatcher and scope

In `mv` (`:4104`), alter **only the one-column typed type-12 batch** around `:4232–4257`: choose `per = MV_ROWS * R` for eligible jobs and use `ceil(n_out/per)` for **both** cumulative `blk_end` and total launch blocks; launch the new kernel for that batch. Using sixteen-row indexing with the old eight-row block boundaries is incorrect for multi-job batches. Preserve the typed kernel for ineligible jobs and all other quant types.

Eligibility: opt-in enabled in ordinary c=1 decode, type12, `ncol==1`, `n_in%256==0`, valid aligned Q4_K/Q8_1 buffers, positive shapes; handle arbitrary `n_out` tails. Keep alpha/bias and skip behavior as above. A per-call/RAII decode scope is safer than assuming ncol=1 implies decode: `mv_nt` falls back to `mv` for n<=1 (`src/core/glm_rows.cu:253`), and other window/drafter code calls the same API. Preserve prompt MMQ/light-row, multicol, and `mv_nt` implementations. Keep the switch off before capture and fixed for a session; existing captured graphs bake the chosen kernel.

**Instrumentation:** report the selected row count at startup; use the distinct kernel name to establish 150 launches/token for an all-eligible diagnostic and compare substage durations. If host dispatch counters are added, distinguish capture-time selection from replay; capture-only counters cannot be reported as per-token engagement. Preserve quant/weight payload counters and route/placement counters. Inspect compiler resource reports only if a regression motivates it; register spilling is a plausible cost, not a measured defect in the current kernel.

**Bound:** practical target 0.2–0.5 ms/token in dense arithmetic; an optimistic 0.8 ms is already close to the bandwidth-only ceiling. Larger claims need a specific additional source of waste. This estimate is provisional; R=2 can lose if halving blocks reduces memory concurrency. The very fast head should be reported separately so a projection gain cannot conceal a head regression.

**Decisive experiment:** parent only, same binary/fixture/config, R=1 vs R=2, first+repeated 128-token CSV requests, full forced logits and route/tier counters. Require exact all logits, finite values, unchanged routing/CPU assignment and payload bytes. Compare native unprofiled wall time; one diagnostic capture only if wall improvement exists or attribution is needed. Reject a kernel-duration gain without a useful whole-loop gain. A single second workload at 8K occupied context can qualify prefill/graph/buffer pressure for a candidate that passes; do not launch a broad suite or R/U/grid sweep.

## C2 — exact main-route selection and coalesced publication

Locations: `topk_argmax` (`src/kernels/cuda/glm_fast.cu:2839`), `moe_route_kernel` (`:2919`), main request publication (`:3205`), and `MoeRequest` (`include/strata/kernels/glm_fast.hpp`).

The retained ordinary route has E=288,k=8, no prediction, no ahead layers, no near-rank logging. The current320-thread top-k performs eight rounds, each with two CTA barriers and a tid0 scan of ten warp winners. Scores use `expf`, sigmoid+bias, and exact stable ties by lower expert ID. After selection, tid0 performs the double-precision probability sum/normalization and order-sensitive CPU/spare planning; preserve those parts.

**Complete selection implementation:** provide an opt-in specialization for the ordinary geometry/flags. Keep the existing score computation/nonfinite check, then let warp0 load `s_sel[lane+32*j]` for j=0..8 into nine registers/lane, with invalid IDs at -infinity. For each of eight ranks, choose the best local `(score,id)` and use the existing lexicographic comparator through XOR shuffles16,8,4,2,1. Every lane gets the winning ID, sets only its own matching register to -infinity, and lane0 writes `s_ids[rank]`. One final CTA barrier publishes IDs to the unchanged planner. The LOOKAHEAD loop already implements this pattern in the same file; it is a source precedent, not a measured main-route gain.

Fallback to existing top-k for near-rank logging, prediction/ahead, other shapes, and any exceptional control path. Do not change `expf` to `__expf`, sigmoid/bias, probability precision, frequency aging, chosen CPU experts, route order, spare claim order, or the invalid-route publication protocol. A compare-only top-k reduction may change thread mapping without changing arithmetic results.

### Minimum exact router patch

For the first implementation, change only the first `topk_argmax` call after the existing score/nonfinite-check barrier. All threads evaluate the same eligibility expression from shared `bad` and immutable arguments. The old helper remains the fallback; prediction and near-rank calls remain unchanged. Do not move the probability normalization or planner in this first patch.

```cpp
// s_p and s_sel are filled by the existing expf/sigmoid/bias code.
// The existing __syncthreads() precedes this point.
const bool use_warp = opt_in && E == 288 && a.k == 8 && bad == 0 &&
    a.near_n == 0 && a.pred_logits == nullptr && a.n_ahead == 0;
if (use_warp) {
    if ((tid >> 5) == 0) {
        float values[9];
        #pragma unroll
        for (int j = 0; j < 9; ++j) values[j] = s_sel[lane + 32 * j];
        for (int rank = 0; rank < 8; ++rank) {
            float bv = -INFINITY;
            int bi = 0x7fffffff;
            #pragma unroll
            for (int j = 0; j < 9; ++j) {
                int ei = lane + 32 * j;
                float v = values[j];
                if (v > bv || (v == bv && ei < bi)) { bv = v; bi = ei; }
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            // Every lane now has the same stable winning (score,id).
            #pragma unroll
            for (int j = 0; j < 9; ++j)
                if (lane + 32 * j == bi) values[j] = -INFINITY;
            if (lane == 0)
                s_ids[rank] = bi >= 0 && bi < E && isfinite(bv) ? bi : -1;
        }
    }
    __syncthreads(); // publish all s_ids before tid0 checks/normalizes them
} else {
    topk_argmax(s_sel, E, a.k, s_ids, s_bv, s_bi);
}
// Continue the original s_ids validity check, prediction/ahead branches,
// double probability sum, CPU selection, spare planning and publication.
```

`opt_in` must be a kernel argument or template selection fixed before graph capture; a device kernel cannot call `getenv`. Host selection can choose an optimized RouteArgs specialization while keeping the current interface/default behavior. Comparing max pairs is order-independent under the original finite-score/tie rule; none of the probability arithmetic changes. The normal specialization does not mutate `s_sel`, which is safe only because near and prediction are disabled. Keep those eligibility guards rather than silently permitting a later consumer of the unconsumed score array.

**Complete publication implementation:** after the unchanged planner, use lane i<k to write `ids[i]`, `w[i]`, `promo_ptr[i]`, `pred[i]`, and `cpu_src[i]` to their existing mapped arrays, while tid0 writes scalar header fields. Cooperatively publish any active PF/ahead/near arrays with their existing initialized values. Keep the CPU input copy unchanged and preserve a system fence for **every writer**, CTA completion, tid0's final system fence, and seq written last. The host must never see seq before payload. Skipping inactive-field writes is unnecessary for the first implementation and risks stale ring fields if consumers still read them.

**Engagement:** distinct optimized route kernel/specialization, 42 normal optimized routes/token, fallback reasons at setup/capture, and diagnostic publication/selection phase stamps if needed. Avoid enabling near16 route logging for the performance headline because it intentionally adds extra selection work. CPU service-start and answer-publication NVTX spans can show whether earlier route completion advances the CPU lane.

**Bound:** 0.976 ms is the entire route-family ceiling, including shared activation CTAs, GPU plan mutation, mapped host input writes, and fences. A practical0.1–0.3 ms reduction is plausible but unmeasured. Top-k alone may be a minority of this kernel. This is especially compatible with CPU scheduling: earlier seq publication can help the current CPU answer while preserving assignment and transfer demand.

**Experiment:** the same frozen CSV pair, exact full logits/route order/masks and bytes, optimized selection+publication versus retained; if a useful gain exists, optional one ablation isolates selection versus publication. Use the second 8K workload only for a passing combined candidate. STOP/recovery and teardown must retain seq/ownership behavior; no arbitrary full sweep.

## C3 — remove HC grid-spin dependencies while preserving the exact reduction tree

Locations: `hc3_kernel` (`src/kernels/cuda/glm_fast.cu:2153`), `grid_barrier` (`:2024`), `hc` dispatcher (`:4349`), `HcArgs`, and `fast_layers` (`src/core/glm_fast_path.cu:4779`).

This is lower priority. HC3 reads only786,432 bytes of weights/call, 90 calls/token, yet costs0.811 ms. It uses32 embedding-range blocks, two global spin barriers, and repeats the25 total reductions and20-iteration Sinkhorn in every block. Those are source-visible low-parallelism dependencies, not evidence that another launch arrangement will win. HC3 already preloads weights and replaces earlier serial total reductions with warp reductions.

**Complete captured-only alternative:** split the same operation into three graph nodes, preserving exactly32 blocks and original per-thread assignments:

1. Phase A performs the current weight preload, R write,24 partial dots, and local sum of squares. Write the existing `[25][32]` partials; finish by kernel completion instead of a grid spin.
2. Phase B reduces each total with the original lane=block XOR tree, computes each block's exact four pre-gates and128 mixed values, and writes the original local mixed-square partial. Only block0 computes/writes the unchanged Sinkhorn/post/comb; all blocks persist mixed values to a new4096-float buffer. With inter-kernel ordering, block0's work needs no added grid barrier.
3. Phase C reduces the32 mixed-square partials using the original XOR tree, computes the same inv2, norm multiplication, and Q8_1 store for128 values/block. Read the persisted mixed buffer.

Allocate the4096-float intermediate once in `FastState` and retain it through graph teardown. Phase B reads `R_new` when `block_out` exists, else `R_old`; do not read an unwritten first-layer residual. No changes to residual pointer swaps or graph key accounting. Scope the new dispatcher mode to captured recurrent ordinary decode; uncaptured MLA layers retain HC3 because extra host submissions previously lost. Preserve prefill and other hc callers.

This removes the two spin barriers and redundant Sinkhorn work, but adds two graph nodes and intermediate global reads/writes. It **cannot** choose a different block count/reduction tree if exact parity is required. Reducing Sinkhorn iterations, using tensor cores for the BF16 mix, or fusing sums with a different order would require different numerical behavior and is excluded.

**Bound/engagement:** all90 HC calls cost0.811 ms; captured recurrent subset costs approximately a share of that, not the full amount. New phase kernels should appear exactly twice per captured recurrent layer/token,68 calls of each phase if all34 recurrent layers are enabled. Practical net ceiling perhaps0.1–0.3 ms, with zero or negative gain quite credible because added node latency replaces cheap grid barriers. Require exact intermediate norms/Q8_1 and full logits. One CSV pair can reject it; only proceed if C1/C2 fail to offer a better bounded path.

## Smaller or incomplete opportunities

| Mechanism | Source/evidence | Useful bound and decision |
|---|---|---|
| Small-context DSA selector | `dsa_topk.cuh` does four histogram passes,1024-thread scans, then sorts <=1024 winners even for n_vis7–39 | Current short selector0.181 ms across11 calls. Specialize small n_vis with direct stable sort of all scores, retaining descending score/lower-index order and tail cells; likely <=0.1–0.15 ms. Selecting all visible pools in chronological order would change attention accumulation and is excluded |
| Short last MLA chunk | `mla_split_kernel:2743` always performs32 score/context iterations, including invalid last-chunk cells | Skip exactly masked scores/cells only with identical partial outputs; one final chunk's waste, likely tens of microseconds/token. Changing chunk size/head grouping changes softmax reductions and cannot be presumed exact |
| Q4_K routed rows/shared down | `rows_dot:1043` shares activations for Q2_K/Q3_K but uses separate Fm::dot for Q4_K; fused shared down uses generic any_row_dot | Same DecW activation sharing can cover six Q4_K expert layers and shared down, but Q4 expert GU+down only0.146 ms and shared down is inseparable in timing. Do not sell the whole1.245 ms family as removable redundancy |
| KDA recurrence | `kda_rec_kernel:2365` keeps64 state rows/lane in registers and reads/writes state once |0.222 ms total recurrence. No substantial redundant state read established. Changing half reductions, state precision, or truncating history violates scope |
| MLA latent reads | Split attention already shares32 latent rows across16 heads/block | Current code already removed the per-head64x latent reads. A proposal to “share KV once” must explain what remains; generic FlashAttention branding adds no new mechanism |
| Capturing the remaining DSA layers | `fast_dsa:4509` uses host p, dynamic score grid/n_sel/n_chunks and lazily managed MLA scratch; `LayerGraphs` only captures recurrent layers | Potential fraction of1.156 ms inactive budget, which also includes boundary/head/embedding work. A complete design needs device step parameters, bounded grid buckets, owned preallocated scratch, bucket graph invalidation, and unchanged score/softmax order. Capturing host p as-is would replay stale positions. Too large to precede the exact C1 experiment |

For DSA capture, simply launching a maximum-context grid is not a complete optimization: at capacity131072 it would enqueue thousands of unused score blocks on short requests. Device live sizes must bound access and bucketed grids must bound unused work. Fixed attention buckets may add all-masked chunks, and their contribution to maxima/sums must be bit-exact before accepting parity. Mutable graph-node updates are another option but need their submission cost measured; they are not automatically cheaper than the current direct path.

## Combinations with transfers and CPU scheduling

The retained down32 fork/join begins after all GU prefixes arrive (`fast_moe_experts:4719`) and joins before down reads and slot reuse. Expert arithmetic totals2.037 ms but much of GU is already under down-copy tail. A faster GU can reduce overlap instead of wall time. Preserve those events, source lifetimes, and consumer order in any arithmetic experiment.

The older resident-only GU prototype on `research/maya-transfer-prototypes` did compute resident and fetched halves in separate masked launches. It was exact, but overlap1.037 ms was outweighed by arithmetic9.154→10.257 ms and inactive gaps1.942→4.169 ms. The later full-grid component fork/join also produced2.768 ms overlap while arithmetic rose9.283→11.347 ms. Down32 plus captured-only scope is the reason the retained design avoids these costs. Any retry must remove the known dependency costs, preserve one computation per expert, restrict contention, and show lower wall time. Enabling an old flag again is insufficient.

An exact early-CPU route is a scheduling alternative: current `fast_moe:4617` batches the BF16 router with Q4_K shared GU, and `mv` issues the typed Q4 work before the generic router. `moe_route` then couples publication to shared SwiGLU readiness. A complete ordinary-captured alternative would compute router first, publish a route-only request, compute shared GU, then its shared SwiGLU before fused shared down. Keep planner/CPU choices and sequence accounting unchanged; separate publication from shared activation once, never double-publish a route or double-apply CPU answers. The saved shared GU projection duration is only0.3525 ms/token, which bounds the same-token useful CPU lead from moving just that preceding work. New host submissions on DSA can erase it; captured-only trial is the appropriate initial scope. Prefill stays unchanged. This proposal is compatible with C1/C2 but improvements are not additive.

Earlier route publication may also advance CPU reads into the transfer window and worsen shared host-memory contention. `docs/MAYA-TRANSFER-ATTRIBUTION.md` establishes this tradeoff: CPU-first raises fetch bandwidth37.32→50.78 GB/s but slows warm wall23.495→27.900 ms; eight CPU workers were neutral,45.10 vs45.20 TPS. The actual-address replay found copy-only51.95 GB/s, plus CPU45.16, plus routed GPU50.20, plus both43.21. Those findings rule out serializing CPU work merely to improve the copy bar. Keep the normal concurrency and judge the complete loop.

C1 should first run with the retained transition/down32 configuration. If C1 changes GPU time, the previously calibrated per-layer CPU split may become slightly stale; do not immediately resweep it. Only a clear new exposed-wait pattern justifies one bounded scheduling follow-up. Existing D6 altered seven of42 plans and lost, and earlier admission/CPU gains were not additive after expanded backing removed churn.

## Bounded acceptance plan

One parent-owned inference/build slot. Preserve the retained binary/config/evidence and benchmark against that baseline. The first workload is the frozen CSV pair used by the latest evidence; same prompts/forced IDs, initial profile, pool/backing, CPU plan and flags. Candidate arithmetic/scheduling must match every logit exactly, remain finite, and retain ordered route IDs, masks, demand bytes and owner counts. Compare timing separately from the diagnostic capture. Do not treat forced inputs as proof of fixed routes.

If the first experiment is useful, select **one** second workload at8192 occupied context using batched prefill, then a bounded continuation. Check prefill time and exact output, graph capture/replay/invalidation, retained scratch/slot lifetimes, STOP/recovery and teardown. A short passing kernel screen alone is insufficient to claim preserved prefill. Do not introduce new arithmetic thresholds or use existing hybrid-placement differences as permission for a new discrepancy.

C1 and C2 together could plausibly save a few tenths of a millisecond; C3 is a riskier bounded follow-up. Even combining their speculative ranges does not establish60 TPS (16.67 ms/token). The current22.871 ms loop would need about6.20 ms further reduction. Transfer demand/CPU completion remain the larger opportunity, and no arithmetic redesign here eliminates their dependency.

## Sources checked

- Frozen source: `src/kernels/cuda/glm_fast.cu`; `src/core/glm_fast_path.cu`; `src/core/glm_fast_state.hpp`; `src/core/glm_layer_graphs.hpp`; `src/kernels/cuda/dsa_topk.cuh`; `include/strata/kernels/glm_fast.hpp`; `include/strata/core/glm_model.hpp`; ordinary versus prefill/window callers.
- Latest profile and manifest under `docs/evidence/maya-decode-latest/latest`, including readonly `timeline.sqlite` aggregation; exact source GGUF metadata/tensor header and pack metadata named by that manifest.
- `docs/MAYA-RESIDENCY-OVERLAP.md`; `docs/MAYA-DEMAND-CAMPAIGN.md`; `docs/evidence/maya-demand/RESEARCH.md`; `docs/evidence/maya-transfer/RESEARCH.md`; `docs/evidence/maya-deep-profile/RESULTS.md`; `docs/MAYA-TRANSFER-ATTRIBUTION.md`; `docs/evidence/maya-transfer-attribution/RESULTS.md`; `docs/GLM-KDA-GRAPHS.md`.
- Read-only `git show` of retained/prototype kernels. Prior DS4/Strata transfer/early-shared results were considered through the campaign records; they do not prove a new gain on this model/hardware.

No implementation or new timing result is claimed by this note. Parent implementation and tests proceed separately.
