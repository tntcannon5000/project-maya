# Exact Q4_K IMMA feasibility experiment

This is an isolated prototype. No engine dispatcher, defaults, model files, or numerical thresholds change. The author performed source inspection only: no build, GPU execution, inference, or performance claim. Parent validation subsequently passed240 strict cases; results and binary hashes are in docs/evidence/dflash-batched-verification/imma-q4-feasibility. T3 loses substantially; large-K T5 shows a small isolated advantage, not a full-model gain.

## Finding and scope

The main projections already use `MvJob.ncol=T` and `mvc_kernel_t`, decoding each row's weights once before the column loop (`src/kernels/cuda/glm_fast.cu:1666`). A full Q4_K INT8 Tensor Core implementation already exists (`mvc_mma_q4_K`, same file near line1573). Its `approx` gate is necessary: it computes full32-value integer dots, reassociates the scale/min expression, splits superblocks across eight warps and sums their float partials differently. Setting `approx=true` on target jobs would not preserve this campaign's exact-logit gate.

Q4_K has256-value/144-byte superblocks, eight32-value subblocks, six-bit scales/mins and twoFP16 global coefficients (`third_party/ggml/ggml-common.h`). Activations already contain32 signedINT8 values plus FP16 scale/sum in36bytes. No new activation quantization or global weight dequantization is required. Only the INT8 values and scale are used; the FP16 sum is deliberately excluded.

The current exact `DecW<12>` instead computes an eight-value integer product for each quarter of a32-value subblock, combines a pair of neighboring subblocks in FP32, and returns one16-value contribution. A logical lane visits superblock parity0 or1, one of four subblock pairs and one of four quarters. It accumulates successive superblocks with stride2, then reduces the32 logical lanes with XOR offsets16,8,4,2,1 (`F<12>`, `row_dot`, `mvc_kernel_t`, `warp_sum`). Exact integer MMA across all32 values loses those FP boundaries.

## Complete tile design

Each trial CTA has eight warps and16 real output rows. Each warp owns `(superblock parity, subblock pair)`. Its K loop is exactly `kb=parity; kb<nb; kb+=2`. Four virtual columns represent each token, masking the B fragment to select only q8 words `q` and `q+4` for quarterq. Thus a normal `m16n8k32.s32.s8.s8` instruction returns the same eight-value integer dots as the original DP4A nesting. Other K positions are zero; values and integer sums are far below INT32 overflow.

T3 needs12 virtual columns/two8-column tiles, T5 needs20/three tiles. Both subblocks of a pair retain their separate activation scale and integer scale/min products. The original two-step `sumf_d`, `sumf_m`, global coefficient expression, and per-logical-lane accumulator are preserved. A shared scratch array restores the original32 logical lanes before the identical XOR tree. Output alpha/bias uses the engine's explicit separate rounded multiply/add.

Scratch is `16*T*32` floats:6KiB atT3 and10KiB atT5. There is no expanded global weight buffer and no extra staging kernel. This is a complete output tile; all original K contributions, minima, scales, row tails and output strides are included. The source-level arithmetic mapping is complete, but compiler FMA choices and MMA register mapping still require the strict binary comparison. The trial adds two/three times the MMA tile work of the approximate full32-value route, B masks, more live FP state, shared stores/loads and a remapping barrier. It may lose to DP4A.

## Hardware and ceilings

Local evidence identifies RTX5090, Windows, CUDA13.2 (`docs/MAYA-LOCAL-REPRODUCTION.md`); the root build defaults tosm120. The instruction used here is already present in the engine'ssm80+ implementation. Existing Nsight timeline summaries provide durations and overlaps, not achieved DRAM/SM/tensor throughput or instruction-stall counters. They cannot establish that T3/T5 projections are compute bound.

Current serial phase evidence (`docs/evidence/dflash-batched-verification/hybrid-phase-summary.json`, off-resume) reports6.04ms for KDA/DSA projection intervals, denseFFN and head combined;7.62ms including router/shared gate-up;8.38ms including shared down. These contain BF16 work, launches and possible host gaps. Prior header-only payload accounting (`docs/MAYA-THREE-COSTS-COMPUTE-RESEARCH.md`) gives4.1848GB of Q4 dense weights per complete pass, plus0.19818GB shared down. At the5090's nominal1.792TB/s those bytes alone imply ideal floors2.335/2.446ms. Under the same weight stream, even ideal elimination of every other cost in those inclusive stages saves at most5.285ms, or5.934ms with shared down. These are optimistic bandwidth counterfactuals, not forecasts or hardware-counter results; caching and actual transactions differ.

Before the tile measurements, a 1–3 ms full-round improvement was a bounded hypothesis. The initial result and actual shape accounting below reduce the apparent opportunity to well below 1 ms. Neither could supply the current sample's 12.48 ms/round gap to 60 TPS. The principal current costs remain 26 ms fetch and an 8.5 ms expert interval.

## Build and run by parent

Use the same Visual Studio/CUDA developer environment as the existing build. The standalone project links the existing `build/strata_kernels.lib` and `build/strata_core.lib` so the reference is the actual engine's `mv()`. It does not compile a copied reference or rebuild the engine. Override `MAYA_KERNELS_LIBRARY` and `MAYA_CORE_LIBRARY` if the archives are elsewhere. No fast-math flags should be added.

```powershell
cmake -S tests/imma_q4 -B build/imma-q4 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/imma-q4 --target imma_q4_feasibility
build/imma-q4/imma_q4_feasibility.exe
build/imma-q4/imma_q4_feasibility.exe --bench
build/imma-q4/imma_q4_feasibility.exe --model-shapes
```

The test forces the ordinary typed tiled engine path with `approx=false`. Its240 correctness cases compare the trial against both the current multicolumn kernel and separate singleton `mv()` calls using exact bitwise equality, including signed zero. There is no relative/absolute tolerance. Cases cover ncol3/5, n_in256/512/768/4096/8192/12288, row tails1/15/16/17/67, arbitrary packed six-bit scales/mins and extrema, varied finite FP16 coefficients/scales including zero and a negative global coefficient, signedINT8 activation extrema, nondefault activation/output strides, biased/unbiased alpha, and output padding canaries. Activation stride padding is nonzero poison. Each q8FP16 sum is deliberately unrelated to its integer sum.

`--bench` repeats the exactness gate first, then times full8192-output-row matrices with input widths4096/8192/12288 and ncol3/5. Its distinct physical weight-address ring exceeds twice the reported L2 capacity. The engine/trial/trial/engine order reports both controls; allocations, input copies and warmup are outside timing. This avoids benchmarking one small L2-resident matrix repeatedly. Timing is CUDA event time for synthetic matrices, not an end-to-end prediction or controlled GPU clock experiment. Retain ptxas register/spill/shared reports. Stop on any mismatch; do not widen tolerances. A compute-sanitizer run can check illegal accesses before timing if needed.

Only after this gate passes and the complete tile beats the actual DP4A kernel should the parent consider optional target integration and the existing forced full-logit/partial-commit/reset/pressure gate. Require useful clean full-round savings with identical histories, route/placement/payload accounting and acceptance; a faster isolated tile alone is insufficient.

## Initial result and actual model shape accounting

The parent-measured T3 trial lost materially at every tested shape. At T5 the initial8192-output-row benchmark was neutral for K4096 and approximately10.6%/12.5% faster on average for K8192/K12288. Controls also varied appreciably. Crucially, none of the target model's large-K dense Q4 matrices has8192 output rows. All have4096 output rows, and the DSA output has a previously unmeasured K16384. Shape-gating based only on the first synthetic wins would therefore be unsupported.

A second header-only inspection of the exact GGUF named in `docs/evidence/maya-decode-latest/latest/manifest.json` ended at byte9,516,456, before any tensor payload. It identifies the following actual Q4_K shapes for target layers0–44; K is `n_in` and N is `n_out`:

| Family | K | N | Matrices per complete window | Q4 bytes per window |
|---|---:|---:|---:|---:|
| KDA output |8192|4096|34|641,728,512|
| Dense FFN down |12288|4096|3|84,934,656|
| DSA output |16384|4096|11|415,236,096|
| **Large-K total** |||**48**|**1,141,899,264**|

The remainder is102 KDA Q/K/V matrices K4096,N8192;11 DSA Q-a K4096,N1536;11 DSA Q-b K1536,N16384;6 dense FFN gate/up K4096,N12288;84 shared gate/up K4096,N2048;42 shared down K2048,N4096; and the head K4096,N154880. The full Q4 dense inventory, excluding embeddings and MTP, is305 matrices and4,382,982,144bytes including shared down. This reconciles with the earlier4,184,801,280-byte count excluding shared down. Native pack index quant types are placeholders; its shapes corroborate the GGUF header, whose real type12 establishes Q4_K.

Source confirms one large-K output projection per layer per full window: `fast_window_layers` creates KDA output with `(DI,E)`, dense down with `(g.n_ff_dense,E)`, and `fast_dsa_window` creates DSA output with `(g.n_head*g.v_head,E)` in `src/core/glm_window.cu` (currently near lines1569,1593 and554). Each uses the full group ncol. Reduced/staggered groups would need their own eligibility rather than assuming T5.

Using only the first benchmark's average deltas and scaling linearly from N8192 to N4096 predicts0.1086ms for all34 KDA outputs and0.0161ms for the three dense-down matrices: **0.1248ms per T5 window**. Extrapolating K8192 to K16384 adds another0.0703ms over eleven DSA outputs, for **0.1950ms total**. These are deliberately labeled unsupported shape extrapolations, not forecasts: halving N changes CTA count and memory concurrency, and K16384 was not measured. A generous15% saving across the entire earlier T5 KDA-output1.34ms, DSA-output0.69ms and dense-FFN0.38ms intervals would still save only0.362ms, and that includes ineligible FFN work and event overhead. The likely opportunity is well below1ms, around0.2% of an88ms window, before integration overhead or exposed waits.

The bounded next measurement is `--model-shapes`: it retains the initial240 strict cases, adds six exact comparisons at N4096,K8192/12288/16384,T5 with two scale patterns, then times precisely those three shapes with the same L2-exceeding address-ring method. `--bench` remains unchanged. No new production implementation or GPU execution was performed by the author. Unless this one exact-shape measurement reveals a much larger effect than the first data imply, **reject production integration as immaterial to the current throughput goal**. T3 remains excluded regardless of a T5 result.
