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

A1–3ms full-round improvement is a bounded hypothesis worth testing, not an established gain. It cannot supply the current sample's12.48ms/round gap to60TPS by itself. The principal current costs remain26ms fetch and8.5ms expert interval. No source evidence justifies promising the needed whole-round leap from dense IMMA alone.

## Build and run by parent

Use the same Visual Studio/CUDA developer environment as the existing build. The standalone project links the existing `build/strata_kernels.lib` and `build/strata_core.lib` so the reference is the actual engine's `mv()`. It does not compile a copied reference or rebuild the engine. Override `MAYA_KERNELS_LIBRARY` and `MAYA_CORE_LIBRARY` if the archives are elsewhere. No fast-math flags should be added.

```powershell
cmake -S tests/imma_q4 -B build/imma-q4 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/imma-q4 --target imma_q4_feasibility
build/imma-q4/imma_q4_feasibility.exe
build/imma-q4/imma_q4_feasibility.exe --bench
```

The test forces the ordinary typed tiled engine path with `approx=false`. Its240 correctness cases compare the trial against both the current multicolumn kernel and separate singleton `mv()` calls using exact bitwise equality, including signed zero. There is no relative/absolute tolerance. Cases cover ncol3/5, n_in256/512/768/4096/8192/12288, row tails1/15/16/17/67, arbitrary packed six-bit scales/mins and extrema, varied finite FP16 coefficients/scales including zero and a negative global coefficient, signedINT8 activation extrema, nondefault activation/output strides, biased/unbiased alpha, and output padding canaries. Activation stride padding is nonzero poison. Each q8FP16 sum is deliberately unrelated to its integer sum.

`--bench` repeats the exactness gate first, then times full8192-output-row matrices with input widths4096/8192/12288 and ncol3/5. Its distinct physical weight-address ring exceeds twice the reported L2 capacity. The engine/trial/trial/engine order reports both controls; allocations, input copies and warmup are outside timing. This avoids benchmarking one small L2-resident matrix repeatedly. Timing is CUDA event time for synthetic matrices, not an end-to-end prediction or controlled GPU clock experiment. Retain ptxas register/spill/shared reports. Stop on any mismatch; do not widen tolerances. A compute-sanitizer run can check illegal accesses before timing if needed.

Only after this gate passes and the complete tile beats the actual DP4A kernel should the parent consider optional target integration and the existing forced full-logit/partial-commit/reset/pressure gate. Require useful clean full-round savings with identical histories, route/placement/payload accounting and acceptance; a faster isolated tile alone is insufficient.
