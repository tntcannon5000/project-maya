# DFlash investigation: results

2026-10-14. Branch `research/dflash-verification`. Implementation checkpoint `35efea2`. Local experiments only; daily settings unchanged.

## Verdict

A real verification-path problem was found and bypassed with validated opt-in scratch ownership: the old eight-slot scratch pool overflowed, causing whole verification windows to run and then rerun token by token. Wider drafting improved from 21.65 to 35.52 TPS in the selected comparison. This is not a demonstrated large gain over ordinary decoding (~42 TPS in the faster controls). High acceptance alone does not make verification cheap.

## Engine comparisons

Same GSQ-RCO 3.0bit Q4-attention target, c=1, 131072 allocated context, short occupied coding prompt, greedy 192-token output. Maya uses identical packed weights and Q4_K_M DFlash2. Independent engines use the original GGUF and different execution/cache backends. These are small viability tests, not confidence intervals or long-context qualification.

| Engine/path | No draft TPS | 2 drafts TPS | 4 drafts TPS |
|---|---:|---:|---:|
| Stock Maya 1.1.7 | 28.46 | 19.44 | 14.15 |
| Stock Maya 1.1.9 | 28.77 | 24.58 | 16.71 |
| Saved retained executable | 42.48 | 38.81 | 21.65 |
| New scratch/reuse, staggered | 34.67 | 32.08 | 30.49 |
| New scratch/reuse, full-layer hybrid | 41.82 | 42.39 | 35.52 |
| Full-layer GPU-only | 28.66 | 31.17 | 28.24 |
| Full-layer + CPU batching, hybrid | 33.70 | 39.04 | 31.83 |
| Full-layer + CPU batching, RAM misses CPU | 34.28 | 36.24 | 34.99 |
| Final hybrid CPU-batch lifecycle smoke | 35.14 | 34.14 | cancelled |

Controls varied substantially. OBS was present during parts of testing; causality is not established. Use within-load controls; do not interpret small differences or maxima as reliable wins. Free-generation histories can differ. Lightweight round counters were enabled, without Nsight.

Independent neurall llama.cpp `9d35489`: warmed no-draft 26.36 TPS; MTP1 21.22 TPS despite 47/49 drafts accepted (95.9%). Adaptive drafting did not propose every step. DFlash loaded but failed the first request with `t_layer_inp[il] != nullptr`: the GLM graph did not expose required layer inputs. DS4 `4daa992` rejected the same target before inference, expecting 46 blocks instead of 45. Neither failed configuration supplies a DFlash/MTP speed comparison. Stock Maya 1.1.9's relevant GGUF verification code is unchanged from 1.1.7; its speed difference is not evidence of a verified code improvement.

## Implemented mechanisms and engagement

- `MAYA_WIN_SCRATCH=64`: reserve capacity before expert-pool sizing and give each token eight disjoint slots when sufficient capacity exists. No retries in qualified candidate runs. Scratch rises from 108 to 864 MiB (+756 MiB); observed permanent expert slots fall from 2618 to 2523. Defaults remain unchanged; an undersized scratch configuration still uses the legacy fallback.
- `MAYA_WIN_REUSE=1`: reuse completed earlier expert copies within the same layer/group, without changing permanent ownership. Full-layer hybrid reused 58/489 GPU rows at widths 2/4; GPU-only reused 555/1812 but was slower overall.
- Existing `STRATA_GLM_WIN_STAGGER=0`: full-layer grouping improves sharing scope, at the expense of CPU overlap. This supplied the best absolute speculative result in this screen.
- `MAYA_WIN_SINGLE_NODE_BATCH=1`: reuse upstream merged CPU expert-row processing on a single node, using two half-sized unpinned pools for verification. Ordinary decoding retains its original pool. Hybrid runs reduced 11929 entries to 11540 unique reads and 15664 to 14760. Functionality is proven; a robust throughput advantage is not. Keep experimental, not a daily default.

Reproduce the final smoke with `tools/bench_dflash_matrix.py --out <new-directory> --quant Q4_K_M --widths 2 --env MAYA_WIN_SCRATCH=64 --env MAYA_WIN_REUSE=1 --env STRATA_GLM_WIN_STAGGER=0 --env MAYA_WIN_SINGLE_NODE_BATCH=1 --env STRATA_GLM_WIN_ALLCPU=0 --cancel-smoke`. Exact commands, environment, binary and drafter hashes are in each manifest.

## Numerical and lifecycle validation

Each full-logit comparison covers 28,188,160 values (182 rows), alternating verification windows of 5/8 tokens, partial commits, reset/repeat and eviction pressure with a 16 GiB expert pool. The actual model exercises Q2_K/Q3_K/Q4_K experts.

| Controlled comparison | Result |
|---|---|
| Expanded scratch, reuse on vs off | Bit-exact all logits |
| Expanded scratch + reuse vs retry-free token path | Bit-exact all logits |
| CPU merged vs serial, fixed CPU placement | Bit-exact all logits |
| Legacy overflow/retry vs retry-free token path | Max absolute difference 6.574832; 181/182 greedy rows match |

The legacy discrepancy is isolated to that path; its precise restore/retry failure mechanism is not fully traced. The new path avoids it. GPU/CPU placement differences are not claimed to be universally bit-exact. Native CPU row tests also passed. Raw logits remain local and are excluded from Git; summaries are in `parity/ablations.json` and `parity/cpu-result.json`. The older `parity/result.json` records the legacy mismatch, not a failure of the new controlled comparisons.

Final smoke successfully cancelled a width-4 request, recovered with a width-2 request (32 tokens, 33.03 TPS), and exited cleanly. No inference engine remained running. Prefill was recorded per request but this short screen does not establish broad prefill or occupied-128K performance parity.

## Why the huge gain has not appeared

At the best narrow-draft sample, 123/138 drafts were accepted (89.1%), about 1.78 accepted drafts per round. Verification still costs roughly 63 ms plus ~2 ms drafting for about 2.78 output tokens. To reach 60 TPS, that whole round must take about 46 ms. Avoiding retries removes waste but does not make multiple target-token computations cost one token.

The GPU window gate/up implementation still performs per-member dot products; grouping is not a tensor-core GEMM verification implementation. CPU merging saves few expert reads in realistic hybrid routing, despite greater reuse under fixed all-CPU placement. Future substantial gains require cheaper complete target verification and useful weight reuse, not merely a larger or higher-quality drafter. GPU-only placement and unconditional all-CPU misses did not solve this. Q5/Q8 are not retested here: the earlier 21-cell quant screen is preserved, and this investigation isolates execution-path defects using Q4.

Retain the validated scratch/reuse implementation and optional CPU-batching research path. Do not enable them universally or claim 60 TPS. The strongest result is a repaired wide-verification path, with clear remaining costs and rollback checkpoints.
