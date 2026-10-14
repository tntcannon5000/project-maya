# Maya v1.1.7 integration qualification

- Date: 14 October 2026. Upstream release `1e225b9`, integration `736ac46`, pre-merge retained source `80d4abb`.
- Same GSQ-RCO 3.0bit Q4-attention pack/profile and retained switches; no MTP, K4V, token speculation or concurrent-graph experiment.
- c=1: 192K allocated capacity, short/8192-token occupied prompts, 256 forced continuation tokens, two requests. Fixed token/weight hashes verified. TPS is tokens / summed forward latency; logits writes are outside timed regions, no Nsight.
- Run order: before short, merged short, merged 8K, before 8K. GPU-only runs are correctness diagnostics, not headline TPS.
- This is a bounded non-regression screen, not a guarantee on all workloads or hardware. Historical TPS is not substituted for the fresh baseline.

| Context | Request | Before TPS | Merged TPS | Change | Before prefill ms | Merged prefill ms |
|---|---|---:|---:|---:|---:|---:|
| short | 1 | 27.11 | 27.39 | +1.04% | 1343.3 | 1334.2 |
| short | 2 | 30.03 | 29.49 | -1.79% | 1228.4 | 1088.5 |
| 8k | 1 | 25.13 | 24.68 | -1.78% | 11118.5 | 10675.7 |
| 8k | 2 | 26.00 | 26.01 | +0.05% | 10390.2 | 10292.5 |

## Reversed-order short confirmation

| Request | Before TPS | Merged TPS | Change |
|---|---:|---:|---:|
| 1 | 28.76 | 29.49 | +2.55% |
| 2 | 31.36 | 30.87 | -1.59% |

## Capacity-matched short check

| Request | Before TPS | Merged TPS | Change |
|---|---:|---:|---:|
| 1 | 28.76 | 29.77 | +3.52% |
| 2 | 31.36 | 32.23 | +2.76% |

- This ablation requests 97.822 GiB instead of 101 GiB to approximately match the old binary physical RAM tier; per-class slot counts still differ, so this is a capacity sensitivity check, not an exact placement control or portable default.

## Numerical checks

| Comparison | Logits | Exact | Max absolute difference | Matching argmax rows |
|---|---:|---|---:|---:|
| before-short -> merged-short | 79,298,560 | False | 2.92651 | 492/512 |
| before-8k -> merged-8k | 79,298,560 | False | 6.88861 | 512/512 |
| gpu-before -> gpu-merged | 39,649,280 | False | 7.51957 | 256/256 |
| gpu-before -> gpu-merged-glu-off | 39,649,280 | False | 7.69527 | 256/256 |
| gpu-stock117 -> gpu-merged | 39,649,280 | False | 8.02515 | 256/256 |
| gpu-short-before -> gpu-short-merged | 39,649,280 | True | 0 | 256/256 |
| confirm-before -> matched-capacity | 79,298,560 | False | 3.22037 | 496/512 |

### Same-binary GPU-only repeat controls

| Binary/fixture | Repeat exact | Max difference |
|---|---|---:|
| gpu-before | False | 7.39745 |
| gpu-merged | False | 7.11752 |
| gpu-merged-glu-off | False | 7.82686 |
| gpu-stock117 | False | 7.78646 |
| gpu-short-before | True | 0 |
| gpu-short-merged | True | 0 |

- Hybrid comparisons can change CPU/GPU assignments; a fixed input history alone does not establish identical routing or arithmetic. No thresholds are relaxed.
- The first 73 short-fixture rows matched exactly. The first differing row also changed hit/fetch/CPU-expert counts. GPU-only comparison isolates backend placement; it does not make hybrid outputs exact.

## Memory and execution

| Run | RAM tier GiB | GPU pool GiB | Slots | Minimum free RAM GiB |
|---|---:|---:|---:|---:|
| before-short | 97.23 | 21.84 | 2674 | 7.06 |
| merged-short | 98.71 | 21.84 | 2674 | 8.34 |
| before-8k | 97.82 | 21.84 | 2674 | 9.34 |
| merged-8k | 98.71 | 21.84 | 2674 | 8.29 |

- Upstream pinning-limit sizing can allocate more RAM backing despite identical requested budgets. This is a real integration effect and a capacity confound, not a per-kernel attribution.
- Transfer-byte fields in this stock-compatible driver are placeholders and must not be read as zero traffic.

## Concurrent server smoke/performance

| Build | First aggregate TPS | Repeat aggregate TPS |
|---|---:|---:|
| before | 39.82 | 41.47 |
| merged | 39.25 | 43.04 |

- c=4 uses four real coding generations at 32K per slot, 256 output tokens each, two cohorts; aggregate TPS covers the interval all four lanes emit. It is not comparable directly with c=1 replay or proof of numerical parity.
- Focused native split/rows exact CPU tests passed; lifecycle and GPU ownership results are recorded in their logs.
- Full hardware audit and PR-ready work bullets: ../../PR-WORK-LOG.md. Raw logits remain local and ignored.

## Verdict

- Integration is built and checkpointed. No broad throughput loss is established: the normal-budget short warm case is 1.6-1.8% lower in two pairs, the repeated 8K case is flat, and repeated c=4 is 3.8% higher. Do not claim literally zero loss at every setting.
- At approximately matched RAM capacity, short warm decode is 32.23 TPS versus 31.37 TPS for the confirmed old binary (+2.8%). This supports memory/placement sensitivity; it is not proof of a universally faster allocator or code path. The ordinary 101 GiB preset remains available; the 97.822 GiB setting is a local ablation, not a new default.
- Short serial-prefill GPU-only decode is exact across 39,649,280 logits. Long batched-prefill repeat controls differ even within the old and stock upstream binaries; exact long-prefill parity remains unresolved. Disabling upstream fused SwiGLU did not resolve it.
- Merged lifecycle: warmup, STOP after eight tokens, recovery and QUIT passed. Native split and multi-row exact tests passed. Ownership checks passed for live owners and sampled weight bytes per mixed quant class.
- Prepared an ignored local preset, maya-retained-v117.local.json, with the retained settings and original 128K capacity; no server is left running. This is not a new product-wide default.
