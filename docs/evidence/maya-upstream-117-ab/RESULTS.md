# Maya v1.1.7 versus retained fork at 192K

2026-10-13. Upstream `1e225b9`; retained implementation `57b4d14`, all latest experimental switches off. No engine implementation changed for this comparison. Upstream checkout: `../project-maya-upstream-v1.1.7`. Both full upstream engine and common benchmark driver built successfully.

## Method

c=1, 196608-token allocated context, same GSQ-RCO 3.0bit Q4-attention packed weights/profile, fixed CPU plan 011223344, requested 22 GiB GPU pool, 101 GiB RAM budget. Both actually fitted 2674 GPU slots / 21.837 GiB. Upstream retained its normal exclusive tier ownership; our fork enabled its retained RAM shadow and scheduling configuration. KV streaming and all speculative token decoding disabled. This is a configuration A/B, not an ablation of one change.

Two workloads: 30-token prompt and 8192-token occupied source-code prompt. Each scores 512 forced-input tokens from the first 512 tokenizer tokens of serve/server.py at 57b4d14, twice after tier warming. This is a code replay fixture, not independently generated output or a task-quality benchmark. Occupancy grows by 512; neither fills the 192K window. Each pair has identical token-file and packed-weight hashes. Fixed history does not imply identical floating-point routes across implementations.

Order: upstream short, retained short, retained 8K, upstream 8K. TPS is 512 divided by summed forward() latency; logits writes are outside timed regions. No Nsight profiling. Normal tier diagnostics are enabled equally. First means first scored request after loading/warming, not cold disk startup. Only two requests per condition: no confidence intervals and no broad workload claim.

## Decode

| Occupied prompt | Request | Upstream TPS | Retained TPS | Retained gain |
|---|---|---:|---:|---:|
| 30 | First | 30.22 | 40.43 | 33.8% |
| 30 | Repeat | 30.38 | 42.54 | 40.0% |
| 8192 | First | 27.25 | 36.11 | 32.5% |
| 8192 | Repeat | 27.49 | 38.14 | 38.7% |

## 8K prefill

| Request | Upstream seconds | Retained seconds |
|---|---:|---:|
| 1 | 3.512 | 3.610 |
| 2 | 3.505 | 3.564 |

## Capacity and limits

| Build/workload | RAM tier GiB | Minimum free system RAM GiB | Peak GPU MiB | Mean disk reads/token (repeat) |
|---|---:|---:|---:|---:|
| upstream-short | 87.84 | 18.84 | 31532 | 0.000 |
| retained-short | 97.82 | 10.30 | 31524 | 0.000 |
| retained-8k | 97.82 | 10.31 | 31602 | 0.000 |
| upstream-8k | 87.84 | 20.53 | 31594 | 0.000 |

Our retained RAM backing is about 10 GiB larger; the GPU pool matches. This comparison includes that deliberate memory/performance tradeoff. VRAM is nearly full: 192K fits these controlled runs, but this is not foreground-use headroom qualification. CPU-expert and residency counts are recorded in summary.json; custom transfer counters are zero placeholders in the shared stock-compatible driver and must not be interpreted as zero traffic.

All calls completed with finite logits; repeated requests and process teardown succeeded. Numerical parity has not been qualified against new upstream, and no per-feature speed attribution is claimed. Current results support preserving the retained work while integrating upstream, not immediate PR sign-off. Raw logits stay local and ignored; manifests, timing CSVs, hardware logs and reproducible drivers are retained. No external publication.
