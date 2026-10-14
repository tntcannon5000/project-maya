# DFlash2 quantization and draft-width screen — 2026-10-14

## Scope and method

- Maya v1.1.7 retained fork, branch `research/dflash-quant-width`; source checkpoint `2940ce7`.
- Same GSQ-RCO 3.0bit Q4-attention target pack, one RTX 5090, c=1, 131072-token allocated context. The prompt is short; this is not 128K occupied-context performance.
- One FastAPI dashboard coding prompt, greedy generation, 192 output tokens per request. One warmup and one no-draft control per drafter load, then seven widths in order 4,1,7,2,6,3,5.
- `spec=N+1` includes the anchor; confidence cutoff and cumulative cutoff disabled, minimum drafts N. Per-round records confirm each requested width engaged.
- Main-model MTP, suffix drafting and KV streaming are off. These are DFlash2 tests, not separate MTP or K4V tests.
- Lightweight upstream per-round statistics and timing enabled; no Nsight profiler. TPS is DONE output tokens / decode wall time, excluding prompt processing. No independent profiler-free confirmation was added to this small screen.
- Same retained settings and requested 22 GiB expert pool. Automatic VRAM sizing means larger drafters reduce actual expert slots: a realistic deployment comparison, not an isolated precision-only ablation.
- Single sample per cell, sequential repeated requests, no confidence intervals. Cache history/order, foreground activity, and hybrid CPU/GPU numerical differences can affect results. Output histories were recorded and do differ; acceptance is not measured against identical forced routes.

## Results

Cells: decode TPS / accepted draft percentage. Acceptance is accepted prefix tokens divided by verified draft tokens.

| Draft tokens | Q4_K_M | Q5_K_M | Q8_0 |
|---:|---:|---:|---:|
| 1 | 42.22 / 93.9% | 40.92 / 90.1% | 41.38 / 92.0% |
| 2 | 44.64 / 89.1% | 41.38 / 85.2% | 41.72 / 83.3% |
| 3 | 25.87 / 77.0% | 25.30 / 78.9% | 26.92 / 81.0% |
| 4 | 24.28 / 77.1% | 20.77 / 66.0% | 23.69 / 75.0% |
| 5 | 21.25 / 69.3% | 21.41 / 71.4% | 20.71 / 67.3% |
| 6 | 19.45 / 61.4% | 16.51 / 50.3% | 15.85 / 48.3% |
| 7 | 19.77 / 64.1% | 18.80 / 61.9% | 19.58 / 64.1% |

## Loaded controls and memory

| Drafter | File GB | Drafter CUDA MB | Expert pool GiB / slots | No-draft TPS | Best draft width / TPS |
|---|---:|---:|---|---:|---|
| Q4_K_M | 0.697 | 717 | 21.37 / 2618 | 45.83 | 2 / 44.64 |
| Q5_K_M | 0.830 | 843 | 21.25 / 2602 | 44.17 | 2 / 41.38 |
| Q8_0 | 1.254 | 1248 | 20.85 / 2551 | 45.32 | 2 / 41.72 |

## Verification costs

Mean per-round timing from upstream counters. Draft and verification costs below are diagnostic wall times, not separate CUDA kernel timings.

| Quant | Width | Rounds | Draft ms/round | Verify ms/round | Mean accepted/round | First difference from loaded no-draft control (0-based) |
|---|---:|---:|---:|---:|---:|---:|
| Q4_K_M | 1 | 99 | 1.89 | 43.78 | 0.94 | 37 |
| Q4_K_M | 2 | 69 | 1.97 | 59.96 | 1.78 | 170 |
| Q4_K_M | 3 | 58 | 2.02 | 125.25 | 2.31 | none |
| Q4_K_M | 4 | 47 | 1.80 | 165.42 | 3.09 | 170 |
| Q4_K_M | 5 | 43 | 1.80 | 207.13 | 3.47 | none |
| Q4_K_M | 6 | 41 | 1.82 | 237.48 | 3.68 | none |
| Q4_K_M | 7 | 35 | 1.84 | 273.83 | 4.49 | 170 |
| Q5_K_M | 1 | 101 | 1.96 | 44.17 | 0.90 | 98 |
| Q5_K_M | 2 | 71 | 2.21 | 62.59 | 1.70 | 71 |
| Q5_K_M | 3 | 57 | 2.42 | 130.03 | 2.37 | 37 |
| Q5_K_M | 4 | 53 | 2.56 | 170.59 | 2.64 | 71 |
| Q5_K_M | 5 | 42 | 2.73 | 209.27 | 3.57 | 37 |
| Q5_K_M | 6 | 48 | 2.88 | 237.45 | 3.02 | 71 |
| Q5_K_M | 7 | 36 | 3.11 | 278.89 | 4.33 | 37 |
| Q8_0 | 1 | 100 | 1.98 | 44.12 | 0.92 | 37 |
| Q8_0 | 2 | 72 | 2.25 | 61.10 | 1.67 | 37 |
| Q8_0 | 3 | 56 | 2.30 | 124.41 | 2.43 | 170 |
| Q8_0 | 4 | 48 | 2.19 | 165.63 | 3.00 | none |
| Q8_0 | 5 | 44 | 2.18 | 207.12 | 3.36 | 170 |
| Q8_0 | 6 | 50 | 2.21 | 238.20 | 2.90 | 37 |
| Q8_0 | 7 | 35 | 2.27 | 275.98 | 4.49 | none |

## Implementation and validation

- Downloaded all three drafters from the pinned HF revision in `download-manifest.json`; SHA256 checked against LFS hashes. Tensor headers confirm Q4_K/Q6_K, Q5_K/Q6_K and Q8_0, respectively.
- Added Q5_K selector-table reads using the existing GGML block layout and affine dequantization formula. Main matrix-vector kernels already supported Q5_K. Q4/Q8 arithmetic paths are unchanged.
- Q5 existing GPU-versus-CPU drafter reference: PASS, minimum hidden cosine 0.992372, 15/112 candidate differences, 2/7 draft-pick differences under the existing thresholds. This is approximate agreement, not bit-exact parity; no thresholds were weakened.
- All 21 matrix cells completed; actual draft widths checked; all engines exited normally. No persistent daily launcher settings changed. Pre-change executable retained as `build/strata-before-q5-selector.exe`.

## Decision

Use the measured table rather than assuming acceptance increases monotonically with precision. Wider batches are dominated by target verification cost on this hybrid single-GPU setup. This screen does not establish a reliable drafting speedup over its loaded no-draft controls. Leave the daily preset unchanged. These free-generation samples are not a new numerical parity qualification.

All three engine logs report a verification scratch-slot overflow during the first width-4 test. In `src/core/glm_window.cu`, `window()` detects the sticky overflow flag after the batched pass, restores recurrent state, and reruns that window token by token. The warning prints only once per process, so these logs establish engagement but do not count all retries or attribute their total cost. The per-token flag resets for each subsequent window. This is a concrete engine bottleneck, not evidence that larger drafters or DFlash2 are intrinsically unhelpful. Fixing and measuring scratch capacity is a separate follow-up; this matrix deliberately retains the current scheduling implementation.

Raw manifests, token outputs, round records, engine logs and GPU telemetry are alongside this report. RAM-tier sizing and actual expert slots are recorded in each engine log.
