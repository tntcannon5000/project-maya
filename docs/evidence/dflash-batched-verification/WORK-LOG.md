# Batched verification work log

## 2026-10-14 — baseline profile

Parent c730fbb; plan checkpoint 22d03c0. Full-layer groups, scratch64, reuse enabled, single-node CPU batching off. Q4 drafter, 128K capacity, 40-token coding prompt, 192 output tokens. Profiled control 41.49 TPS; width2 41.73; width4 35.38. No retries.

| Per-window CUDA event interval (ms) | Width2 | Width4 |
|---|---:|---:|
| Full verify GPU timeline | 62.81 | 99.75 |
| Routing/wait/fetch combined | 33.00 | 56.60 |
| Expert computation plus shared down/group/combine | 9.33 | 12.72 |
| Exposed CPU answer waits | 3.37 | 5.14 |
| Other stages | 17.11 | 25.29 |

Host enqueue overlaps GPU work and is not additive. CUDA intervals can include host submission gaps. This is diagnostic, not unprofiled throughput. Routing/fetch accounts for more time than expert arithmetic; even eliminating all 9.33ms would not alone meet the ~19ms saving sought at width2. Therefore complete batching must address transfer/route work, not merely arithmetic. Finer counters/events are being implemented before deciding the next supporting mechanism.

Prototype direction: exact-order DP4A weight decode reused across tokens, mixed Q2_K/Q3_K/Q4_K, fallback preserved. Existing prefill multirow helpers do not directly cover all required formats; no tensor-core speedup assumed.
