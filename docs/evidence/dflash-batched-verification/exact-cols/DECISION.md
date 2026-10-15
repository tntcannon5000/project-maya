# Exact-column verification result

The opt-in `MAYA_MV_EXACT_COLS=1` removes padded column computations while preserving native arithmetic. Both off/on configurations passed 587 native-quant and heterogeneous-column cases, plus 240 independent Q4 cases each. Full-model fixed-input T3/T5 windows, partial commits and repeated requests passed **7,434,240 exact logits / 48 rows**.

| Short coding run | Disabled | Enabled |
|---|---:|---:|
| Ordinary decode TPS |44.32|44.66|
| Four-draft TPS |47.82|48.32|
| Verification ms/round |86.393|85.417|
| Accepted/offered drafts |97/120|97/120|

Same binary, model, 94 GiB RAM allocation, requested 22 GiB GPU expert pool, 128K allocated context, 40 occupied prompt tokens and 128 output tokens. Component streaming is disabled in both; HC batching and paired experts are enabled. All output IDs match across corresponding warmup, ordinary and speculative requests. These use round counters without event-stage profiling.

Throughput differs by about 1.05%, while ordinary decode differs by 0.77%. Verification differs by 0.976 ms/round. The isolated larger five-column Q4 microbenchmarks improved around 14–16%, but their shape distribution and share of total cost prevent extrapolating that percentage to complete inference.

Decision: retain as opt-in research, not a recommended default or a robust end-to-end gain. Do not expand into a full workload suite for this small result. No claim of 60 TPS. Broader prefill, occupied long-context, lifecycle and non-NVIDIA qualification has not been performed for this flag. The controlled GPU comparison establishes exact arithmetic; it does not erase previously documented hybrid placement differences.

The candidate and component streaming each show roughly 1% contemporary gains separately. That is not enough evidence to justify a broad combination sweep; their savings are not presumed additive. The next investigation must establish a materially larger critical-path opportunity before further implementation.
