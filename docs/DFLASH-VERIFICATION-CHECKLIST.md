# Efficient verification checklist

- [x] Preserve prior checkpoint and create isolated research branch.
- [x] Record objective, controls, qualification rules and decision sequence.
- [x] Delegate focused GPU and scheduling source investigations.
- [x] Collect width-2/4 diagnostic breakdown on retained baseline.
- [x] Measure group-size distribution, actual branch engagement and transfer deduplication.
- [x] Rank complete designs using measured phase costs, multiplicity and compiled resources.
- [x] Implement first complete grouped Q2/Q3/Q4 prototype with engagement counters.
- [x] Validate grouped arithmetic: mixed-format logits and window state bit-exact. Scheduling changes require separate validation.
- [x] Measure first grouped-kernel clean A/B: no material gain.
- [x] Qualify paired grouped refinement: exact logits, only small clean TPS differences.
- [x] Isolate async allocation: matched RAM/staging controls pass exact logits with all experts warmed.
- [x] Prove actual async fetch/route concurrency with four-window Nsight capture.
- [ ] Resolve or bound existing disk-pressure non-reproducibility with ownership diagnostics.
- [ ] Measure refinements and async scheduling combinations.
- [ ] Investigate supporting combinations when justified by evidence.
- [ ] Confirm retained changes on coding/writing, prefill and longer occupied context.
- [ ] Check cancellation/recovery and clean teardown.
- [ ] Commit report, results table, limitations and recommended settings.

Current stage: async all-memory parity and real overlap established; clean matched-capacity TPS comparison next. Disk-pressure serial control itself is non-reproducible and requires separate ownership investigation. No substantial performance gain claimed; all changes remain opt-in research.
