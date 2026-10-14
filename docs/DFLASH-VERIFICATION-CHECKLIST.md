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
- [x] Measure matched-capacity async clean A/B: no convincing verify-round gain.
- [x] Screen narrow assignment swaps; defer based on small optimistic byte savings.
- [x] Implement exact-order HC phase-batching candidate and engagement counters.
- [ ] Qualify HC and combined kernels; whole-model mismatches require isolation.
- [ ] Resolve or bound existing disk-pressure non-reproducibility with ownership diagnostics.
- [ ] Measure refinements and async scheduling combinations.
- [ ] Investigate supporting combinations when justified by evidence.
- [ ] Confirm retained changes on coding/writing, prefill and longer occupied context.
- [ ] Check cancellation/recovery and clean teardown.
- [ ] Commit report, results table, limitations and recommended settings.

Current stage: HC-only and HC/paired whole-model comparisons differ, while the current-binary unchanged control matches exactly. Isolated arithmetic and one-window immutable-weight diagnostics are next. Async overlap did not materially reduce verification round time. No substantial performance gain claimed; all changes remain opt-in research.
