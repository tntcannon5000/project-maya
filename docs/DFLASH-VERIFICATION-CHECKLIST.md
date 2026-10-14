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
- [x] Qualify HC/paired combination on corrected GPU control: 28,188,160 exact logits.
- [x] Localize first mismatch to stale RAM bytes, not HC arithmetic or GPU fetch corruption.
- [x] Identify missing deferred-demotion submission in batched verification.
- [x] Validate window submission fix against immutable pack, exact logits and bounded pressure replay.
- [ ] Resolve or bound existing disk-pressure non-reproducibility with ownership diagnostics.
- [ ] Measure refinements and async scheduling combinations.
- [ ] Investigate supporting combinations when justified by evidence.
- [x] Screen moderate CPU allocation with merged rows; reject performance recommendation.
- [x] Implement and qualify full-layer component fetch with grouped GU overlap (GPU exact and bounded disk pressure).
- [x] Verify component byte accounting and actual GPU overlap before judging speed; narrow confirmation shows only 1.2% TPS difference.
- [ ] Confirm retained changes on coding/writing, prefill and longer occupied context.
- [x] Check cancellation/recovery and clean teardown on corrected HC/paired writing case.
- [ ] Commit report, results table, limitations and recommended settings.

Current stage: missing deferred-demotion submission fixed. Immutable-pack audit passes 42 layers and HC/paired versus corrected control passes 28,188,160 exact logits. Corrected clean throughput comparison is underway. Pressure replay remains outstanding. Async overlap did not materially reduce verification round time. No substantial performance gain claimed; performance candidates remain opt-in research.
