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
- [ ] Measure refinements and async scheduling combinations.
- [ ] Investigate supporting combinations when justified by evidence.
- [ ] Confirm retained changes on coding/writing, prefill and longer occupied context.
- [ ] Check cancellation/recovery and clean teardown.
- [ ] Commit report, results table, limitations and recommended settings.

Current stage: paired refinement (mode 2) also passed 28,188,160 exact logits; clean TPS measurement underway. Async-fetch pressure qualification failed and remains experimental. Next isolate serial pressure reproducibility before attributing all differences to overlap. No new performance gain claimed.
