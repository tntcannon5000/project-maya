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
- [x] Bound the demonstrated pressure failure after deferred-demotion fix (six-window exact pressure replay; not exhaustive stress).
- [ ] Measure refinements and async scheduling combinations.
- [ ] Investigate supporting combinations when justified by evidence.
- [x] Screen moderate CPU allocation with merged rows; reject performance recommendation.
- [x] Implement and qualify full-layer component fetch with grouped GU overlap (GPU exact and bounded disk pressure).
- [x] Verify component byte accounting and actual GPU overlap before judging speed; narrow confirmation shows only 1.2% TPS difference.
- [ ] Confirm retained changes on coding/writing, prefill and longer occupied context.
- [x] Check cancellation/recovery and clean teardown on corrected HC/paired writing case.
- [ ] Commit report, results table, limitations and recommended settings.

Current stage: corrected ownership fix and component/narrow-column arithmetic have controlled exact qualification. Corrected placement ceiling and complete exact IMMA feasibility have been measured; neither justifies a substantial new implementation. Exact-column coding improves only about 1% in a short matched sample. Performance flags remain opt-in; no robust new default gain. Next work must identify a materially larger opportunity before broad qualification or more combinations.

- [x] Capture corrected initial residency, RAM readiness, backend masks and copy owners.
- [x] Validate relaxed fixed-CPU-budget placement ceiling with metadata checks and exhaustive toy cases.
- [x] Implement complete isolated exact Q4_K IMMA tile and pass246 strict numerical cases.
- [x] Test actual model shapes; reject production integration for immaterial estimated savings.

- [x] Implement exact three/five-column DP4A without padded arithmetic; preserve mixed-job geometry.
- [x] Pass 587 strict kernel cases per configuration and 7,434,240 full-model exact logits.
- [x] Complete contemporary four-draft coding comparison; retain opt-in, no robust default speed claim.

- [x] Qualify full combination after 6,793 occupied tokens: 7,434,240 exact GPU logits.
- [x] Record longer coding cold-prefill/ordinary/forced-width/lifecycle comparison.
- [x] Audit transient mapped-RAM arithmetic ownership and prior experiments.
- [x] Measure complete direct-RAM arithmetic versus staged arithmetic, with exact outputs and CPU co-execution: reject substantial slowdown across all quant classes.
- [x] Apply direct-RAM integration gate: no headroom demonstrated, no serving integration.

- [x] Screen cross-window transient scratch retention with capacity/live-workspace accounting; defer serving integration pending a practical policy with material headroom.

- [x] Screen causal scratch policies with six causal/capacity/ownership checks; no practical large saving demonstrated.
- [x] Compare ordinary forward against verification first rows/partial commits on one occupied coding input: controlled GPU exact.
- [x] Attribute first hybrid divergence through actual route masks and a two-step cold-ranking ablation; counter and residency history change placement.
- [ ] Any future claim of full ordinary/hybrid equivalence requires consistent backend placement/quantization, not current GPU-only evidence.

- [x] Validate all gap-ending CUDA API correlations and explicitly test missing-correlation handling.
- [ ] Implement complete opt-in native MoE layer graphs with safe replay credits, capture cleanup and teardown.
- [ ] Verify graph engagement and exact GPU/matched-hybrid fixed-input logits with partial commits/reset.
- [ ] Gate further graph work on one bounded coding throughput comparison.
