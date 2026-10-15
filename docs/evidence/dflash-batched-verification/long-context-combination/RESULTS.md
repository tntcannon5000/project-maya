# Longer-context combined qualification

One coding task containing four actual Python toolkit files: **6,793 occupied prompt tokens**, 128K allocated capacity, 128 output tokens. Same binary, packed model, 94 GiB RAM tier, requested 22 GiB expert pool and staging64. Corrected control disables grouped/HC/component/exact-column experiments; candidate enables paired groups, HC batching, component streaming and exact columns. No speculative token skipping or altered quantization.

| Measurement | Corrected control | Combined candidate |
|---|---:|---:|
| Cold prefill ms |3224.4|3212.7|
| Ordinary decode TPS |36.27|36.75|
| Forced two-draft TPS |29.27|29.88|
| Forced four-draft TPS |19.39|19.62|
| Two-draft verification ms/round |64.515|63.168|
| Four-draft verification ms/round |108.171|106.814|
| Two-draft acceptance |63/128|63/128|
| Four-draft acceptance |71/236|71/236|

All corresponding warmup, ordinary, forced-two and forced-four output IDs match across configurations. Each speculative policy produces different output from ordinary decoding, so its TPS is not an identical-history speed comparison against ordinary. The 1–2% between-configuration differences remain small, with ordinary decode itself varying by1.3%. A single cold-prefill pair is a bounded smoke check, not statistical no-regression proof or full128K qualification.

## Numerical and lifecycle checks

A separate controlled-GPU fixed-input comparison uses the same6793-token prefix, real batched prefill followed by the final prompt token, T3/T5 windows, partial commits and two resets/repeats. All **7,434,240 logits /48rows match exactly**, with no nonfinite values. CPU placement effects are deliberately excluded from that arithmetic comparison. The initial serial-prefix diagnostic was intentionally interrupted before any logit rows; see parity-control/INTERRUPTED.md. It is not counted as a numerical failure or successful test.

Both serving runs passed STOP, a subsequent generation and clean teardown. The candidate adds a confidence-policy request before cancellation, so its cancellation/recovery histories and cache sequence differ from control; those are lifecycle smoke checks, not parity/performance comparisons.

## Existing confidence policy

Candidate additionally used upstream's single-GPU rule: cutoff0.95, minimum0 drafts, maximum4, no cumulative cutoff. It completed at37.65TPS with42/42 accepted drafts across86rounds:53zero-draft,26one-draft,5two-draft and2three-draft rounds. This demonstrates policy engagement and avoidance of many low-confidence verification windows. Its output differs from ordinary at the first token, so do not claim a matched speedup, attribute that rate to our arithmetic changes, or call the hybrid discrepancy resolved. Forced-width screens intentionally disabled these existing safeguards to control window size; they are not optimal daily-policy recommendations.

## Decision

The combined implementation remains opt-in research. Longer-context arithmetic and bounded lifecycle/prefill checks pass, but substantial speed gains remain unproven. Do not run a broad workload suite for these small changes. Raw logits remain local; compact results and manifests are checkpointed. Next source investigation targets a genuinely different transfer mechanism rather than more small kernel variants.
