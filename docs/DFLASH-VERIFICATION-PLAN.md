# Efficient verification: long-term plan

## Objective and baseline

Make complete multi-token target verification cheaper, without expert skipping, extra quantization, weakened correctness thresholds or ordinary-decode/prefill regressions. Keep current GSQ-RCO 3.0bit Q4-attention model, c=1 and 128K capacity. Capacity is not occupied context. Primary baseline is checkpoint c730fbb (validated scratch/reuse implementation), not the old overflowing path. Working branch: research/dflash-batched-verification. Daily configuration unchanged; no external publication.

The previous best narrow-draft sample was ~42 TPS; verification ~63 ms plus drafting ~2 ms produced ~2.78 tokens/round. 60 TPS requires ~46 ms total at that acceptance. This is a target, not a promise. Controls have varied materially with foreground load.

## Decision sequence

1. Profile widths 2 and 4 on one short coding task. Separate routing/fetch, expert arithmetic, CPU waits, attention/dense and host time. Measure actual expert-group sizes and quantify weight-reuse opportunity before choosing kernels.
2. Audit existing grouped/prefill machinery. Prefer reusing proven quantized arithmetic. Prototype complete gate/up, activation quantization, down and combine handling, with singleton fallback and mixed-format support. No claim of useful batching from a flag or grouping alone.
3. Validate controlled-backend full logits, partial commits, eviction/reset/repeat; distinguish new arithmetic differences from CPU placement. Require actual mechanism engagement.
4. Run small end-to-end A/B tests and targeted ablations; verify cost savings survive CPU work and transfers. Reject kernel-only gains that worsen round time.
5. Expand only where evidence warrants: transfer readiness, CPU merged rows, wider group scope, dense/attention batching, tensor-core paths if sufficient work exists. Preserve ordinary path and fallback.
6. Qualify retained combination on coding and writing, prefill, cancellation, and one longer occupied context. No exhaustive sweep. Report remaining bottlenecks and limitations.

## Parallel research

GPU-design subagent: inspect group kernels, supported quant arithmetic, reuse and plausible complete prototypes. Scheduling subagent: route/fetch/CPU readiness, useful group availability, critical-path instrumentation. Only primary agent runs GPU experiments. Keep sources and measurements separate from hypotheses.

## Checkpoints and evidence

Small commits per meaningful implementation/qualification. Separate diagnostic profiling from unprofiled TPS. Store environment, hashes, context occupancy, output tokens, acceptance, round times and memory footprint. Preserve prior raw-logit files locally. Update CHECKLIST.md and work log after each decision. Do not stop at the first kernel gain or broaden to unrelated micro-optimizations.

## Initial source finding

Existing window grouping already identifies shared GPU pointers, but gate/up loops members serially; down assigns members to warps. Existing prefill rows_multi helpers may supply reusable quantized weight decoding. Their format coverage and arithmetic must be audited before reuse. This is a concrete direction, not yet a measured gain.
