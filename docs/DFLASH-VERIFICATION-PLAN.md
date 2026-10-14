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

## Current decision gates (after paired-kernel checkpoint)

Correctness update: batched-window execution missed the retained `layers`
demotion submission hook. This explains stale RAM consumed after waits on old
events. The fixed immutable audit passes 1,599 expert copies across 42 layers.
Requalify corrected control versus batching candidates before using older
whole-model or TPS comparisons; prior exact repeats alone could share this bug.

- Grouped modes 1 and 2 pass controlled GPU exact-logit qualification. Mode 2's clean coding result is only slightly faster; retain experimental status until a supporting mechanism creates a convincing round-time gain.
- Async fetch passed matched-allocation all-memory exact parity and demonstrated real route overlap. Clean hybrid verification-round times barely improved; acceptance changed, so headline TPS differences are not scheduling gains. Keep experimental, do not sweep parameters.
- Serial disk-pressure control failed self-repeatability. Four bounded ownership checks found no mismatch among inspected pointers/five retained blobs; this does not resolve pressure qualification. Do not accept greedy-token agreement as numerical parity.
- Audit shadow-backed eviction identity/bytes under pressure before blaming arithmetic or relaxing thresholds. Fix a demonstrated ownership defect before relying on pressure results.
- Dense projections and head already pass ncol=T and use multi-column paths. Remaining arithmetic investigations must show substantial missing reuse or avoidable work in source and timings; a blanket GEMM rewrite is not justified.
- Narrow window assignment swaps screen at only 2.1%/3.5% optimistic byte savings, below 1.2% without promotion ambiguity. Defer that planner; broader union assignment remains untested.
- Implement and qualify exact HC3 phase batching across tokens (same 32-partial reductions, private scratch, no grid spin). If useful, test it with paired expert arithmetic before choosing a retained candidate. Then qualify one writing workload and longer occupied context; avoid a broad suite.

Possible later gate, not current implementation: a window-specific moderate CPU split paired with merged CPU rows. The current window uses the ordinary decode CPU plan or the already-tested all-CPU extreme; neither establishes the best multi-row makespan. Only pursue a bounded middle-point experiment after current arithmetic/ownership isolation. Account for shared RAM bandwidth and CPU tail, distinguish backend arithmetic changes from scheduling parity, and avoid an exhaustive quota sweep.
