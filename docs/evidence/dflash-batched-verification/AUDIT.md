# Verification campaign completion and remaining-work audit

2026-10-15. Read-only audit against `docs/DFLASH-VERIFICATION-PLAN.md`, its
checklist and saved evidence. The parent is running the longer-prefix
controlled-GPU combined-logit check; this audit does not presume its result.

**The substantial throughput objective is not achieved.** Completing kernels,
diagnostics and numerical tests is a useful R&D result, but does not establish
materially cheaper verification. No new default speed improvement or 60 TPS
claim is supported. The deferred-demotion repair is the clear retained
correctness result; performance flags remain research options.

## Required gates and present evidence

| Gate | Evidence / remaining action |
|---|---|
| Complete exact expert arithmetic and real engagement | Grouped/paired native Q2/Q3/Q4 paths, HC phases, component copies and exact-column DP4A are implemented and exercised. Dense projections already batch columns. No skipping/extra weight quantization was introduced. |
| Correct reference and ownership | Missing deferred-demotion submission fixed; immutable pack matches 1,599 copies/14.56GB. Corrected bounded pressure replay passes 6,040,320 exact logits. Pre-fix pressure failures are historical, not an unresolved current-control gate. |
| Full logits, partial commit, reset/repeat | HC/paired and component each have 28,188,160 corrected exact logits; exact columns have 7,434,240 controlled logits. **Finish the currently running combined long-prefix controlled-GPU comparison**, retaining exact count, first mismatch if any, manifests and engagement. A combined pass cannot be assumed from separate passes. |
| Whole-round attribution | Corrected phase attribution separates fetch/GU/down readiness and CPU tail; clean component confirmation is only about 1.2% TPS. Exact-column coding is about 1.05%. Neither supports adding isolated gains together. |
| One longer occupied coding task | New 6,793-token A/B covers fixed two/four drafts with identical corresponding output IDs. Rates 29.27->29.88 / 19.39->19.62 TPS; ordinary 36.27->36.75. Verification falls about 2.09%/1.25%, comparable in scale to ordinary variation. This is bounded coverage, not evidence at 128K occupied context. |
| Prefill/ordinary decode | That same fresh-process longer-input pair already records cold prefill 3224.4->3212.7ms and ordinary decode above. This closes a bounded prefill/ordinary smoke check; one pair does not establish robust universal nonregression. Reused-prefix prompt times are not cold-prefill measurements. |
| Cancellation/recovery/teardown | Corrected HC/paired writing case already passed. New full-combination longer-input case also cancels, recovers and exits cleanly. Candidate sequence includes an extra confidence request, so recovery histories are not an A/B numerical comparison. |
| Writing | Corrected HC/paired has a 3,239-token writing check (ordinary39.98 versus two-draft32.02 TPS at54.1% acceptance). It establishes workload dependence and lifecycle coverage. Full combined flags have not been qualified on writing; do not imply otherwise. |
| Notes/checkpoints/recommendation | Final compact summary/checklist must include longer-input and combined-parity results, limits and flag recommendations; checkpoint the harness/tool changes and compact evidence. Preserve large raw logits/traces locally. Several old pending-test sentences are superseded and should not become a new work queue. |

## Numerical history distinctions that must remain explicit

The long coding control/candidate match each other under the same forced draft
policy. Both forced-two histories differ from ordinary decode at token13;
forced-four differs at token3. The candidate confidence request differs from
ordinary at its first output token. Thus the confidence sample37.65TPS versus
ordinary36.75 is **not a matched-history performance win**. Its42/42 accepted
drafts and53 zero-draft rounds out of86 show selective engagement, not proof
of target-equivalent numerical behavior or faster batched verification.

Existing hybrid CPU/GPU placement can change quantized activation arithmetic
and weighted-sum partitioning; the cause of these particular long-history
differences has not been proven. A controlled-GPU exact pass isolates new
arithmetic/scheduling from that issue; it does not establish hybrid bitwise
equivalence to token-at-a-time decode. Do not relax thresholds or replace
full-logit evidence with matching greedy choices. No new regression is
established merely by the existing policy differences, either.

## Conditional work that is not required now

- No exhaustive workload/context/quant/width suite. Coding plus the existing
  writing case and the longer coding task cover the requested selected scope.
  A new writing A/B for all flags becomes necessary before recommending that
  combination; it is not justified solely to promote approximately1% ideas.
- No CPU quota sweep, live union planner or production IMMA integration.
  Corrected fixed-budget union ceilings are4.29%/9.62% of total copies, before
  timing/overhead; actual-model IMMA saving scales to only about0.1ms/round.
- No Cartesian combinations of async/component/CPU plans. Async and component
  streaming are incompatible schedules; rejected/disabled CPU placement does
  not need further qualification to remain an explicitly experimental hook.
- No new near-capacity or non-NVIDIA qualification merely to close the current
  research checkpoint. Those are limitations and future promotion gates.
  Combined pressure/stress qualification is required before a production
  recommendation, not evidence that opt-in research already delivered speed.

## Finite next work and performance decision

1. Finish the in-flight long-prefix combined numerical comparison. Investigate
   a failure rather than widening the benchmark matrix. On a pass, record its
   actual scope, including partial commits/reset/repeat and engagement.
2. Publish the compact longer coding/cold-prefill/lifecycle comparison using
   already captured data. Keep the confidence sample separate and explicitly
   unmatched. Do not rerun completed samples solely to obtain a larger gain.
3. Reconcile the checklist and checkpoint the tools/evidence. State that the
   current bounded R&D round is complete with no substantial performance
   improvement, while the original substantial-speed objective remains unmet.
   Keep default settings unchanged and distinguish correctness repairs from
   optional performance experiments.

There is no presently demonstrated larger complete implementation to recommend.
The corrected hybrid component path still spends33.02ms in fetch/expert work
and exposes6.56ms at down readiness; the same sample needs about12.48ms/round
removed to reach60TPS at its emitted-token yield. Even making that readiness
tail disappear would not bridge the full gap, and its disappearance has not
been shown feasible. Dense/HC/KDA rewrites, resident GPU repetition and host
enqueue time must not be counted as unclaimed removable work.

Before another implementation round, require a new measured critical-path
mechanism with enough whole-round headroom. Transfer traffic or
effective transfer throughput is the dominant remaining investigation area,
but current evidence neither identifies a large removable traffic set nor
proves an inefficient transfer primitive. Confidence gating may avoid losses
on low-acceptance tasks; it is a separate policy question and cannot stand in
for the requested cheaper complete multi-token verification.
