# Key findings behind our Maya performance gains

Scope: our Maya fork running GLM-5.3-Flash GSQ-RCO 3.0bit with Q4 attention.
This is a mechanism summary, not a benchmark catalogue. Findings come from the
retained campaigns; experimental flags are not all production defaults.

## 1. Avoid copying immutable experts back to RAM

**Our largest clear base-decode improvement.** Original tier ownership released
the RAM copy after promoting an expert to VRAM. Eviction then required copying
the same immutable weights back to RAM.

We retained bounded RAM backing copies, with explicit ownership and capacity
accounting. An eviction can reuse that backing instead of performing a GPU-to-CPU
copy. Expanded backing plus demotion scheduling produced roughly 23–28% gains
in the original paired screens. Warm copy-back traffic fell by about 99.8% in
one controlled fixture; that is not a guarantee for long or changing workloads.

The tradeoff is additional pinned RAM. Count the RAM tier, duplicate backing,
prefill lending, staging, and OS headroom together. This is especially relevant
to our 128 GB machine and should not become an unconditional low-RAM default.

## 2. Schedule remaining demotions around useful work

Demotions were already asynchronous: adding an async API alone was not the fix.
Moving their submission after the embedding upload and scheduling them by layer
allowed remaining copies to overlap useful computation. Source slots stay owned
until their completion events; publishing a replacement pointer does not make
the old storage safe to reuse.

This complements RAM backing: backing eliminates traffic; scheduling hides part
of the traffic that remains. Their gains must not be added as independent wins.

## 3. Coordinate CPU experts and GPU transfers

CPU expert execution and GPU uploads compete for host-memory bandwidth. Sending
more experts to the CPU does not eliminate their RAM reads. Our retained bundle
uses early CPU-answer handling, PAUSE-based polling, a multi-expert CPU task
pipeline, and bounded phase admission to preserve useful transfer/computation
concurrency. Readiness is attached to each job's pointer and sequence.

Together with GPU scheduling changes, this delivered modest additional gains;
individual flags are not independently proven speedups. Serializing CPU work
made transfers faster but made whole tokens slower, so we rejected it.

## 4. Overlap components, and reduce submission overhead

Gate/up weights are needed before down-projection weights. Component-ready
events and two down bands permit useful work before every byte has arrived.
CUDA graph replay for the MLA MoE suffix and cheaper warp-level route publication
reduce some launch/submission overhead.

The retained combination was useful, but broad resident-expert overlap prototypes
often lost: extra launches, GPU contention and longer CPU tails exceeded the time
hidden. **Measured overlap is not itself a performance gain.** Graph keys must
include changing execution conditions, including upstream's full-residency state.

## 5. Reuse work across speculative verification positions

For DFlash2, we added grouped native Q2_K/Q3_K/Q4_K expert computation, paired-row
work, batched hyper-connection operations, exact-column dense dispatch, component
fetch scheduling, and complete verification MoE graphs for eligible T2–T5 windows.
Token-owned scratch/staging and cross-row expert reuse support these paths.

The objective is to read/decode shared weights less often and submit fewer kernels,
while preserving each required expert calculation. Ownership fixes are essential:
these are supporting correctness changes, not optional performance switches.
Controlled graph comparisons showed modest gains; the full ordinary-versus-DFlash
difference cannot be credited entirely to our kernels.

## 6. Let confidence decide whether extra drafts are worth verifying

We use **DFlash2**, not the original DFlash model. Upstream's confidence cutoff
allows zero drafts on difficult rounds and more when confidence is high. This
policy is upstream functionality, not our invention.

Forcing wider windows was much worse on longer coding: verification processes
rejected positions, their expert transfers and CPU work. Narrow adaptive caps can
be competitive, but our small screens do not establish a universal best cap.
MTP drafted more cheaply but added expert-cache pressure and did not beat our
non-MTP control overall. Neither mechanism removes the target's verification cost.

## Findings that prevented wasted effort

- A perfect-future cache oracle nearly halved uploads in one replay. Our practical
  policies captured little of that opportunity. The oracle is not an implemented gain.
- DMA won isolated copy tests but lost after real scheduling/ownership integration.
- Already-quantized weights compressed poorly; measured CPU decompression was too slow.
- Lower CPU worker counts, additional paired CPU kernels and several residency
  variants did not establish general gains. They remain experiments, not defaults.
- Roughly 9 ms of traced GPU computation was not automatically 9 ms of removable
  waste. Several plausible arithmetic changes failed the whole-loop speed gate.
- Latest-upstream integration preserved measured performance; it did not establish
  a new reliable speedup. Its Windows VRAM-budget safeguard remains valuable.

## Evidence and interpretation

Scheduling-only changes and CPU/GPU placement changes require different parity
comparisons. Controlled comparisons achieved exact logits; arbitrary hybrid
placement can change floating-point reductions and generated histories. Never
hide that by loosening numerical thresholds. No 60 TPS claim is established.

Primary records: [transfer results](evidence/maya-transfer/report/RESULTS.md),
[scheduling campaign](MAYA-THREE-COSTS-CAMPAIGN.md),
[negative follow-up](MAYA-MISSING-COSTS-CAMPAIGN.md),
[verification results](DFLASH-VERIFICATION-RESULTS.md),
[upstream integration](UPSTREAM-INTEGRATION-20261015.md),
[MTP screen](MTP-ONE-DRAFT-20261015.md).
