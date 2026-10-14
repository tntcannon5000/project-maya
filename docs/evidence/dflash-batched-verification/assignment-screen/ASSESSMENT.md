# Window assignment eligibility screen

2026-10-14. Source: [Q4_K_M/engine.log](Q4_K_M/engine.log), lines 27-44; configuration and binary/drafter hashes: [Q4_K_M/manifest.json](Q4_K_M/manifest.json). This is a diagnostic-only screen of the bounded swap policy in [the assignment design](../../../DFLASH-WINDOW-ASSIGNMENT-DESIGN.md), not an implemented planner or a speed comparison.

The narrow policy does not justify a production planner in this sample. Its optimistic removable-copy estimate is only 2.09% at T=3 and 3.48% at T=5; excluding permanent-slot promotion ambiguity leaves 0.75% and 1.13%. Retain the opt-in diagnostic and design; defer implementation of this narrow swap mechanism. Broader union assignment, including CPU-only repeated experts and different CPU/GPU allocations, remains untested and is not rejected by this result.

| Verification tokens / drafts | Windows / layer batches | Actual fetched bytes | Proxy swaps / removable bytes | Proxy fraction | Transient-only proxy swaps / removable bytes | Transient fraction |
|---|---:|---:|---:|---:|---:|---:|
| T=3 / 2 drafts | 24 / 1,008 | 20,300,759,040 | 46 / 424,673,280 | 2.092% | 15 / 151,781,376 | 0.748% |
| T=5 / 4 drafts | 18 / 756 | 26,672,037,888 | 99 / 927,399,936 | 3.477% | 30 / 301,793,280 | 1.132% |

Each total sums Q2_K, Q3_K and Q4_K layer records. Fractions are removable singleton bytes divided by actual copied bytes for the same T, not an average of per-type percentages. In this run copied bytes equal unique fetched-ID bytes. The screen chooses at most two disjoint exchanges per layer in deterministic greedy order; these estimates are not a proven optimum or a bound on all possible assignment policies.

T=3 has 84 first-null/later-GPU experts, all with exactly one null use, and 64 singleton partner pairs. T=5 has 205 first-null/later-GPU experts, 194 with exactly one null use, and 132 partner pairs. The unconstrained screen flags promotion ambiguity in 36/46 selected swaps at T=3 and 77/99 at T=5 (78.3% and 77.8%). The transient-only pass performs its own greedy selection, so its count is not obtained by subtracting ambiguous swaps from the first pass.

The screen infers CPU candidates from null GPU pointers. It does not observe the original CPU mask, initial RAM readiness or baseline shadow-routing state. It does not simulate changed spare availability, subsequent admissions, numerical differences from CPU/GPU repartitioning, or the shifted fetch/CPU overlap. Permanent-pool fetch destinations are conservatively flagged as promotion ambiguity. Even transient-only candidates therefore remain eligibility proxies, not verified legal exchanges or achieved byte savings.

The run used full-layer windows, scratch=64, pointer reuse, hybrid placement and `MAYA_WIN_GROUPED=2`, with a 64-token output cap and one short coding prompt. It does not cover T=4/6/7/8, other prompts, occupied long context, or a controlled placement ablation. It included 1,008/756 synchronous readbacks; drain and copy/wait times perturb throughput. Diagnostic host reductions totaled 3.211/3.909 ms, but that measures existing host reduction and this screen, not the proposed GPU selector/shadow planner's cost. Reported 63.51/95.82 ms verification times must not be treated as unperturbed performance evidence.

A small removable fraction of copy traffic cannot establish the same fraction of verification-time improvement. The production policy would add selection/planning work to every layer and must preserve counts, ownership, complete expert evaluation and retirement while potentially changing cache admission. This sample supports retaining the counters for future evidence and spending implementation effort elsewhere before constructing the narrow planner. Broader assignment would need a separate measured cost model and experiment.
