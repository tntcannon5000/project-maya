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

Next bounded experiment (correctness gate now passed): window-only CPU table
`011233445`, ordinary decode unchanged at `011223344`, existing single-node
merged rows enabled. At 4/6/8 RAM experts this assigns one extra expert to CPU;
the actual mix determines saved transfers. Prior partial-plan merging alone
saved only 3.3%/5.8% CPU expert rereads and did not establish a robust speed win.
Test width 4 once against the corrected candidate. Require shorter total and
verification round times (target at least 5%) before broadening qualification.
CPU placement changes arithmetic; compare merged/serial with the SAME plan to
qualify scheduling, rather than demanding GPU-only equality across placements.

## Component-streaming gate

Moderate CPU split failed the speed criterion. KDA state persistence is deferred:
the whole measured stage is only 1.57/3.68 ms at T3/T5, with plausible savings
below 1-2 ms at T5. Next test MAYA_WIN_COMPONENT_FETCH: route all rows with stable
token-owned disk staging, aggregate gate/up-prefix fetch, overlap down fetch
with existing grouped gate/up arithmetic, join before down arithmetic/combine.
Keep full-layer native formats, disjoint row scratch and identical arithmetic.
No resident-only arithmetic split. Existing GPU-only trace has 6.18/8.85 ms GU
at T5/T8; that is an optimistic overlap ceiling, not current hybrid savings.
Counters must prove component bytes/launches; Nsight must prove fetch/GU
intersection. Full-blob diagnostics join early by design and are not timing
evidence. Test exact logits then one clean width-2/4 screen; only qualify more
widely if total round time improves. Stream failure and next-layer slot reuse
must preserve ownership. No parameter sweep.

## Current gate update — 2026-10-15

The HC, moderate-CPU and component experiments above have completed. Bounded corrected pressure qualification passes. Moderate CPU allocation failed its speed gate; component streaming is exact but only1.2% ahead in the contemporary clean comparison. Follow HYBRID-PHASE-ATTRIBUTION.md for current attribution. Next: corrected route/ownership metadata to bound broader placement, plus standalone exact Q4_K tensor-core feasibility. Do not repeat old pending gates or presume these experiments justify production defaults.

## Exact-column DP4A follow-up

The complete exact IMMA experiment passes246 cases but actual-model shape scaling predicts only~0.1ms/round: do not integrate. Corrected fixed-budget union placement has4.29%/9.62% total-copy oracle ceilings and is deferred.

Source exposes a simpler redundant-work hypothesis: current DP4A rounds verification columns3->4 and5->8, executing unused columns and carrying their registers. Implement opt-in exact3/5 instantiations of the existing kernel while preserving original row geometry, reduction order and mixed-job batching. Prove dispatch, pass mixed-format/same-type-mixed-column tests, then one bounded microbenchmark. Only if meaningful: controlled full-model narrow-window logits and one coding width2/4 comparison; no general sweep. Prefill/single-column dispatch remains unchanged by design, but report what was actually tested.

## Direct mapped-RAM feasibility follow-up (2026-10-15)

Long-context combined qualification is complete: 7,434,240 exact controlled-GPU
logits after 6,793 occupied tokens; cold prefill and lifecycle checks recorded.
No substantial throughput gain was established and all performance flags remain
opt-in. Do not rerun those completed gates.

Next bounded hypothesis: compute transient experts directly from their stable
pinned-RAM backing, retaining all native arithmetic. Saved T3/T5 traces identify
34.76%/53.70% of fetched bytes as transient destinations. This does not remove
PCIe bytes: it removes the intermediate scratch write/read and copy dependency,
and may be slower because native K-quant loads issue inefficient host reads.
The first gate is complete copied versus direct GU/down arithmetic on actual
recorded sources with identical outputs and CPU co-execution. No serving-path
integration unless that decisive feasibility test shows worthwhile headroom.
See DIRECT-RAM-FEASIBILITY.md.

## Bounded full-MoE graph experiment

The direct-RAM experiment is complete and rejected. Causal scratch-retention policies have not justified integration. The clean hybrid trace now contains complete CUDA API correlation: all observed device gaps begin after the next launch API returned. No late host submission is established.

Test one complete native per-layer MoE graph with live route tables and full component-stream fork/join. Keep attention, positional drafter captures, HC and deferred-demotion submission outside. Target the measured 3.25–4.04 ms of MoE gaps per T3/T5 round; do not credit overlapping host enqueue time. Opt in only; preserve the retained baseline.

Gates: source review of request credits/ring safety, capture failure and buffer lifetime; build; exact graph-off/on GPU and matched hybrid fixed-input comparisons with partial commits and two resets; verify actual capture/replay counters; one short coding serving A/B at widths 2 and 4, including cancellation/recovery. If the speed gate passes, use a bounded repeat or trace to establish mechanism. No broad suite or parameter sweep. Keep cold graph creation separate from warm replay and report any graph memory/cache change.

## Completed graph speed gate (2026-10-15)

Current exact GPU/hybrid qualification is followed by two bounded coding workloads. Short40-token prompt: warm two-draft50.82 ->52.42TPS, four-draft51.22 ->52.91TPS. Ordinary control45.27/45.06TPS. Actual84captures/6258replays across the four speculative requests, zero bypass/errors; STOP/recovery and teardown pass. Every paired output history and acceptance count matches.

Reversed-order6,793-token coding prompt: ordinary40.87/40.96TPS; two-draft24.02 ->25.05TPS (+4.29%), four-draft23.57 ->23.78TPS (+0.91%, not a reliable gain from one pair). Cold prefill3232.9 ->3219.8ms, a bounded smoke check. Full paired histories/counts match. Forced drafting remains much slower than ordinary on this task because acceptance is only35.8%/30.7%.

Retain graph implementation opt-in; no default or launcher change. The short-task graph gain is modest, and60TPS is not achieved. Preserve historical intermittent failure as unattributed; production sign-off and forced ownership edge-case validation remain open. Do not repeat the same width/quant matrix.

Next decision: inspect and exercise the existing confidence-based policy with this complete combination, rather than add more unconditional drafting or another general kernel sweep. Offline first-draft counts show the long two-draft task accepts35/74 first drafts overall, but17/19 when confidence>=0.95; this is descriptive only, not a simulated policy/TPS result. Confidence gating changes histories and must be measured if used. At the forced policy's1.73emitted tokens/round,60TPS requires28.83ms total per round versus65.32ms verification alone; dispatch optimisation cannot bridge that gap. See window-graphs/SERVING-RESULTS.md and acceptance-budget.json.
