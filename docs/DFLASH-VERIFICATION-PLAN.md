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

## Confidence policy and adaptive-width graph integration

Existing upstream policy (minimum confidence0.95, no mandatory drafts, up to4drafts) with the complete candidate: short coding57.06TPS versus44.93ordinary; longer coding39.05 versus40.78ordinary. Acceptance94.85%/97.92%. These are one request each and not statistical guarantees. No new acceptance algorithm was invented.

Counters exposed incomplete graph coverage for this policy: short rounds by offered drafts0/1/2/3/4 =3/6/4/5/17; long =48/26/5/4/0. Existing graph gate supported onlyT3/T5. The long run bypassed1,260layer calls atT2/T4. Extend the existing graph cache to completeT2..T5 windows: identical native body, pointer/config keys, per-width ownership credits, component joins and capture rollback. No arithmetic or confidence-policy change. Unit CUDA test now cyclesT2..T5, dynamic pointers and key invalidation:8captures/16replays, all checks pass. Full-model T2/T4 hybrid exact gate underway before serving remeasurement.

## Adaptive graph coverage and confidence result

Implemented and qualified T2..T5graph coverage. Exact hybrid qualification:5,575,680logits/36rows and309,760prefix values; zero sampled ownership failures. Confidence policy now has zero graph bypasses. Short coding56.78TPS versus44.04ordinary; longer39.64versus40.83ordinary. Old graph coverage gave57.06/39.05TPS: small changes, no robust claim from this single pair. All paired old/new output IDs and acceptance counts match; prefill smoke and cancellation/recovery pass. Keep opt-in. High longer-task acceptance(47/48) conceals48zero-draft rounds out of83; it does not imply near2xgain. See window-graphs/CONFIDENCE-RESULTS.md. Next: bounded writing/lifecycle and ownership attribution, not a repeated matrix.

## Writing and ownership qualification (2026-10-15)

One3239-token writing prompt,128output cap, same complete candidate and confidence policy with graph flagoff/on. Ordinary40.22/40.57TPS; confidence40.10/40.96TPS (+2.15%graph pair, ordinary control varied+0.87%). Every paired output ID and acceptance count matches (41/45drafts accepted). Cold prefill2266.1/2265.4ms. This is a bounded workload check, not a statistical guarantee.

Extracted the existing opt-in promotion-retirement metadata transition for direct production-code tests:24cases pass, including evicted/reused GPU winner, ready/pending/absent replacement RAM, invalid reservations, no-mutation failures, sticky errors and double retirement. Full-model integration matches prior5,575,680window logits and309,760prefix values exactly;33superseded retirements, zero sampled ownership errors. DMA timing is outside the unit cases' scope. Background demotion audit found no valid competing-promotion path; no demotion change. See window-graphs/OWNER-TRANSITION-TESTS.md.

The previous turn is concrete progress: implemented shared transition tests, qualified integration and writing. Historical numerical failure remains unattributed; current exact checks do not erase it. Next targeted numerical investigation must compare pre-verification state rather than repeat TPS tests. Do not restart quant/width screening.

## Pre-verification numerical localization

Added fixture-only serialized prefix-state capture, bounded seam preservation and strict comparison tooling(four synthetic tests pass). Old-policy GPU graph-off/on now has both238.75MBprefix snapshots exact,309,760exact prefix logits and7,434,240exact verification logits; reproduces historical off exactly, not the historical failed on. No retroactive root-cause claim.

Same-prompt reset/prefill changes logits by2.682236within either arm before verification. First differing recurrent state: layer4(~5.2e-6), after exact layers0-2. A separate existing-seam diagnostic shows layer3last-row FFNinput exact, output difference1.49e-7, growing downstream. Instrumentation itself changes prefix numerics and is not a drop-in reproduction. Source points to residency-dependent combine partitions/MMQ reductions. Next bounded discriminator: captured layer3expert outputs plus production-combine replay; no further TPS reruns. See window-graphs/PREFIX-STATE-INVESTIGATION.md.

## 2026-10-15: first-MoE production replay

Captured layer3 last-row route weights, expert outputs, initial/final FFN and chronological combine groups on the same6793-token occupied prefix across two resets. Both outputs replay bitwise exactly through the production CUDA combine. Groups change3->4; swapping only partitions changes2135float bit patterns (max5.96e-8). Five/eight expert vectors also differ (max1.788e-7); combine order is not the sole source. Three focused replay tests pass. Diagnostic only, no TPS/threshold/default change. Historical failed graph run remains unattributed. See window-graphs/MOE-COMBINE-INVESTIGATION.md. Next, only if attribution requires it: selected MMQ intermediate/dispatch capture; no new throughput matrix.

## 2026-10-15: DMA qualification and remaining timing budget

- Extended the production promotion-retirement test: 24 metadata cases plus four actual asynchronous 1 MiB DMA/slot-reuse cases pass. Pending event, held source/destination, concurrent CPU backing read, completed-copy bytes, replacement RAM publication, and winner preservation after slot reuse are checked. Full scheduler race stress remains outside this fixture.
- Reconciled recorded confidence-policy round totals against wall time. Short coding needs 5.6% of verification time removed for 60 TPS; longer coding needs 36.7% across verification/ordinary fallback, or 65.1% if only nonzero-draft windows improve. Removing all drafter time gives optimistic ceilings of 58.58/42.19 TPS, not measured results.
- Source confirms zero-draft rounds already take ordinary forward. No redundant one-token window to remove. No new inference matrix, arithmetic/default change, or numerical threshold change.
- See window-graphs/REMAINING-BUDGET.md and remaining-budget.json. Historical numerical attribution remains open; goal remains active.

## 2026-10-15: MMQ down-stage localization

- Added bounded selected-expert input/post-SwiGLU capture and MMQ geometry. One GPU-only fixture with two resets completed; all eight inputs and hidden vectors exact, five down outputs differ (max 1.788e-7).
- Captured expert group sizes/max rows change with residency. Source distinguishes fused tiled GU from potentially Stream-K down reductions. This supports, but does not causally prove, shape-dependent down reduction as the source.
- Both complete combined captures are byte-identical to the previous diagnostic. Actual production combine replay remains exact. No TPS rerun or numerical-threshold/default change.
- Next discriminator, if needed for attribution: replay one captured down matrix and quantized input with both recorded group geometries. Historical graph failure remains unproven. See window-graphs/MMQ-PRODUCT-LOCALIZATION.md.

## 2026-10-15: causal MMQ group-geometry reproduction

Captured the actual layer3/expert61 down batch and replayed both dispatch geometries through production MMQ. Selected weights and quantized input bytes match. Both original outputs reproduce bitwise; zeroing unrelated batch data preserves each result. Between geometries, 3349 float bit patterns differ (max 1.788139e-7). This establishes dispatch/group-geometry dependence, without narrowing it to Stream-K versus tile choice. Corrected tail-ID padding from source review; CUDA memcheck reports zero errors. Combined captures remain byte-identical to preceding diagnostics.

The sampled repeat-prefill mechanism is now explained jointly with combine replay. Stop expanding this standalone diagnostic; return to campaign qualification/performance decisions. Historical graph failure remains separately unattributed because its starting state was not captured. No arithmetic/default/threshold change or TPS claim. See window-graphs/MMQ-DOWN-REPLAY.md.
