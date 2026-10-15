# Verification campaign status

Research branch: `research/dflash-batched-verification`. Daily configuration unchanged. No external publication.

## What is established

- Expanded scratch/reuse remains the starting checkpoint; this campaign does not claim those previous gains.
- The first grouped expert implementation and paired refinement passed 28,188,160 exact logits in their initial controlled tests. Clean coding differences were small.
- Matched-staging asynchronous fetch passed 28,188,160 exact logits with all experts warmed. Nsight proves fetch/route overlap, but clean hybrid verification-round times barely improved.
- Major dense projections already batch weights across token columns. A wholesale missing-GEMM explanation is unsupported.
- Narrow CPU/GPU assignment swaps offer only 2.1%/3.5% optimistic byte reductions at T3/T5; below 1.2% for the transient subset. That planner is deferred.

## Clean coding samples

These are short 192-output-token samples with 40 occupied prompt tokens and 128K capacity, not long-context evidence.

| Configuration | Control TPS | Two drafts TPS | Four drafts TPS | Interpretation |
|---|---:|---:|---:|---|
| First grouped-off reference | 41.93 | 41.91 | 35.09 | Original campaign RAM configuration |
| First grouped kernel | 41.73 | 41.76 | 35.24 | Same output history; no material gain |
| Paired grouped kernel | 42.04 | 42.26 | 35.79 | Same output history; small cross-run difference |
| Matched 94GiB RAM, staging, serial | 41.12 | 39.80 | 36.08 | Separate matched-capacity comparison |
| Matched 94GiB RAM, staging, async | 41.09 | 40.65 | 32.75 | Acceptance/history changed; no reliable scheduling gain |

Matched serial/async verify-round means were 67.262/66.885 ms at two drafts and 105.430/105.257 ms at four. Do not interpret their TPS difference as pure scheduling savings.

### After deferred-demotion correctness fix

The earlier rows predate the fix and are historical, not current controls.

| Configuration | Control TPS | Two drafts TPS | Four drafts TPS | Verify round ms (two/four) |
|---|---:|---:|---:|---:|
| Corrected control | 33.80 | 39.73 | 38.19 | 66.340 / 101.709 |
| Corrected HC + paired experts | 35.48 | 40.96 | 39.11 | 64.395 / 99.192 |

Output histories and acceptance match. Small 2.5-3% verification changes are
not a robust gain given ordinary control variation. Both remain experimental.
Bounded pressure qualification now also passes 6,040,320 exact logits against
the corrected all-memory reference (six windows, RAM64GiB/GPU16GiB).

## Current correctness gate

HC phase batching plus paired experts now passes the corrected GPU-only control exactly: 28,188,160 logits, 182 rows, partial commits and repeated requests. Earlier failures were traced to missing deferred-demotion submission. The old control also changes after the fix (max logit difference 6.4494, all greedy choices unchanged), so it cannot remain the correctness reference.

Isolated HC arithmetic passed all 128 cases/576 rows exactly. The immutable-pack audit localized the first combined mismatch to stale RAM in a pending demotion. GPU bytes exactly matched RAM; scratch addresses were correct. Batched window execution omitted submission of boundary-deferred demotions, so CPU service could wait on an old/unrecorded event and serve stale bytes. Added submission before batched layer routing. The corrected bounded audit passed all 42 MoE scopes: 1,599 expert/pointer pairs and 14,556,266,496 bytes match immutable pack. Clean corrected-baseline performance comparison is next; no HC speed claim yet.

Separately, serial disk-pressure tests were not reproducible. Four ownership-check boundaries passed published-pointer checks and five retained-copy byte comparisons, but do not resolve that issue. A background-transfer API-error handling fix is checkpointed; no observed API failure establishes it as the cause.

## Decisions

New component-streaming prototype also passes 28,188,160 exact logits. A
four-window Nsight capture measures 40.938 ms fetch/GU overlap, 41.599 ms GU
union and zero fetch/down-compute overlap. This proves the readiness mechanism,
not a 10.2 ms/window speedup. Capture includes synchronous group diagnostics;
clean timing confirmation shows only a 1.2% TPS difference: two drafts45.23 on
versus44.70 off with ordinary controls43.03/43.01. Verification58.081/58.841ms.
No robust speed recommendation; keep opt-in. Direct pressure diagnostics verify
22.707GB prefix +11.354GB down across3,627 nonzero sources and304 disk staging
uses; all6,040,320 pressure logits exact. Four-draft throughput43.84 is a sample,
not a confirmed gain. Raw profiler artifacts remain local.

- Keep performance experiments opt-in; no new default speed improvement is claimed.
- Preserve negative results and raw logits locally; commit only compact evidence.
- Qualify combinations only after their isolated correctness gates. Avoid another quant/width sweep.
- After a justified candidate exists, check writing, longer occupied context, prefill and cancellation/recovery.

Detailed chronology: `WORK-LOG.md`. Long-term plan: `docs/DFLASH-VERIFICATION-PLAN.md`. Checklist: `docs/DFLASH-VERIFICATION-CHECKLIST.md`.

## Latest phase attribution (supersedes pending-test wording above)

Corrected bounded pressure passes; prior unresolved pre-fix pressure observations are historical. Hybrid component-off/on diagnostics measure59.417/57.431ms verification, with83/90 accepted drafts and identical output IDs. Fetch/expert span34.50/33.02ms;6.56ms remains exposed at down readiness. These are profiled runs, not clean speed qualification. See HYBRID-PHASE-ATTRIBUTION.md. Broader union metadata and standalone exact IMMA feasibility are in progress.

## Latest checkpoint: placement, IMMA and exact columns

Corrected fixed-CPU-budget union placement is deferred: total-copy oracle ceilings 4.29% (T3) / 9.62% (T5), before overhead. Exact IMMA passes 246 strict cases but actual model shapes imply only about 0.1 ms/round saved; production integration rejected.

Exact-column DP4A removes padding, passes 7,434,240 full-model exact logits and 587 native/mixed-column cases. Coding four-draft A/B is 47.82 / 48.32 TPS with identical histories; ordinary controls vary 44.32 / 44.66. Retain opt-in only. See exact-cols/DECISION.md. No substantial new default gain and no 60 TPS claim. Broader qualification is not justified by this small result; future work needs a larger evidence-backed opportunity.

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

## Direct mapped-RAM expert arithmetic: measured and rejected

Implemented isolated full native arithmetic replay, four staged/direct x CPU
contention modes, exact output/HQ checks outside timing. One16-token fixed
coding fixture, two repeats;2688 measured layer records exact. With CPU work,
GPU timeline7.872->14.318ms and wall10.705->16.496ms per token-equivalent.
AllQ2/Q3/Q4 classes lose; do not integrate or sweep. Logical225.56MB remains
host-read traffic even with no explicit copy. This is replay, not servingTPS.
See direct-ram-probe/RESULTS.md. Retained candidate/defaults unchanged.

## Cross-window scratch retention screen

Saved and reproduced bounded CPU-only trace simulation with64physicalscratch
slots and current-layer live workspace protected. Unified causalLRU avoids
10.40%/3.21%ofT3/T5H2Dbytes; optimistic weighted future interval bounds reach
18.47%/21.74%before lookup/admission/D2D costs. No serving implementation orTPS
gain claimed. Ordinary recency is insufficient to justify a substantial new
path; next policy work must demonstrate practical reuse near the optimistic
case on fixed routing before live integration. See SCRATCH-RETENTION-FEASIBILITY.md.

## Ordinary-forward attribution checkpoint

Added fixed-input ordinary-forward/window-first comparison with1-token commits.
After6793occupied tokens, GPU-only1,084,160logits match exactly. Hybrid prefix
and first input exact; next input differs with sameIDs/residency atlayer3 but
CPU165/GPU256 swapped. Discarded future rows increment cold-selection counters;
commit does not rewind them. Ranking-off ablation removeslayer3difference but
residency changes cause CPU-placement difference atlayer6. No thresholds
relaxed, nohybridexact claim orperformance recommendation. See
forward-window-diagnostic/RESULTS.md. No new serving changes.

## Current next experiment

The layer-position causal scratch policy saves only 9.36%/9.17% of replay bytes (T3/T5), declining on the later half; no serving cache integration. A clean warm trace rules out late host submission for all measured gaps. Proceed with a bounded native MoE layer-graph experiment targeting device dispatch gaps; see hybrid-submission-trace/RESULTS.md.

## Current checkpoint: graph parity failure and ownership evidence

Native MoE graphs are implemented opt-in with actual CUDA unit coverage. Original full GPU-only parity failed; unchanged repeated and narrow checks passed. Do not accept passing reruns as a fix. Host-only owner diagnostics prove competing background/demand promotions leave up to 34 duplicate/orphan resident slots on the fixture, but that diagnostic's logits are exact, so numerical causation remains open. No serving graph TPS comparison has been run. See window-graphs/STATUS.md. Baseline/defaults remain intact; goal is not achieved.

## Hybrid graph qualification (2026-10-15)

With the ownership fix enabled in both arms, 7,434,240 verification logits (48 rows) and 309,760 separately recorded prefix logits (two resets) are bitwise exact and finite. Each arm retires 44 superseded uploads; all 16 sampled boundaries have zero duplicate residents, orphan residents or invalid pinned sources. Requested 22 GiB pool gives the same actual 20.95 GiB / 2,564 slots. Graphs perform 84 captures and 420 replays with no reported graph errors or bypasses.

Warm fixed-input call means: T3 61.113 -> 59.450 ms; T5 96.916 -> 95.323 ms. First-request means do not improve (T3 91.422 -> 91.584, T5 126.163 -> 126.852 ms). These are diagnostic call timings, not TPS; prefix logging and ownership diagnostics are enabled.

The original intermittent failure remains preserved and its cause unproven. Current matched graph-off/on qualification passes, allowing a bounded experimental serving speed gate, not production sign-off. Superseded-winner eviction and replacement-RAM edge cases remain source-reviewed rather than separately forced runtime cases. See window-graphs/hybrid-owner-fix/qualification.json and summary.json.

## Completed graph speed gate (2026-10-15)

Current exact GPU/hybrid qualification is followed by two bounded coding workloads. Short40-token prompt: warm two-draft50.82 ->52.42TPS, four-draft51.22 ->52.91TPS. Ordinary control45.27/45.06TPS. Actual84captures/6258replays across the four speculative requests, zero bypass/errors; STOP/recovery and teardown pass. Every paired output history and acceptance count matches.

Reversed-order6,793-token coding prompt: ordinary40.87/40.96TPS; two-draft24.02 ->25.05TPS (+4.29%), four-draft23.57 ->23.78TPS (+0.91%, not a reliable gain from one pair). Cold prefill3232.9 ->3219.8ms, a bounded smoke check. Full paired histories/counts match. Forced drafting remains much slower than ordinary on this task because acceptance is only35.8%/30.7%.

Retain graph implementation opt-in; no default or launcher change. The short-task graph gain is modest, and60TPS is not achieved. Preserve historical intermittent failure as unattributed; production sign-off and forced ownership edge-case validation remain open. Do not repeat the same width/quant matrix.

Next decision: inspect and exercise the existing confidence-based policy with this complete combination, rather than add more unconditional drafting or another general kernel sweep. Offline first-draft counts show the long two-draft task accepts35/74 first drafts overall, but17/19 when confidence>=0.95; this is descriptive only, not a simulated policy/TPS result. Confidence gating changes histories and must be measured if used. At the forced policy's1.73emitted tokens/round,60TPS requires28.83ms total per round versus65.32ms verification alone; dispatch optimisation cannot bridge that gap. See window-graphs/SERVING-RESULTS.md and acceptance-budget.json.

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

## 2026-10-15: consolidated experimental qualification

Completed a requirement-by-requirement evidence audit and consolidated DFLASH-VERIFICATION-RESULTS.md. Corrected the serving summarizer's hard-coded short-coding label and regenerated writing/long summaries from existing raw evidence. Recorded whole-device memory samples with explicit lack of isolated graph-allocation attribution. Reconciled stale checklist entries; production-specific historical attribution gates remain unchecked.

Final current build, one short coding workload: ordinary44.85TPS, confidence57.76TPS,92/97drafts accepted. All output IDs/acceptance counts match the earlier run;168captures/1806replays, zero graph failures/bypasses, STOP/recovery and clean teardown. Diagnostic capture flags disabled. Previous explicit WIN_ALLCPU=0 was omitted in this smoke, recorded in comparison.json; do not call this a strict same-config speed A/B. Both ordinary and confidence rates moved similarly, so no new optimisation gain claimed.

Bounded experimental outcome is now documented, with no60TPSor production/default/PR claim. No engine left running. See EXPERIMENTAL-QUALIFICATION-AUDIT.md and the consolidated results report.
