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
