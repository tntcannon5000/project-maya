# Batched verification work log

## 2026-10-14 — baseline profile

Parent c730fbb; plan checkpoint 22d03c0. Full-layer groups, scratch64, reuse enabled, single-node CPU batching off. Q4 drafter, 128K capacity, 40-token coding prompt, 192 output tokens. Profiled control 41.49 TPS; width2 41.73; width4 35.38. No retries.

| Per-window CUDA event interval (ms) | Width2 | Width4 |
|---|---:|---:|
| Full verify GPU timeline | 62.81 | 99.75 |
| Routing/wait/fetch combined | 33.00 | 56.60 |
| Expert computation plus shared down/group/combine | 9.33 | 12.72 |
| Exposed CPU answer waits | 3.37 | 5.14 |
| Other stages | 17.11 | 25.29 |

Host enqueue overlaps GPU work and is not additive. CUDA intervals can include host submission gaps. This is diagnostic, not unprofiled throughput. Routing/fetch accounts for more time than expert arithmetic; even eliminating all 9.33ms would not alone meet the ~19ms saving sought at width2. Therefore complete batching must address transfer/route work, not merely arithmetic. Finer counters/events are being implemented before deciding the next supporting mechanism.

Prototype direction: exact-order DP4A weight decode reused across tokens, mixed Q2_K/Q3_K/Q4_K, fallback preserved. Existing prefill multirow helpers do not directly cover all required formats; no tensor-core speedup assumed.

## First complete prototype

MAYA_WIN_GROUPED=1 uses native quantized DP4A arithmetic with decoded weights shared across members. Both gate/up and down are implemented for Q2_K/Q3_K/Q4_K; singleton groups and unsupported formats retain original arithmetic. No extra tensor conversion, quantization or launches. Dimensions must be divisible by 256 and T<=8. Register pressure and shared allocation also affect singleton CTAs in the specialized kernel, so source reuse alone is not a speedup claim.

Instrumentation now separates route / disk wait / fetch and shared down; group diagnostic reads actual slot membership and copied bytes. It deliberately synchronizes, so diagnostic TPS cannot be used as a performance result. Actual on-device branch counters are being added before the diagnostic run. Baseline executable preserved as build/strata-before-grouped-verification.exe.

Next gate: controlled GPU-only full-logit comparison on fixed windows, partial commits and eviction pressure. If exact, run one small end-to-end A/B and inspect group multiplicity. Investigate complete window scheduling only where the split profile supports it. Avoid assuming that all repeated CPU experts should move to GPU: that can increase transfer demand and destroy overlap.

## Grouped arithmetic qualification

Full-model grouped-off vs grouped-on comparison passed bit-exactly: 28,188,160 logits, 182 rows, zero mismatches, zero retries. Includes alternating T5/T8, partial commits, two resets/repeats and 16GiB expert capacity eviction pressure. Source arithmetic matches checkpoint 59d5b5f; test binary hash is recorded in manifests (built before diagnostic-only counters). This qualifies controlled GPU arithmetic, not hybrid-placement changes or a throughput win.

Follow-up scheduler design must address disk temporary staging: F->stage[m] can be reused on a later route, so merely placing fetches on a side stream would be unsafe. All-RAM-only gating would change the baseline configuration and may not engage; investigate bounded per-window staging or a precise source-retirement fence. Keep this separate from grouped arithmetic.

## Actual engagement and group diagnostic (64-token coding sample)

The GPU counters equal eligible repeated groups for every Q2/Q3/Q4 stage: 4342 repeated GU/down groups at T3 and 5816 at T5. Actual GPU entries/slot groups: 20776/15376 at T3; 25817/16034 at T5. This is ~26%/~38% fewer decoded weight-call memberships in the ideal reusable portion, not an equal speedup or DRAM-byte claim. Pointer groups equal GPU logical-ID groups: no observed pointer fragmentation. Fetch entries equal distinct fetched IDs in every quant/scope: no duplicate fetch-copy bytes remain within current placement.

Fine diagnostic intervals: T3 fetch21.57ms, route2.75, disk-wait0.67, experts11.61; T5 fetch37.98, route5.02, disk-wait1.04, experts13.81. Diagnostic synchronization/readback perturbs CPU overlap and adds ~4.1ms/window; do not compare these intervals directly to earlier baseline as speedup evidence. Clean end-to-end grouped run is next. Files in group-diagnostic/Q4_K_M preserve exact counts and bytes.

This evidence deprioritizes redundant-copy elimination. Prioritize a safe route/fetch overlap experiment and arithmetic profiling; union-aware placement remains proposed and requires a measured capacity/makespan opportunity, not assumed free savings.

## Clean A/B: first grouped kernel

Identical output histories/acceptance, no phase/group diagnostics:

| Variant | Control TPS | 2 drafts TPS | 4 drafts TPS |
|---|---:|---:|---:|
| Grouped off | 41.93 | 41.91 | 35.09 |
| Grouped on | 41.73 | 41.76 | 35.24 |

No material gain. Retain as opt-in research, not recommended/default. Compiled GPU resource audit found GU registers ~59/60/58 baseline vs128/128/127 grouped for Q2/Q3/Q4; no local-memory spills. At512 threads this plausibly halves active CTAs. Refine based on resource/multiplicity evidence rather than endless rerunning. Exact performance attribution still needs kernel timing/occupancy, not register counts alone.

## Ranked next mechanisms

1. Safe asynchronous per-token fetch submission: complete bounded prototype built; join before all consumers; extra864MiB pinned RAM protects disk fallback source lifetime. Preserve serial route/disk-response ordering. Pressure parity now tests64GiB RAM tier to exercise disk paths, not just warm all-memory behavior.
2. Grouped-kernel resource refinement: preserve singleton efficiency and smaller member groups; reject extra launch/grid overhead if it outweighs reuse. Await concrete resource-grounded design.
3. Union-aware assignment: documented but deferred until a bounded eligibility experiment shows available byte/makespan savings. Current fetches are already unique under existing assignments. Do not implement a large shadow-routing simulator without evidence of benefit.

Trace capture hook added to fixed-window test for four warmed windows only; intended to demonstrate actual stream overlap if async candidate is correct/promising. Throughput remains separately measured.

## Async pressure qualification failed — blocked from speed claims

With64GiB RAM tier and16GiB GPU pool, async-off/on differed: max5.3676, mean0.1619;181/182 greedy rows. First14 logit rows exact; divergence starts row14 (second token of third window). No retries; token-owned disk staging exercised1344 times across repeats. Normal-memory async also differs from older normal reference, but extra pinned allocation changes available RAM tier, so this is not a clean same-capacity isolation. Serial normal vs serial pressure differ already fromrow0, likely a capacity-dependent prefill path; do not conflate that with the same-config async discrepancy.

Next decisive isolation retains async allocation and staging but joins each token's fetch before advancing routes. Added rare ring-lap/skipped-sequence counters without changing behavior. No numerical thresholds weakened. No performance claim for async while qualification fails.

## Paired refinement and async isolation follow-up

Mode 2 separates singleton work and processes repeated groups in pairs, reducing compiled register pressure without changing quantized arithmetic. The controlled normal-memory GPU-only comparison passed all 28,188,160 logits bit-exactly (182 rows, max absolute difference zero). Evidence: pairs-parity/comparison.json. Clean coding widths 2/4 are being measured; compiled resources alone are not a speed claim.

Async JOIN_EACH isolation retains the same staging allocation and side stream, but waits after each token fetch. It agrees with pressure serial reference for 78 rows, then differs; fully asynchronous diverges at row 14. No nonzero ring-lap report was observed. Next run the identical serial pressure control once to test reproducibility before further ownership changes. Extra pinned allocation and different memory-tier capacities must not be conflated with scheduling effects.

Clean mode-2 coding sample completed: control 42.04 TPS, two drafts 42.26 TPS, four drafts 35.79 TPS. Prior clean grouped-off was 41.93/41.91/35.09 TPS. Small cross-run differences are not a robust gain; no recommendation/default change. Evidence: pairs-clean/Q4_K_M (same acceptance and output histories to be retained for attribution).

Serial pressure repeat was not reproducible either: first differing row 14, max absolute difference 8.83584, greedy 182/182. This prevents attributing the early async mismatch solely to async scheduling; it does not qualify async. Independent staging allocation control added in 14d0bb9. Next paired runs use identical 94GiB requested RAM tier and token staging for both scheduling modes, below the previously observed pinning cap.

Matched-stage qualification passed: both controls used 94.00GiB RAM/11,389 slots,16GiB GPU/1,939 slots,64 token-owned staging slots. All28,188,160 logits exact,182/182 greedy; all12,096 experts warm and zero disk fallback staging uses. This qualifies RAM-backed async scheduling, not disk pressure. Four warmed GPU-only windows captured651.41ms kernel-span,414.11ms fetch union and27.59ms overlap with route kernels (6.90ms/window),37.29ms overlap with any other kernel. This proves concurrency but is not saved critical-path time or clean throughput; other kernels include waits. Trace manifest and overlap.json preserve details.

Matched-capacity clean async screen (94GiB RAM,22GiB GPU, hybrid plan011223344): serial control41.12,2draft39.80,4draft36.08 TPS; async control41.09,2draft40.65,4draft32.75 TPS. Average verify rounds barely changed:67.262→66.885ms at2draft,105.430→105.257ms at4draft. Acceptance/history differed (2draft89.13%→91.18%;4draft73.98%→64.35%), so TPS deltas cannot be credited to scheduling. Actual route/fetch overlap has not translated into a convincing verification-round saving. Retain opt-in experimental only; avoid further parameter sweep. Next source-backed arithmetic candidate is exact-order HC phase batching, while bounded assignment diagnostics test a transfer-reduction opportunity.

Bounded ownership diagnostic passed four pressure-window boundaries: five shadow-backed eviction blobs/41,287,680 bytes exact; published GPU/RAM pointer checks passed at all four boundaries. This rules out corruption only for those inspected blobs/states and is timing-perturbing; pressure non-reproducibility remains unresolved. Separate commit a4fe36c checks previously unchecked background copy/event submissions, preventing unsafe ownership publication on CUDA API failure; no such error was observed, so it is not claimed as the mismatch fix.

Window assignment proxy screen: removable singleton bytes2.092% of fetched bytes atT3 and3.477% atT5; transient-only0.748%/1.132%. CPU provenance/initial RAM readiness were not observed and promotion effects remain ambiguous, so these are optimistic proxy opportunities, not implementable savings guarantees. Defer narrow swap planner: complexity/launch cost is not justified by this bounded estimate. Broader union assignment remains untested. See assignment-screen/ASSESSMENT.md. This diagnostic also verifies mode2 grouped branches executed across Q2/Q3/Q4.

Implemented opt-in MAYA_WIN_HC_BATCH=1: existing HC3 arithmetic split into three stream-ordered phases with grid(32,T), private832float partials per token, identical reductions/q8 arithmetic, both attention andFFN HC call sites. Original path retained for unsupported dimensions/modes/T. Scratch3,328bytes/token (53,248bytes at current tmax16); successful dispatch counts report actual engagement. Engine and parity target builds passed; combined HC+paired expert full-logit qualification now running against matched-stage serial reference. No speed claim yet.

HC qualification not passed: HC+mode2 differs from matched-stage serial at row34 (window p0=60,T8,token3/position63), max5.50564,182/182 greedy. Same-current-binary HCoff/grouped0 repeat is bit-exact against retained reference. HC-only differs later, firstrow170,max3.63517,182/182 greedy. No thresholds relaxed or speed claim. An isolated HC arithmetic fixture is being built; potential next diagnostic compares batched versus original HC on identical actual input buffers, independent of changing cache schedules. Combined candidate remains experimental/unqualified.

Isolated HC unit harness passed128cases/576rows exactly, including partialscratch, guards, alias combinations and offset/paddedstrides. Fullmodel HC remains unqualified; synthetic coverage is not proof for every activation. Added one-window immutablepack audit and bounded fixture repeat/step options. Firstaudit rejected releasedWindows mappings before checkingbytes (diagnosticguarderror, notmodelcorruption); guard nowaccepts retaineddirecthandle with unchangedrangechecks. Corrected audit runningonlyfirstsixwindows/onepass atp0=60.

Decisive immutable-weight audit failure: p0=60,layer3,token3/rank4,expert48 has incorrect GPU scratch bytes atoffset0 versus immutablepack (source reported RAM-tier);26expert/pointer pairs and368,050,176bytes verified before firstmismatch. This matches firstcombinedcandidate logit divergence atposition63. Next distinguish whether RAMsource itself differs orGPU destinationwas overwritten/aliased; HC arithmetic is not leading suspect afterunitpass. No repairorparityclaim yet.
# Deferred-demotion correctness investigation

Provenance audit at p0=60 found RAM class 0 slot 248, layer 3 expert 48,
in kRDemote state. GPU bytes exactly equal that RAM source; both differ from
immutable pack. All 64 scratch pointers are correct. Thus this is not evidence
of HC arithmetic error or stale GPU cache loads.

Source review identifies the missing dependency: fast_boundary queues NEW
deferred demotions in `layers` mode; fast_layers submits them, but
fast_window_layers did not. The CPU service can then wait on an unrecorded or
recycled event and serve old RAM bytes. Added submission at batched-window layer
entry before route publication. Bounded immutable-pack audit passed 42 layers,
1,599 expert/pointer pairs and 14,556,266,496 bytes with no mismatch. Corrected
control and combined HC/paired full-logit runs follow, without byte diagnostics.
Performance claims remain gated on validation.

## Corrected clean screen

HC+paired matches corrected control across 28,188,160 logits. Clean coding
screen (192 output cap, 40 occupied prompt tokens, 128K capacity, RAM94GiB,
staging matched): control ordinary/two/four drafts = 33.80/39.73/38.19 TPS;
HC+paired = 35.48/40.96/39.11 TPS. Verify-round times fall 66.340 to 64.395 ms
and 101.709 to 99.192 ms. Identical output histories/acceptance. The 2.5-3%
round-time change is small, while ordinary decode varies ~5%; do not claim a
robust speedup. Keep opt-in. Next decisive candidate is moderate window-only
CPU allocation plus existing merged CPU rows. Bounded RAM-pressure regression
also checks whether the demotion fix resolves previous non-reproducibility.

Bounded pressure regression passed: RAM64GiB/GPU16GiB, combined HC/paired,
six alternating windows and partial commits. All 6,040,320 logits (39 rows)
match the corrected all-memory control exactly. This covers the previously
observed early pressure failure, not an exhaustive pressure/lifecycle claim.

## Moderate window CPU split

Implemented strict opt-in MAYA_WIN_CPU_PLAN, ordinary decode unchanged. Test
011233445 with two 8-thread window lanes and merged CPU rows at four drafts.
Engagement: 2,013 batches, 16,092 CPU entries, 15,141 unique reads, 48 windows,
zero retries. Ordinary control 43.07 TPS, speculative 38.35 TPS; verification
101.295 ms/round versus previous HC/paired 99.192 ms. Acceptance/history changed
(147/192 versus 144/192 accepted); not a pure scheduling comparison. Even with
faster ordinary control this did not meet the required 5% round-time gain.
Reject as recommended performance setting; no quota sweep. Keep opt-in hook
and negative evidence. Same-plan merged/serial numerical qualification remains
required before recommending the experimental override.

## Writing and lifecycle check

Corrected HC/paired, override disabled, 3,239 occupied prompt tokens and 128K
capacity. Short 128-output writing sample: ordinary 39.98 TPS, two drafts
32.02 TPS at 54.1% acceptance, verification 61.777 ms/round. This workload
benefits less from drafting; no universal drafting gain claimed. STOP cancelled
the four-draft request, the following two-draft recovery completed, no retries,
clean process exit. This is a lifecycle/longer-input smoke test, not a matched
prefill speed comparison or qualification at 128K occupied context.

## Complete component streaming prototype

Split the existing expert API into unchanged group/GU and down/combine phases.
MAYA_WIN_COMPONENT_FETCH resolves all routes first into stable sources, copies
the GU prefix as one aggregated operation, then copies down components with a
32-CTA side-stream kernel while grouped GU runs. Joins protect down consumers
and later layer reuse; unsupported cases fall back, async-fullblob combination
is rejected, full-weight audit joins early. Actual bytes use diagnostic source
readbacks, separate from clean submitted-launch counters.

Qualification passed 28,188,160 exact logits versus corrected control, 182 rows,
partial commits and repeated requests. Each repeat engaged 588 full-layer
component groups with zero fallbacks/retries. A four-window Nsight capture with
traffic diagnostics is next; no performance claim from parity or flags.

Component trace proves 40.938 ms fetch/GU overlap over four GPU-only windows,
41.599 ms GU union, and no fetch/down arithmetic overlap. These profiled T5/T8
windows are not hybrid clean latency measurements. Trace logits remain exact.
Windows Nsight did not forward target stderr, so direct pressure diagnostics
provide byte evidence instead: 252 layer groups, 3,627 actual fetch entries,
22,707,437,568 GU-prefix bytes and 11,353,718,784 down bytes, 304 disk staging
uses, no fallbacks/retries. All 6,040,320 pressure logits match corrected control.

First clean component sample: ordinary 43.03 TPS, two drafts 45.23 TPS, four
drafts 43.84 TPS. Verification rounds 58.081/88.273 ms versus earlier HC/paired
64.395/99.192 ms. Acceptance matches (87.14%/75%). Ordinary control also improved,
so a narrow reverse-order component-off confirmation is required before crediting
the apparent 10-11% round reduction to scheduling.

Reverse-order two-draft confirmation: component-off ordinary43.01 TPS versus
component-on43.03; speculative44.70 versus45.23 TPS; verification58.841 versus
58.081 ms. Identical output history/acceptance. This reduces the apparent gain
to 1.2% TPS / 1.3% verification time, insufficient for a robust recommendation.
Four-draft gain is not confirmed under matched contemporary conditions. Do not
attribute the earlier 10-11% change to component scheduling. Keep experimental
and disabled by default; no worker-count sweep or further weak-idea tests.

## 2026-10-15: corrected hybrid phase attribution

Measured 2026-10-15; same binary, RAM94GiB, requested GPU22GiB, 128K allocated context, 40 occupied prompt tokens, 128 output tokens, two draft tokens. Both runs use HC batching and paired expert kernels. These are phase-profiled diagnostics, not clean throughput qualification.

| Per verification round | Component off | Component on |
|---|---:|---:|
| Host verification elapsed, ms | 59.417 | 57.431 |
| GPU event timeline, ms | 58.56 | 56.60 |
| Fetch + expert stage span, ms | 34.50 | 33.02 |
| Exposed CPU wait, ms | 3.00 | 3.11 |
| Disk-service wait, ms | 0.82 | 0.98 |
| Profiled generated TPS | 45.93 | 47.50 |
| Ordinary control TPS | 44.50 | 43.40 |

Both runs completed 45 rounds and accepted 83/90 offered drafts. All corresponding warmup/control/speculative output token IDs match. This does not independently prove identical internal routes or numerical states.

Component-on expert sequence: prefix fetch17.15ms -> gate/up6.09ms (down transfers run concurrently) -> exposed down-readiness join6.56ms -> down/combine3.22ms. Total33.02ms. Off fetch26.00ms + experts8.50ms =34.50ms. The sum identifies a1.48ms stage-span reduction; it does not isolate bandwidth changes or arithmetic contention. The join is the remaining transfer tail, not the complete down-copy duration. Exact overlap requires stream intervals.

Host enqueue43.51/41.98ms overlaps device execution; adding it to GPU event times would double-count. Host final wait15.16/14.73ms similarly is not an additional independent GPU cost. The prior GPU-only T5/T8 capture's40.938ms overlap across four windows cannot be subtracted from these hybrid T3 rounds.

The clean contemporary comparison remains45.23 versus44.70TPS (1.2%), insufficient for a robust default change. Diagnostic pair suggests a small benefit, not the earlier apparent10-11% gain. No60TPS claim. The interrupted hybrid-phase-off run produced no completed requests and is excluded.

## Corrected placement and exact tensor-core feasibility

Implemented opt-in pre-route residency/ownership snapshots and actual route/copy-owner metadata. Corrected64-token T3/T5 sample validated1008/714layer records. All available corresponding output IDs match contemporary corrected histories. Fixed-CPU-budget oracle ceilings are4.29%/9.62% of total fetched bytes; this ignores planning costs and relaxed per-token quotas. Pending unpublished demotions require host responses, not necessarily disk reads. Raw trace stays local; compact summary includes its hash.

Built complete standalone exact Q4_K IMMA kernel.240 varied cases plus6actual model-shape cases pass bit-exactly. T3 slower; actual T5 model-shape gains extrapolate to only~0.098ms/round. Defer production integration. Checkpoint source/harness/evidence; no daily/default changes.

## Exact-column DP4A checkpoint

Implemented complete opt-in three/five-column instantiations, retaining legacy row geometry and mixed-job block accounting. Baseline and candidate each pass 587 strict native/mixed-column cases; full-model T3/T5 verification passes 7,434,240 exact logits. Compiled registers fall (Q4: 103 to 80 for T3, 80 to 75 for T5).

Contemporary four-draft coding A/B: 47.82 to 48.32 TPS, verification 86.393 to 85.417 ms/round, identical output IDs and 97/120 accepted drafts. Ordinary control 44.32 to 44.66 TPS. Small result is not a robust default recommendation; keep opt-in. Do not launch a larger suite or claim additive component gains. No inference remains running after the checks.

## Longer-context combination qualification

Completed one 6,793-token coding task with corrected control and grouped/HC/component/exact-column combination. Forced two/four-draft TPS29.27/19.39 versus29.88/19.62; corresponding output IDs and acceptance match. Ordinary36.27/36.75, cold-prefill3224.4/3212.7ms. Lifecycle checks pass; candidate extra confidence request means later cache/history sequence differs. No robust speed/default claim.

Added an explicit upstream-confidence policy request to the benchmark tool and configurable fixed-prefix/batched-prefill support to the logit fixture. The initial long fixture used serial forward and was deliberately interrupted before logits; its replacement uses serving prefill semantics. Combined candidate passes7,434,240 exact controlled-GPU logits after the6793-token prefix, including partial commits/repeats.

Confidence policy engaged (42/42 drafts accepted;53of86 rounds skipped drafting), but differs from ordinary output history; its37.65TPS is not matched speed evidence. VRAM audit finds only~1.4GiB potentially lendable futureKV at short contexts, requiring coordinated ownership rather than a safe isolated allocation change. Direct mapped-RAM arithmetic for transient experts is now under source review; this does not eliminate PCIe traffic.

## Direct mapped-RAM expert arithmetic: measured and rejected

Implemented isolated full native arithmetic replay, four staged/direct x CPU
contention modes, exact output/HQ checks outside timing. One16-token fixed
coding fixture, two repeats;2688 measured layer records exact. With CPU work,
GPU timeline7.872->14.318ms and wall10.705->16.496ms per token-equivalent.
AllQ2/Q3/Q4 classes lose; do not integrate or sweep. Logical225.56MB remains
host-read traffic even with no explicit copy. This is replay, not servingTPS.
See direct-ram-probe/RESULTS.md. Retained candidate/defaults unchanged.
