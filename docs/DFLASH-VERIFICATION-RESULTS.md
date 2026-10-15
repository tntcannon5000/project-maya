# Efficient multi-token verification: experimental campaign results

2026-10-15. Branch: `research/dflash-batched-verification`. No publication, daily
launcher change or new performance default. The experimental implementation is
qualified within the bounded tests below; **60 TPS was not achieved**. This is not
an unconditional production or PR sign-off.

## Outcome

Complete grouped native-quant computation and supporting scheduling are implemented,
with demonstrated engagement, controlled numerical checks and end-to-end measurements.
The practical short-coding result is **56.78 TPS**, compared with **44.04 TPS** ordinary
decode in that run. Longer coding and writing remain around **40 TPS**. The existing
confidence policy supplies much of the practical improvement over forced drafting;
it is upstream behavior, not a new algorithm from this campaign.

The campaign began from the expanded scratch/reuse baseline `c730fbb`. A missing
deferred-demotion submission was subsequently found and fixed; measurements before
that correctness fix are historical, not a sound cumulative speed baseline. Changes
in capacity, placement and policy also prevent claiming one cumulative percentage
from the earliest sample to the final candidate. The controlled individual A/Bs
below are the defensible attribution.

## Final serving samples

Same GSQ-RCO 3.0bit Q4-attention target pack, Q4_K_M DFlash2 drafter, c=1, 128K
allocated capacity, CPU plan `011223344`, requested GPU22/RAM94 GiB, 128 output cap.
Capacity is not occupied context. These are bounded samples, not independent repeated
trials or guaranteed workload performance.

| Workload | Occupied prompt | Ordinary TPS | Complete candidate + confidence TPS | Accepted / offered drafts |
|---|---:|---:|---:|---:|
| Short coding | 40 | 44.04 | 56.78 | 92 / 97 |
| Longer coding | 6,793 | 40.83 | 39.64 | 47 / 48 |
| Writing | 3,239 | 40.57 | 40.96 | 41 / 45 |

Confidence policy: maximum4 drafts, minimum confidence0.95, minimum drafts0,
cumulative threshold0. Ordinary versus speculative generation need not have identical
histories, so this table is a workload comparison, not isolated kernel attribution.

Matched graph-off/on results preserve output IDs and offered/accepted counts:

| Matched comparison | Graph off TPS | Graph on TPS | Interpretation |
|---|---:|---:|---|
| Short coding, two mandatory drafts, repeated request | 50.82 | 52.42 | Modest ~3% gain |
| Short coding, four mandatory drafts, repeated request | 51.22 | 52.91 | Modest ~3.3% gain |
| Longer coding, two mandatory drafts | 24.02 | 25.05 | +4.29%; still slower than ordinary |
| Longer coding, four mandatory drafts | 23.57 | 23.78 | +0.91%; not robustly established |
| Writing, confidence policy | 40.10 | 40.96 | +2.15%; ordinary control varied +0.87% |

Sources: [serving](evidence/dflash-batched-verification/window-graphs/SERVING-RESULTS.md),
[confidence](evidence/dflash-batched-verification/window-graphs/CONFIDENCE-RESULTS.md),
and [writing paired data](evidence/dflash-batched-verification/window-graphs/writing-summary.json).

## Implementations and decisions

| Work | Verified mechanism/result | Decision |
|---|---|---|
| Complete grouped Q2/Q3/Q4 expert arithmetic, paired refinement | Native GU, activation, down and combine; exact controlled outputs; small clean gains | Retain opt-in `MAYA_WIN_GROUPED=2` |
| HC batching | Same reduction order, private scratch; isolated and full-model exact checks | Retain opt-in `MAYA_WIN_HC_BATCH=1` |
| Component fetching | GU-prefix readiness, down-copy overlap, join before down; real Nsight overlap; clean pair only ~1.2% | Retain opt-in `MAYA_WIN_COMPONENT_FETCH=1` |
| Exact3/5-column DP4A | Removes padded columns while preserving row/reduction geometry; mixed-format checks pass; ~1% sample gain | Retain opt-in `MAYA_MV_EXACT_COLS=1` |
| Complete native MoE CUDA graphs | T2..T5, live pointer/config keys, bounded per-width credits, stream fork/join, rollback/fatal handling and teardown | Retain opt-in `MAYA_WIN_MOE_GRAPH=1` |
| Deferred demotion submission | Batched routing now submits pending copies; immutable audit and corrected pressure checks pass | Correctness fix retained |
| Superseded promotion ownership | Preserve newer GPU owner; keep copies reserved until ready; publish valid RAM replacement before freeing old source | Retain opt-in `MAYA_PROMOTION_OWNER_FIX=1` |
| General asynchronous route/fetch | Exact matched-allocation checks and real overlap, no convincing verify-round gain | No performance recommendation |
| Moderate extra CPU work / merged rows | Complete tested placement; failed speed gate | Reject recommendation |
| Exact Q4_K IMMA | 246 exact cases, actual-shape estimate only ~0.1ms/round | Reject production integration |
| Direct mapped-RAM arithmetic | Native arithmetic tested with CPU co-execution; substantially slower across quant classes | Reject |
| Union placement | Corrected fixed-budget oracle savings only4.29%/9.62% of copy bytes at T3/T5 before overhead | Defer |
| Cross-window scratch retention | Causal policies and capacity assessed; no demonstrated material practical headroom | Defer |

The performance flags are a combination, not independent guaranteed speedups. Complete
native graphs require full-layer native Q2/Q3/Q4 top8, c1, row scratch and token-owned
disk staging; unsupported scopes fall back. Tested configuration also uses
`MAYA_WIN_SCRATCH=64`, `MAYA_WIN_REUSE=1`, `MAYA_WIN_FETCH_STAGE=1`, and
`STRATA_GLM_WIN_STAGGER=0`. Full inherited retained settings and exact binaries are
preserved in each manifest; this list is not a replacement daily launcher.

## Correctness and mechanism qualification

| Requirement | Authoritative evidence and scope |
|---|---|
| Native arithmetic | Corrected HC/paired combination28,188,160 exact logits; exact-column587 isolated cases and7,434,240 full-model values. Do not sum overlapping tests as independent evidence. |
| All adaptive graph widths | T3/T5 GPU and matched-hybrid comparisons each7,434,240 exact logits; T2/T4 hybrid5,575,680 exact logits plus309,760 exact prefix values. |
| Actual graph engagement | Longer adaptive run:126 captures/1,344 replays, zero bypass/errors; short including STOP/recovery:168/1,806. Actual CUDA helper covers T2..T5, pointers, invalidation, joins, credits and abandoned capture. |
| Partial commit/reset/occupied context | Exact fixtures include6,793 occupied tokens, partial commits and two resets; separate ordinary/window first-row GPU check. |
| Expert ownership | Immutable pack audit1,599 copies across42 layers; bounded pressure parity;24 production metadata cases and4 real asynchronous DMA/CPU-read/slot-reuse cases. Not arbitrary scheduler-interleaving proof. |
| Request lifecycle | STOP, recovery and clean teardown recorded. Current phase did not repeat a full suite. |
| Prefill | Cold writing2266.1→2265.4ms; short718.2→718.4ms; longer3234.8→3215.0ms. Bounded smoke checks, not full128K occupancy qualification. |
| Sampled prefill differences | Actual combine and MMQ replay prove grouping-dependent rounding with identical selected data; CUDA memcheck0errors. No relaxed threshold. |

Core machine-readable checks: [T2/T4](evidence/dflash-batched-verification/window-graphs/adaptive-width-parity/graph-off-vs-graph-on.json),
[GPU T3/T5](evidence/dflash-batched-verification/window-graphs/owner-fix/graph-off-vs-graph-on.json),
[hybrid T3/T5](evidence/dflash-batched-verification/window-graphs/hybrid-owner-fix/graph-off-vs-graph-on.json),
[ownership](evidence/dflash-batched-verification/window-graphs/OWNER-TRANSITION-TESTS.md),
[DMA extension](evidence/dflash-batched-verification/window-graphs/REMAINING-BUDGET.md),
[causal prefill replay](evidence/dflash-batched-verification/window-graphs/MMQ-DOWN-REPLAY.md).

### Final current-build smoke

After the diagnostic additions, rebuilt the deliverable and ran only the short coding
workload plus STOP/recovery. Ordinary44.85TPS; confidence57.76TPS;92/97accepted drafts.
All request output IDs and acceptance counts match the earlier short run. Cold
prefill725.2ms versus718.4ms previously. Graph engagement168captures/1,806replays,
zero bypass/failure; process exits cleanly. No numerical capture flags enabled.

The previous ordinary/confidence rates were44.04/56.78TPS. Their similar percentage
movement does not establish a new optimisation gain. Manifest comparison also records
one setting difference: the previous run explicitly set STRATA_GLM_WIN_ALLCPU=0;
the smoke omitted it (source only activates the default all-CPU plan when both window
lanes exist). Preserve that distinction; this is a final functional/performance smoke,
not a strict same-configuration A/B. Earlier matched comparisons remain the attribution
evidence. See [current-build comparison](evidence/dflash-batched-verification/window-graphs/final-current-smoke/comparison.json).

## Memory and remaining costs

Actual reported expert pool remains2,523slots across matched graph runs, around20.63GB
after workspace sizing. RAM tier reports93.99GB pinned; token-owned staging and RAM
backing settings are additional considerations. Requested pool22GiB is not actual
available expert capacity. No claim that all memory is free for experts.

Saved device-wide1-second samples peak at31,857/31,805MiB (writing off/on) and
31,849/31,808MiB (long forced off/on); these include loading, prefill and other desktop
allocations. They **do not isolate incremental CUDA graph allocation** or imply graph
memory is negative/free. Graph-specific incremental bytes remain unmeasured; unchanged
expert slot capacity is established. See [memory observations](evidence/dflash-batched-verification/window-graphs/memory-observations.json).

Verification remains the dominant cost. On the recorded short sample,60TPS requires
5.6% less total verification time; on longer coding,36.7%. If only nonzero-draft windows
improve, the longer requirement is65.1%. Zero-draft rounds already use ordinary forward.
Even eliminating all drafter time yields optimistic fixed-history ceilings of58.58 and
42.19TPS respectively. These are subtraction budgets, not measured counterfactuals.

## Limitations and production gates

- The original intermittent graph comparison failed and its starting state was not
  saved. It remains unattributed. Current exact comparisons do not erase it; the causal
  prefill results explain a separately captured repeat-prefill effect, not that old run.
- Hybrid placement can change CPU/GPU arithmetic. Exact matched-placement results do
  not establish unrestricted ordinary/hybrid numerical equivalence.
- Samples are short and c1; no broad quant/context/hardware claim, statistical speed
  guarantee, or128K occupied-context validation. Hardware-specific flags remain opt-in.
- No default-change or unconditional PR sign-off. A production recommendation requires
  resolving/reproducing the historical issue with captured starting state, plus an
  explicit decision on acceptable validation scope and memory headroom.

The completed bounded R&D work should not be extended with repeated identical tests
or speculative kernel rewrites lacking material measured headroom. The code, negative
results, diagnostics and rollback commits are preserved for the next justified step.

## Rollback checkpoints

- `c730fbb`: prior scratch/reuse baseline (predates later correctness discovery).
- `4eb007e`: superseded-promotion ownership correction.
- `fd22a61`: full adaptive graph widths.
- `148d423`: confidence integration evidence.
- `c2976bf`: production ownership metadata tests.
- `f7aea5b`: writing qualification.
- `0558c57`: real DMA ownership tests and timing budgets.
- `fae2b46`: causal MMQ grouping replay and sanitizer evidence.

All experimental implementation/evidence is local. Raw large tensors/profiler captures
stay local; compact evidence is committed. See the plan, checklist and chronological
work log for individual decisions and the preserved unsuccessful tests.
