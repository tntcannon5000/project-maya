# Evidence, mechanisms and evolving decisions

Baseline `20af539`, not original stock Maya. Historical report remains in
`../maya-transfer/report/RESULTS.md`. Original and new kernels operate on the
same immutable packed GSQ-RCO weights. Experimental switches are opt-in.

## Source-grounded observations

- `fast_setup` distributes slots by a profile-derived artificial share
  `1/(1+global_rank)`, subject to floors and a byte budget. It already has
  nonuniform allocation: 49–87 slots/layer, 2,693 total. Do not claim baseline
  equal slots or no heuristic cache. Q4/Q3/Q2 blobs differ materially in size.
- `moe_route_kernel` assigns the colder missing experts to CPU and promotes
  GPU-fetched experts into spares. Host `fast_boundary` uses aged counts,
  recency tie breaking. Existing `lru` selects a different eviction policy.
- `moe_fetch_kernel` uses one SM-wide grid and loops over selected experts;
  it pulls the whole blob before gate/up. Component streaming instead pulls
  gate/up first and down on a side stream while the single original gate/up
  computes. No extra gate launches or changed reduction order are required.
- `moe_cpu_wait_kernel` waits at the layer end. Global CPU plan ignores the
  actual *exposed* waits per layer. A measured per-layer plan is a bounded
  alternative; fewer CPU experts also increases PCIe bytes, so improvement
  is not guaranteed.
- Maya already has KDA layer graph support and capture/replay counters.
  Its opt-in path is worth a complete measured trial, rather than writing a
  second graph system. D3 currently refuses graphs/prediction combinations
  explicitly: event/stream ownership requires separate qualification.

## Bounded arithmetic inspection

`arithmetic-inspection.json` aggregates kernel durations only within the 256
scored retained trace token ranges, excluding intervening prompt forwards.
Average CPU-answer wait is 2.804 ms/token. Largest arithmetic family is Q4_K
dense projections (~3.20 ms), already using batched projection launches.
Q2_K expert gate/up ~0.976 ms and down ~0.666 ms; other quant classes add
smaller contributions. No measured redundant weight reads or occupancy
defect was established. Do not infer hardware saturation from these numbers;
equally, do not begin a generic arithmetic rewrite from duration alone.
The component-overlap ceiling is nearer total gate/up time (~1.2 ms) than
the optimistic original 3 ms. Exposed waits and fetches remain higher priority.

## Capacity / reuse study

Collected actual routes, IDs, fetch/CPU/disk masks on retained build. Route
logging itself requests 16 near-miss ranks, adding GPU work: route-logged
screens are diagnostic, not the final unprofiled headline. Initial diagnostic
runs overlapped a CPU build; their timings are excluded from final comparisons.
Held-out slot training uses explanation/web; CSV is evaluation only.

Offline replay over 316 prompt+continuation token histories estimates:
original LFU 411 MB GPU demand/token, LRU 467 MB, future-oracle 228 MB.
Trained allocation/seed: LFU 402 MB, oracle 215 MB. These are *modelled replay*
values, not engine traffic or achieved speed: replay simplifies promotion,
spares and CPU expert identities, and assumes routes are fixed. The oracle
has inaccessible future knowledge; it is an optimistic capacity opportunity,
not a rigorous lower bound for the full engine. It suggests traffic is not
entirely forced by capacity, but held-out frequency allocation alone promises
only ~2% byte reduction (~0.25 ms), below ordinary timing variation.

## Public and local grounding

Use the sources archived by campaign one: CUDA Best Practices (pinned-memory
overlap and asynchronous ownership), Fiddler (CPU/GPU assignment), ProMoE
(expert prediction), plus local Maya CUDA graphs and actual DS4/Strata traces.
Links and retrieved copies are indexed in `../maya-transfer/RESEARCH.md`.
Public proposals establish mechanisms, not results on this model/hardware.
DS4 independent transfer/graph experiments and prior Maya resident overlap
show that extra streams/events can lose even with real overlap. D3 removes
the earlier duplicate gate launches, while D4 changes transfer work mapping
without another host submission/handshake. Those are specific reasons to
revisit these directions, not repetitions of enabled flags.

## Status

Screens, mechanism traces and final decisions will be appended. No performance
or parity result is inferred solely from compilation or successful startup.

## First implementation screens

Diagnostic route logging includes near-rank selection; do not compare these
directly to an unlogged headline. Retained route-logged first/repeat: 37.31 /
40.04 TPS (initial CPU build overlap makes it provisional). Fresh unlogged
retained baseline: 38.22 / 41.39 TPS.

| Hypothesis | Diagnostic first/repeat TPS | Evidence / evolving decision |
|---|---|---|
| D1 trained placement | 37.69 / 40.69 | Demand 463.6/406.3 MB vs 472.9/413.8; small held-out byte gain, not yet a reliable speed gain |
| D2 LRU | 37.21 / 39.17 | Repeat demand rises to 433.8 MB; reject |
| D3 component streaming | 36.50 / 39.49 | Exact all logits; standalone speed does not support retention; graph dependency and overlap trace pending |
| D4 batched pull | 37.30 / 40.27 | Exact all logits; negligible standalone change; combined graph trial 38.61/41.69 |
| D5 layer graphs | 38.85 / 41.91 | 34 captures, 10,710 replays, zero invalidations; exact all logits. Needs unlogged confirmation |
| D6 per-layer lane balance | 37.54 / 38.90 | Traced cost model leaves 35 of 42 plans unchanged; only seven change. Little transfer reduction; reject standalone |
| D7 rank-0 demand-first prediction | 37.37 / 40.11 | Repeat demand 373.29 + prediction 79.90 MB; 76.54 MB actually used (~95.8%). Total upload increases, no gain |

D6 did not blindly remove CPU experts because waits exist. Measured median
CPU/expert, fetch/expert and gate/up/down durations minimize max(CPU lane,
GPU lane) per layer and miss count; those estimates largely reproduce the
existing global plan. Reducing a wait can be worse than leaving it if the
alternative transfer is slower. Training uses retained CSV traces, so any
retention would require writing/web hold-outs.

D7's "used" means selected by the next route, not necessarily a transfer
avoided versus CPU execution. High selection precision is insufficient.
The candidate CPU/GPU split and cache placement change, so attribution needs
actual route/mask comparisons rather than assuming forced inputs fix routes.

The D3+D5 fork/join is implemented inside CUDA stream capture: the side
stream joins via the down-ready event before the dependent down projection.
This removes duplicated gate launches from the previous campaign and allows
graph replay to test the submission-cost dependency. Prediction+D3 remains
explicitly excluded because it would share event/buffer ownership.

## Complete combinations and traces

- D3+D5 ran successfully: 34 captures, 10,710 graph replays; 39.10/41.82 TPS
  with route logging. D4+D5: 38.61/41.69. Neither beats simpler D5 reliably.
- Unlogged D5 confirmation: 39.31/42.85 vs fresh retained 38.22/41.39.
  D1+D5: 39.80/42.95, not a meaningful extra gain. These changes are small
  relative to the prior ~4% run variation; final task/8K qualification follows.
- Traces explicitly set `--cuda-graph-trace=node` to measure constituent
  kernels rather than treating graph launches as opaque work.
- Mean first+repeat D3+D5 shows 2.768 ms pull/arithmetic overlap, but GPU
  arithmetic grows 9.283→11.347 ms versus D5 and inactive gaps grow
  1.166→2.063 ms. Component copies engage; contention plus scheduling costs
  consume their benefit even without duplicate gate launches. Reject.
- Repeated D5 versus retained trace: wall 25.222→24.560 ms, inactive gaps
  2.004→1.176 ms, pulls 11.229→11.270 ms, arithmetic 9.244→9.288 ms,
  CPU-wait kernels 2.698→2.785 ms. The demonstrated improvement removes
  submission gaps; it does not reduce expert traffic or CPU wait.
- All D3/D4/D5 and graph-combination scheduling-only outputs matched every
  compared logit and ordered route exactly. Placement variants differ in
  ordered routes despite identical forced input histories. Route set overlap
  and exact fractions are recorded separately in parity.json.
- Prototype checkpoint `c3cb3a4`, branch `research/maya-demand-prototypes`,
  preserves all implementations. The retained candidate will use the original
  R1 engine plus its existing graph switch; experimental transfer/policy
  changes are removed from the final engine, not quietly left enabled.

## Final qualification and retained checkpoint

- Retained configuration/source checkpoint `799c34f` restores the R1 engine
  source exactly and adds `configs/maya-demand-graphs.json`. Both engine and
  lab rebuilt successfully. Prototype branch remains at `c3cb3a4`.
- Short repeated unprofiled performance: 41.39 -> 42.85 TPS. At 8,192 occupied
  tokens: 36.40 -> 37.81 TPS. Writing/coding free continuations show 3.3–4.2%
  pooled gains across two requests per task, with identical generated IDs.
  These are bounded observations, not precise universal speedup estimates.
- Final short and 8K comparisons each match all 39,649,280 logits exactly,
  with finite values and zero maximum absolute difference. No thresholds changed.
- Repeated 8K prefill: 3578.417 -> 3578.549 ms, effectively unchanged.
  Cancellation at eight tokens, recovery, teardown and dual-owner checks pass.
- Final stage diagnostics completed baseline/placement/LRU/batched-pull/CPU
  cases (128 scored tokens each). Optional prediction stage capture stopped
  during loading; it is explicitly incomplete and not used as evidence. Its
  earlier complete main experiment already establishes rejection.
- User updated testing scope: from now on select one or two decisive cases,
  not automatic full suites. The stage runner now enforces that and timestamps
  new evidence directories. No rerun of the optional interrupted case.
- Updated report contains byte accounting, memory/clock ranges, measured
  component-overlap losses, timelines, dependency findings and negative results.
  60 TPS was not achieved. No further inference is running; nothing published.
