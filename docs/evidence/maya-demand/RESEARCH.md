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
