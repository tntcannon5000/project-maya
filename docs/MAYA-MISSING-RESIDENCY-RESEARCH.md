# Missing-expert residency research

2026-10-13. Starting checkpoint: retained `86c6ea0`. Scope: source review and
offline fixed-route replay; no new inference, GPU microbenchmark or build was
run by this research task. An explicitly requested, opt-in header prototype is
included for the parent campaign to qualify. All eight experts, immutable mixed
Q2_K/Q3_K/Q4_K payloads and the existing CPU plan remain required. No MTP, K4V,
expert skipping or new quantization is proposed.

## Decision

There is no established substantial residency win in the available evidence.
The best bounded causal candidate is an online, completed-window reuse model:
learn whether an expert returns within the next **16 observed same-layer
routes**, rather than only the next route. It can reuse the existing host policy
matrix and eviction interface without adding speculative uploads, router GEMVs,
CUDA events or GPU allocations. Its repeat-only replay improves upload traffic
by 3.3–4.6%, but first requests regress by 3.2–4.0%. Across both requests, CSV
regresses 0.72% and web improves only 0.30%. This warrants one opt-in feasibility
screen, not retention or a speed claim.

A second, potentially stronger mechanism is **activation-conditioned resident
value prediction**, used to choose residents and admissions rather than to fetch
unrequested experts. It is technically feasible, but the current traces lack
the activations needed to establish feasibility. Its first implementation should
be bounded feature collection and a task-disjoint offline decision test. A
learned predictor or a published MoE speedup does not establish byte savings on
Maya.

Do not implement a static frequency admission margin, a static writing/web
conditional model, generic LRU, or rank-0 prefetch as a missing-byte solution.
The retained policy is already nonuniform and causal; replacing it is a real
policy comparison, not filling a missing feature.

## What actually limits the retained configuration

[Three-costs results](evidence/maya-three-costs/RESULTS.md) establish 45.90 repeat
TPS on the short fixed continuation and 42.25 TPS at 8192 occupied tokens. The
short candidate still uploads 383.000 MB/token and serves 60.461 experts/token
on the CPU, with 564.156 MB/token of CPU **logical** weight payload. These two
numbers describe different consumers of missing experts. Their sum, about
947 MB/token, is a useful logical demand measure; it is not measured physical
DRAM traffic. Cache lines, CPU dequantization reads and controller traffic are
not established by those counters.

The repeat diagnostic has 10.377 ms/token of fetch intervals at 36.908 GB/s
effective payload throughput, including 1.271 ms overlapping computation. Its
disjoint exposed-fetch partition is 9.107 ms/token; CPU-answer waiting remains
3.160 ms/token. Raw fetch durations and CPU waits cannot be independently added
as predicted speedups.

The 22 GiB pool contains 2693 slots over 42 layers. Three reserved spares per
layer leave 2567 identities available as ordinary residents, and the RAM tier
contains 11819 slots / 97.822 GiB. Warm short D2H is only 1.137 MB/token. More
RAM shadows would largely repeat the already retained demotion improvement.
The 8K lifecycle still has 86.394 MB/token D2H after lending/reclaiming, so short
warm results do not certify the long-context case.

[Churn audit](evidence/maya-churn/RESULTS.md) finds approximately 70% resident
selection hits, despite only about 21.2% of expert identities being resident.
Repeat loss-to-return upload traffic is 7.90/8.01 MB/token within four tokens
and 54.46/59.87 MB/token within sixteen, for CSV/web respectively. These are
observed reuse categories, not obtainable savings: keeping those experts can
displace more useful residents. The large opportunity is not a demonstrated
one-token eviction/refetch bug.

## Source map and complete dependencies

| Location at `86c6ea0` | Existing behavior and why it matters |
|---|---|
| `src/core/glm_fast_path.cu:1051–1155`, `fast_setup` | Slots are already allocated by routing share with floors and a byte cap. Profile order is converted to artificial `1/(1+global_rank)` shares, or `expert_counts.txt` supplies actual normalized shares. Different quant classes have different slot strides. |
| `src/core/glm_fast_path.cu:1243` | `MAYA_REUSE_POLICY=transition` initializes the retained causal policy. `tools/run_three_costs.py` explicitly enables it. |
| `src/core/glm_reuse_policy.hpp`, `observe` | Current same-layer route conditions a next-route transition distribution, with 16-observation frequency smoothing and row aging at 2048. It predicts eviction value; it does not move weights. |
| `src/core/glm_fast_path.cu:3564–3597`, `3826–3832` | Service bookkeeping observes actual IDs. Ordinary c=1 policy observation occurs after CPU-answer publication. Access is serialized through the existing service/boundary mutex. |
| `src/core/glm_fast_path.cu:4245–4292`, `fast_boundary` | Aged counts/recency choose a fallback victim; the transition score chooses the actual completed resident victim. An already held immutable RAM shadow permits an immediate spare refill; otherwise D2H ownership is retained until completion. |
| `src/kernels/cuda/glm_fast.cu:3133–3229`, `moe_route_kernel` | The device first assigns colder missing RAM experts to CPU using the existing `011223344` plan, then promotes GPU-fetched experts into at most three spares; the remaining fetches use global scratch. Admission is usually eager and route-order based. |
| `include/strata/kernels/glm_fast.hpp:191–255` | `kSpares=3`; spare tables, `MoeDev`, `MoeRequest`, ring publication and fences are shared protocols. Changing spare count or request layout is not a local eviction edit. |
| `src/core/glm_fast_state.hpp:327–350` | `LayerPool` owns contiguous main/tail regions. `slot_of`, slot state, pointer-to-slot translation and host mirrors all assume per-layer ownership. |
| `src/core/glm_fast_path.cu:4743–4819`, `3233–3270` in the CUDA kernel | Existing prediction applies the next layer's actual router to this layer's FFN input, then can claim its spare slots and upload predictions. High selection accuracy does not prove baseline GPU demand was displaced. |
| `src/core/glm_fast_path.cu:4838–4839` | Retained component overlap explicitly requires prediction off. A new activation value predictor must be decoupled from `max_pf` and avoid the existing prefetch side-stream ownership. |
| `src/core/glm_prefill.cu:1007`, `1866–1950`, `2293–2346` | Lending and resident MMQ use the layer's contiguous partition and slot bounds. A global shared cache needs a compatible prefill mapping and a lending/drain protocol, not merely a new slot allocator. |

Line numbers in this map refer to the retained baseline; additions to the reuse
header change its own line numbers. No route weights, reduction order, fetch
band ordering or CPU-answer fences need changing for the completed-window
prototype. Residency can still change later CPU/GPU assignment, so unchanged
weights and eight computed experts do not by themselves prove exact logits.

## Reproducible offline feasibility

New tool: [`tools/research_missing_residency.py`](../tools/research_missing_residency.py).
Saved numerical record:
[`residency-research-replay.json`](evidence/maya-missing-residency/residency-research-replay.json).

Run from the repository:

```powershell
.venv/Scripts/python.exe tools/research_missing_residency.py
```

It reads the existing churn route logs, prior profile, actual native-expert blob
sizes and historical pool capacities. It asserts that its retained-policy clone
matches `audit_residency_churn.replay` traffic exactly. Source commit and SHA256
of code, metadata, profile, routes and input tokens are saved. It never launches
the engine, compiler or profiler.

The replay retains the existing global CPU plan, aging, transition eviction and
at-most-three GPU-fetched admissions. It does not reproduce asynchronous slot
states, exact device counters/ties, D2H scheduling or batched-prefill lending.
Actual repeat upload is 383.000/382.490 MB/token; baseline replay is
383.163/382.641. Repeat residency-mask disagreement is 0.084%/0.116%; first
request disagreement is 1.086%/1.616%. Good baseline fidelity is not proof of a
counterfactual's full-engine fidelity.

Static models train only on the existing `train-writing` and `train-web`
requests. CSV is a task-disjoint static-model holdout. `current-web/tokens.txt`
has the **same SHA256 as `train-web/tokens.txt`**, so web is explicitly a training
domain check, not a second holdout. The online models have no offline training
task: at each decision they use only past/current observations. This is one
fixed feasibility screen, with no parameter fitting to CSV or iteration after
its results.

| Fixed policy / repeat upload MB/token | CSV | Web | Interpretation |
|---|---:|---:|---|
| Retained transition replay | 383.163 | 382.641 | Reference |
| Static four-route return model | 730.777 | 372.732 | Severe domain failure; reject |
| Static sixteen-route return model | 747.304 | 379.590 | Severe domain failure; reject |
| Drift-gated static sixteen-route mix | 381.283 | 380.644 | Nearly falls back; mean mix 0.00478 / 0.01456; no substantial gain |
| Pure online completed sixteen-route model | 370.683 | 365.063 | Repeat opportunity; first-request regression below |
| Confidence-mixed online sixteen-route model | 378.604 | 379.383 | Mean mix 0.03910 / 0.03751; small, inconsistent pooled gain |

The static model counts `P(expert appears at least once in the next H routes |
one current expert)`, averages over the current eight experts, and uses 16
prior anchors. Labels never cross a training request boundary. Four/sixteen
models have only 296/272 complete training anchors per layer, respectively.
They cannot be described as a broadly trained domain predictor.

The conservative static mix is predetermined: maximum weight 0.25 multiplied
by `(1-total_variation)^4` between the last 32 observed routes' expert marginal
and the training marginal. Before 32 observations it uses the retained model.
The conservative online mix uses maximum weight 0.25 multiplied by
`mean_anchor_support/(mean_anchor_support+128)`. Both normalize component
scores before mixing. These rules guard shift/sample scarcity; they are not
formal guarantees and are not an inferred permission to tune the holdout.

| Pure online16 traffic, MB/token | CSV first | CSV repeat | Web first | Web repeat |
|---|---:|---:|---:|---:|
| Baseline GPU upload | 462.465 | 383.163 | 477.297 | 382.641 |
| Online16 GPU upload | 481.052 | 370.683 | 492.329 | 365.063 |
| Baseline CPU logical payload | 651.686 | 564.564 | 660.543 | 563.780 |
| Online16 CPU logical payload | 665.959 | 549.647 | 679.677 | 544.964 |

CSV repeat logical host demand falls 947.727→920.330 MB/token; web falls
946.420→910.026. Both first requests increase. Equal-length first+repeat GPU
upload is CSV 422.814→425.867 MB/token (+0.72%), web 429.969→428.696
(-0.30%). The confidence-mixed online variant is CSV -0.57%, web +0.08%
pooled. These are insufficient evidence of a useful end-to-end speedup.

## Mechanism 1: completed-window online reuse — opt-in prototype

Implemented only in `src/core/glm_reuse_policy.hpp`. Set
`MAYA_REUSE_POLICY=transition` and `MAYA_REUSE_HORIZON=16`; unset/other horizon
values preserve the literal default one-route update and arithmetic. This
prototype is unbuilt and unqualified by this research task.

Each layer stores a fixed 17-entry ring of up to eight expert IDs per route.
After observing route `t`, it can train anchor `t-16` using the union of IDs
already observed at `t-15` through `t`. At time `t`, **no demand after t is
consulted**. Each anchor/target pair increments once per completed window, not
once per repeated occurrence inside it. The current route then queries those
completed historical anchors for resident value. Smoothing uses
`1-(1-frequency_share)^128` as a marginal prior over sixteen eight-selection
routes. It is a heuristic prior, not an independence claim about actual routing.

Existing `transitions`, `seen` and `score` storage/interfaces are reused; no
second 288×288 table is added. Fixed ring and scratch vectors add less than
0.1 MiB across 46 layers. A mature union has at most 128 IDs and is updated
sparsely. There is no allocation during `observe`; the prior uses seven
squarings instead of a libm power call per expert. Existing row aging remains
2048. New public fields `horizon` and `matured_anchors` let the parent log model
engagement alongside `observations` and `changed_victims`.

`matured_anchors` counts complete layer-window labels, not distinct experts or
scored tokens. Each complete anchor contributes one sample to each of its up
to eight anchor rows; `seen[e]` is that row's effective, possibly aged sample
count. A layer with n observations has at most `max(0,n-16)` matured windows
before accounting for reinitialization. Warmup and prompt/service observations
must be identified separately from scored decode. Batched prefill may not call
this policy for every actual token, so a 16-observation window spanning a
prefill/request boundary is not automatically a 16-position language-model
window. Do not report an exact token horizon without instrumenting that path.

This fits the existing serialized host policy flow and cannot add speculative
weight traffic. It does add host scoring/update work; source-level boundedness
is not a latency measurement. Its repeat byte saving of 12.48–17.58 MB/token
corresponds to only 0.34–0.48 ms of raw fetch duration at the observed
36.9 GB/s, before overlap changes, CPU effects or policy overhead. Early
requests lose in replay; no retention recommendation follows yet.

Bounded campaign qualification: one short two-request CSV case, with the
retained engine and only the horizon changed; preserve first/repeat byte,
CPU-selection, logits and policy counters separately. Parent-selected
GPU-only exact-logit isolation can distinguish ownership/arithmetic errors
from hybrid assignment differences. A surviving candidate then needs one
task switch or 8192-occupied case, chosen for the actual unresolved lifecycle
concern. Reject on pooled regression or numerical failure under the existing
qualification rule; do not relax that rule because placement changed.

## Mechanism 2: activation-conditioned value — feasible, not proven

ProMoE's learned predictor establishes a concrete way to exploit intermediate
layer states. Its skip-based predictor resembles Maya's already available
next-router-on-current-input path. Crucially, the paper predicts **later layers
of the current token**; predicting an expert's return over the next sixteen
tokens is a different learning problem. The static route-conditioned holdout
failure above is evidence that task/context conditioning must be established.

A bounded Maya implementation can use 64 predetermined sampled coordinates of
the actual `MoeIo::x`, current route IDs, frequency/recency and a phase marker to
predict return value. Start with a linear 64→288 model per layer: about
3 MiB FP32 coefficients per horizon for 42 layers, rather than adopting the
paper's approximately two-million-parameter MLP per layer. A 64-float feature
record for 42 layers is 10.75 KB/token, around 176 MB for 16384 tokens. Feature
projection, collection bandwidth and inference cost must still be measured.
Sparse coordinate sampling avoids 42 new standalone projection launches.

Exact prospective locations: sample `io.x` in `fast_moe_experts` and publish a
small opt-in feature record through cooperative route publication in
`moe_route_kernel`; define the ABI in `MoeDev`/`MoeRequest` in
`include/strata/kernels/glm_fast.hpp`; retain ownership in `FastState`; consume
the result in `GlmReusePolicy`/`fast_boundary`. Do not enable `max_pf` merely to
collect prediction features: that enables uploads and conflicts with retained
down-band overlap. Captured graphs, ring allocation size, STOP/recovery and
request sequence fences must all include any changed request layout.

The first experiment should predict and log only: train on disjoint prompts,
evaluate on unseen task/context sequences, and ask whether byte-weighted
resident decisions beat the retained transition after counting displaced hits,
GPU/CPU assignments and uncertainty. An activation classifier's top-k accuracy
alone is not a cache-traffic result. Only after that gate should it alter
resident value or the relative ranking of actual fetched admission candidates.
Actual-route computation always remains exact and complete. This mechanism
could address the medium/long-return pool, but the current data do not support
a numeric attainable saving.

## Why other residency approaches are lower priority

**Reclaiming reserved spares.** Three spares/layer occupy 1,116,536,832 payload
bytes, about 1.040 GiB; one spare per layer is about 0.347 GiB. A capacity-only
replay keeping the same policy estimates the following repeat traffic:

| Hypothetical added residents/layer | CSV MB/token | Web MB/token |
|---|---:|---:|
| 0 | 383.163 | 382.641 |
| 1 | 377.927 | 376.550 |
| 2 | 372.340 | 371.285 |
| 3 | 365.428 | 365.706 |

This does **not** allocate real memory or implement spare reclamation. It treats
the capacity as extra hypothetical resident slots while leaving replay admission
semantics intact. Reclaiming all three promises only about 17–18 MB/token
(4.4–4.6%) in this model, roughly 0.46–0.48 ms raw fetch duration.

A practical implementation would keep one real spare and turn two completed,
RAM-backed resident slots into lazy replacement descriptors. The actual route
would protect all eight selected hits, then replace an unselected descriptor
only when a GPU fetch needs its slot, logging the old key for host reconciliation.
An unbacked/in-flight victim must remain owned and the request must fall back to
existing scratch. This needs device replacement metadata, request ABI changes,
`slot_of` reconciliation and prefill lending qualification. Merely lowering
`kSpares` breaks shared assumptions. The limited capacity benefit does not
justify this ownership work before cheaper policy/transport candidates.

**Global cache or runtime slot reallocation.** A shared quant-class pool could
react to task-specific layer demand and avoid stranded per-layer capacity.
However, main/tail addressing and resident MMQ currently use contiguous layer
partitions. The historical trained layer allocation removed only about 2% of
actual demand, so no large gain is established. A class-global allocator also
needs causal marginal fault curves, compatible prefill gather/mapping, borrowing
boundaries and accounting for migration bytes. A future-demand oracle cannot
provide that allocator or a rigorous engine lower bound.

**New admission margins.** The strict 1.5× aged-frequency rule already increases
repeat upload 10.94%/12.32% in faithful replay. Actual promotion nearly covers
all GPU fetches: baseline repeat scratch uses are only 30/16 across 128 tokens
in the research replay. Ranking the fourth fetched candidate ahead of the third
is therefore a small standalone target. Any broader admission change needs
contextual evidence that the rejected candidate is less valuable than the
resident it preserves; a generic count threshold repeats a negative result.

**Existing prefetch.** Historical selective prediction reduced demand upload
413.79→373.29 MB/token but added 79.90 MB/token prediction, raising total
uploads to 453.19. Of the predicted bytes, 76.54 MB were selected; many replaced
CPU work rather than baseline GPU transfers. Timing or selected-byte precision
cannot excuse net traffic inflation. Demand-only prefetch can hide transfer
latency but cannot, by itself, remove total demand bytes. Use a full
counterfactual GPU/CPU/eviction ledger before treating it as a missing-traffic
mechanism.

**Lossless compression.** Existing sampled quantized experts compress only
2.9% with LZ4 or about 6.7% with Zstd. CPU-decoded transport was far slower than
the saving. Even a free GPU decoder would save only about 11–26 MB/token from
383 MB/token under those sampled ratios; whole-model compressibility and GPU
codec costs remain unestablished. A novel structured encoding is research,
not a demonstrated substantial byte reduction.

## Savings scale and campaign implication

At 45.901 TPS, the scored loop is 21.786 ms/token. Reaching 60 TPS would require
removing about 5.12 ms/token. At the diagnostic's 36.9 GB/s, pretending every
removed upload byte converts directly to critical-path time would require
roughly **189 MB/token**, almost half the current GPU upload. Neither the
completed-window repeat opportunity nor reclaimable spares approach that
scale, and first-request/CPU/overlap interactions make that optimistic conversion
less reliable. Residency work should be evaluated jointly with the separate
host-traffic/CPU contention investigation, without adding purported gains.

## References and verification status

- [Three-costs results](evidence/maya-three-costs/RESULTS.md), retained trace
  `deep-summary.json` and `partition-means.json`: measured bytes, stages,
  capacity and numerical/lifecycle qualification.
- [Churn audit](evidence/maya-churn/RESULTS.md) and its `audit.json`: observed
  return categories, admission survival and rejected causal admission margin.
- [Demand results](evidence/maya-demand/report/RESULTS.md) and
  [research log](evidence/maya-demand/RESEARCH.md): negative policy/prediction
  screens, CPU/GPU attribution and simplified oracle limitations.
- [Transfer research](evidence/maya-transfer/RESEARCH.md): immutable shadow
  backing, lossless compression samples and failed transport combinations.
- [ProMoE full paper](https://arxiv.org/html/2410.22134v3), sections 4–6,
  retrieved 2026-10-13: learned intermediate-state prediction, prediction
  latency/accuracy, prefetch interference, per-layer LRU and training details.
  Its published speedups and default CPU predictor placement are not Maya
  estimates; this CPU already computes experts.
- [Fiddler](https://arxiv.org/abs/2402.07033) and the archived local abstract
  support treating CPU execution as an alternative to weight movement.
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
  supports pinned-memory/event ownership and overlap dependencies; it does not
  prove a net performance gain from another stream.

Offline replay completed and its retained-policy equivalence assertion passed.
The reuse-header diff passes `git diff --check`. The opt-in C++ prototype has
not been compiled or run here. The parent campaign owns actual byte, numerical,
latency and lifecycle qualification. No commit or external publication was made.
