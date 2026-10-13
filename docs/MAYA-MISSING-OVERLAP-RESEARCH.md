# Remaining transfer and overlap opportunities

2026-10-13. Research against retained `86c6ea0`, with source locations below
referring to that revision unless explicitly labelled otherwise. No inference,
build, transport benchmark, or engine edit was performed by this investigation.
Saved CSV/SQLite evidence was aggregated on the CPU. This document is the only
file written. The parent campaign owns implementation and execution evidence.

Keep the GSQ-RCO 3.0-bit Q4-attention pack, all required experts, the existing
CPU assignment, rank-order sums, RAM shadows, transition eviction and captured
MoE paths. No token speculation, MTP, K4V, expert skipping or new quantization.

## The important critical-path correction

The retained warm diagnostic has 383.000064 MB/token of logical demand,
10.377092 ms of fetch, 1.270564 ms of fetch/computation overlap and 9.106528 ms
of fetch without computation. Its profiled loop is 22.879977 ms, essentially
unchanged from the older diagnostic despite the independently confirmed
unprofiled scheduling improvement. These interval partitions are not separate
costs to add to other stage durations.

I read `retained-trace/deep-layers.csv`, selected repeat 1 (128 tokens, 5,376
layer routes), parsed every saved fetch/GU/down interval and summed interval
intersections. This gives a more useful bound than subtracting aggregate
overlap from all expert arithmetic:

| Warm time per token | All routes | Actual demand routes | Zero-demand routes |
|---|---:|---:|---:|
| GU plus shared down | 1.305732 ms | 0.814778 ms | 0.490954 ms |
| GU outside fetch intervals | 0.450194 ms | **0.000152 ms** | 0.450042 ms |
| Routed down | 0.980628 ms | 0.676614 ms | 0.304013 ms |
| Down outside fetch intervals | 0.565601 ms | **0.261588 ms** | 0.304013 ms |

There are 3,498 routes with demand and 1,878 without it. Recorded copy kernels
on zero-demand routes are still classified as fetch intervals, although they
perform no payload copies; their tiny intervals explain why zero-demand GU
duration and exposed GU differ. They are not useful transfer overlap.

The first down-copy band takes 1.657046 ms/token; the second takes 1.644808 ms.
The first band already hides effectively every GU operation on a demanded
route. Only 85 demanded routes have GU duration greater than their first-band
copy duration, and their aggregate exposed GU is negligible. Thus the remaining
1.015795 ms aggregate expert arithmetic outside fetch comprises mostly work on
routes with no transfer, plus approximately 0.262 ms of demand-route down tails.
That is not a one-millisecond schedulable same-layer opportunity.

For the usual demanded layer, the existing critical path is approximately

```
prefix + max(first-down-copy, GU) + max(second-down-copy, first-down)
       + second-down + remaining-CPU-wait
```

This is an explanatory max-plus model, not a replacement for recorded interval
unions. When both copy bands dominate their consumers, moving GU under the
second prefix half merely removes GU from an interval in which it was already
hidden. More intersection alone does not predict a faster token.

The parent has nevertheless implemented a bounded paired-prefix/GU-band trial.
Its first screen reportedly passes exact logits and improves throughput; that
is useful evidence to investigate transport/CPU interaction, not evidence of
an unopened millisecond of GU arithmetic overlap. The parent's run manifests,
fresh comparator and qualification must establish the result.

## Exact source contract

| Source at `86c6ea0` | Relevant behavior |
|---|---|
| `src/core/glm_fast_path.cu:4733–4762` | Router and shared gate/up jobs are submitted together through `gf::mv`. |
| `src/kernels/cuda/glm_fast.cu:4336–4373` | On CUDA, typed shared Q4 jobs launch before the generic BF16 router. Job independence permits reordering while preserving per-row kernels. |
| `glm_fast_path.cu:4795–4824` | Actual route, one request-accounting increment, then disk/source publication wait. |
| `glm_fast.cu:3127–3227` | CPU ranks, promotions, scratch destinations, plan weights and fetch sources are established in route order. A promoted miss still needs a copy. |
| `glm_fast.cu:3277–3332` | Immutable request payload and CPU activation are published before the request sequence, with system fences. |
| `glm_fast.cu:3335–3361` | `moe_wait` supplies disk pointers/table updates. A zero fetch source is not proof of residency until this fence. |
| `glm_fast.cu:3394–3421` | Whole/component copies use contiguous coalesced `uint4` loads/stores; each CTA loops through ranks and grid-strides through a component. |
| `glm_fast_path.cu:4846–4916` | Captured prefix/down fork and two down-row bands; every side-stream consumer joins before subsequent reuse. |
| `glm_fast.cu:3455–3514` | GU chunks independently compute 32 gate/up rows and one q8_1 h block. Shared down is the special `ei == k` branch. |
| `glm_fast.cu:3550–3581` | Down rows are independent; each row reads the entire h vector and sums ranks in original order. |
| `glm_fast.cu:4779–4808` | Original GU grid, null-shared convention and per-format dispatch. |
| `include/strata/kernels/glm_fast.hpp:193–220` | Rank-index plans/sources and default scheduling fields. |
| `glm_fast_path.cu:1767–1822` | CPU task graph: half the workers wait for a route-tagged prefix marker, per-expert h quantization follows all that expert's GU jobs, down then becomes ready. |
| `glm_fast_path.cu:3464–3488` | Early CPU answer uses a stable request snapshot but keeps boundary retirement unpublished until bookkeeping completes. |
| `glm_fast_path.cu:4036–4040` | Boundary waits GPU completion and all published service requests before source/table reuse. |
| `glm_fast_path.cu:3181–3205` | Error teardown joins compute, copy and side streams before releasing sources/graphs. |
| `glm_fast_path.cu:4951–4956`, `5080–5108` | Ordinary graph guard excludes windows, prediction/lookahead, and captures recurrent layers or stable MLA MoE suffixes. |
| `src/core/glm_layer_graphs.hpp:18–71` | Graph key and explicit route-request replay accounting. |

## Complete paired-prefix/GU-band design and its real budget

This is technically complete and can alter contention. It is not a justified
large arithmetic-overlap claim against the retained topology.

1. After route and `moe_wait`, derive `matrix_bytes = Ly.down_off / 2` and
   `row_bytes = matrix_bytes / n_ff`. Require native equal gate/up matrices,
   an even number of complete 32-row q8_1 groups (`n_ff % 64 == 0`), and exact
   16-byte copy alignment. Validate the dimensions instead of applying this
   to EXL3, arbitrary padding or mixed gate/up lengths.
2. Copy the first gate row band **and matching up row band**. A paired-range
   kernel maps a logical two-plane span to `row_first * row_bytes` and
   `matrix_bytes + row_first * row_bytes`. This gives two prefix launches total,
   rather than separately copying gate/up in four launches. Each packed byte
   has one destination and is copied once.
3. Run the first GU row band with its original rank storage while the second
   paired prefix band copies on the bounded side stream. Each routed GU
   chunk uses its original absolute chunk index for both matrix rows and hq.
   Its quantization is only over its own 32 output rows: there is no cross-band
   h reduction. A full h vector is still required before any routed down row.
4. Join the second prefix before second-band GU. Publish the existing
   `prefix_ready=route_sequence` marker only after **both** prefix bands.
   Publishing after half a prefix silently changes the CPU phase policy.
5. Retain the two down bands and final CPU addition. Compute shared down once.
   If it rides the first half-GU launch, its output partition changes from
   64 to 32 x-blocks at the model's 2048 h rows. The `per` calculation at
   `glm_fast.cu:3461` still covers every output row, and its dot is unchanged,
   but each shared CTA now owns twice as many rows; this can lengthen the
   critical first GU band. Do not accidentally offset shared chunks by the
   routed GU row offset or run shared down in both launches.

An alternative first launch keeps the original GU x-grid and returns unused
routed half-band blocks, preserving the shared-down partition; it costs empty
blocks. Compare only if a trace shows shared-down partitioning is material.
There is no basis for an automatic extra experiment.

Numerical hazards are an incorrect absolute hq chunk, using the band count as
the full up-matrix stride, partially quantizing a 32-row h block, or splitting
a down dot across h bands and changing its addition order. Ownership hazards
are issuing prefix copies before disk pointers are ready, letting a subsequent
route overwrite the rank plan before the final join, or error teardown freeing
RAM sources before both streams complete.

Engagement needs byte ranges, first/second GU row counts, exactly one shared
down, both joins and prefix publication order. At every route the GU row
counts must sum to the original required GPU rows. A copy launch on a route
with zero fetch mask is not engagement. Preserve exact logits and ordered
route/placement attribution, not just aggregate byte counters.

The ideal new arithmetic-overlap budget on actual demanded routes is near
zero at unchanged durations. A positive result must instead be investigated
as a change in copy bandwidth, smaller GU occupancy, CPU gate timing, shared
partitioning or overhead. Its enabled flags are not an explanation.

## Ranked next experiments

### 1. Software-pipelined mapped copies in the concurrent retained path

This is a specific new transport mechanism, unlike repeating integrated DMA
or another resident-GU split. `moe_component_kernel` presently executes one
`uint4` load/store per thread per grid-stride iteration. Stage four independent
aligned `uint4` loads into registers before their stores, with bounded tails,
while preserving the existing one-launch component API, rank order, grid caps,
source visibility and byte ranges. Issue independent addresses before waiting
for earlier loads; verify the compiler actually emits that ordering in SASS.
Use the same pattern in a paired-prefix copy if that candidate is retained.

The plausible mechanism is more outstanding mapped PCIe reads at the 32-CTA
down grid, not better coalescing: the current accesses already coalesce.
Four values add registers and can lower occupancy, so compilation is not a
performance result. Do not introduce noncoherent cached-load instructions or
alter host-memory attributes without separately proving reused RAM-slot
visibility. Ordinary immutable-byte loads avoid that new correctness problem.

Complete experiment: one static opt-in kernel choice inside the current
captured scheduling graph, with exact actual-address byte readbacks outside
timing, then one matched fixed-input engine screen with CPU work active.
Copy-only replay is a prerequisite screen, not the acceptance result. Inspect
prefix/down useful bandwidth, arithmetic duration, CPU-answer tail and full
token completion together. Stop if isolated gains fail to survive concurrency.

Useful scale bounds, all conditional rather than predictions:

- Raising the current 36.91 GB/s to the earlier **concurrent** actual-address
  replay's 43.21 GB/s would reduce 383 MB fetch duration by about **1.51 ms**.
- The earlier copy-only capability 51.95 GB/s implies approximately **3.00 ms**
  less fetch duration, but CPU-first already demonstrated that recovering
  isolated bandwidth can make the whole loop substantially slower.
- Raising only current down traffic (127.667 MB, 3.302 ms) to 51.95 GB/s has
  at most **0.844 ms** duration headroom. Much of that can be absorbed by
  CPU completion; it is not guaranteed exposed saving.

These reference cases use different traces/address windows. They bound the
scale of an experiment; they do not prove the current kernel is latency-limited
or predict achievable net savings. Record actual copied bytes per component,
nonempty ranks, chosen grid/unroll, source/destination addresses or stable
identity hashes, byte readback result, copy phase duration and whole-loop time.

### 2. Exact early router plus a genuinely bounded predictive staging screen

First expose the exact small lead cheaply: submit router-only `gf::mv`, call
the same `moe_route` with `n_ff_sh=0`, publish/wait for sources, then overlap
shared gate/up and its separate unchanged `swiglu_q8` with demand prefix copy.
The shared SwiGLU formula at `glm_fast.cu:2869–2875` matches the shared route
branch at `2972–2977`. Gate/up job kernels and actual route arithmetic remain
unchanged. Run shared down exactly once after shared hq is ready. Capture the
entire fork/join; leave prediction/window/prefill callers on their old path.

Read-only SQLite aggregation checked both kernels immediately preceding all
5,376 warm routers on their compute stream: typed shared Q4 projection, then
generic router. Shared projection costs **0.357460 ms/token**, router projection
0.191290 ms, and the shared-to-router gap only 0.010235 ms. Shared duration
median is 7.808 us/layer and p95 9.472 us. Thus moving the exact router before
shared projection creates only about **0.36 ms/token** of lead, before added
launches/contention. It advances CPU publication too, but the two gains cannot
be added independently. This is a small bounded experiment, not a several-ms
route-lead proposal.

A larger prediction experiment needs a changed mechanism and an opportunity
screen first. Prior full prediction added 591.6 MB/token; prior rank-0,
demand-first prediction selected 95.8% of its bytes but added 79.9 MB and
reduced demand by only about 40.5 MB. Selection precision did not establish
GPU-transfer avoidance. Neither policy should simply be re-enabled.

The complete changed design would use private, generation-tagged prefix
staging, with **no `tab` admission or spare claim before the actual route**,
and choose only predicted ranks expected to remain on GPU under the existing
CPU split. After the exact route has fixed CPU/GPU assignments, a staged
prefix may replace mapped reads only for an actual GPU-fetched rank with an
exact expert key, quant layout and ready generation. A device-to-device prefix
copy still fills the real promotion/scratch slot; original `fetch_src` remains
available for demanded down bytes. Selected CPU experts are counted as waste.

The prior transfer research already proposed private prefix staging; the new
required work is the graph-compatible ownership and CPU-selection filter,
not renaming the prior predictor. Keep `F->max_pf=0` and `n_ahead=0` so the
current graph guard and down pipeline survive. Use a separate device staging
record and fixed predictor/copy nodes, not mutable host `pf_pending` control
or shared `ev_pred`/`ev_pf` handles. All speculative work must join within its
own captured graph before teardown/reuse. A predictor that must extend graph
completion beyond useful work loses its lead and must be rejected.

There is a serious graph-boundary constraint: a previous layer graph cannot
leave a side-stream speculative copy unjoined for the next layer's attention
to hide without redesigning capture/ownership. Starting after current demand
copies provides only the current final-down/CPU-tail interval. Use a bounded
prefix fraction that can finish there, or explicitly design a larger paired
layer graph; do not silently disable all retained graphs. Dynamic MLA position
arguments make a generic full-token graph a separate substantial project.

A useful offline screen must count **actual GPU demand** matches and measured
available lead, not top-k set overlap. Existing historical route logs supply
IDs and masks, but do not contain a current predictor's logits/confidence or
all new staging-ready times. They cannot prove predictor precision on this
checkpoint. Count staged/ready/late/aborted bytes, selected-GPU reuse, selected
CPU waste, unused bytes, physically avoided mapped prefix bytes, D2D bytes and
graph completion stalls. Keep scratch within the same capacity budget, or
report the changed resident capacity explicitly.

For useful timely bytes U, a rough latency opportunity is U/current-bandwidth
minus predictor work, D2D movement, demand interference and exposed CPU delay.
**50 MB of timely avoided prefix reads is about 1.39 ms at 36.09 GB/s**;
that is a conditional target, not an observed opportunity. With no supported
current precision/lead estimate, this direction ranks behind the transport
kernel experiment. If the screen cannot support roughly that scale, skip the
full implementation rather than repeat the rejected prediction trials.

## Experiments that lack a new large opportunity

- Resident-GU splitting has already been completed, traced and rejected; GU
  row tiling is a different complete schedule but has the critical-path limit
  above. Do not infer a gain from extra overlap alone.
- Four or more down output bands have only approximately 0.262 ms of currently
  exposed demand-route down arithmetic to attack, before extra launches.
- More expert-level fetch/GU readiness does not remove the current first-band
  bottleneck. Prefix-to-GU gaps are about 1.7 us/layer; route-to-fetch totals
  only 0.047734 ms/token. Launch-after-readiness is not the missing 9 ms.
- DMA requires a materially different supported mechanism. Actual-address DMA
  replay was 43.61 GB/s versus mapped copy-only 51.95; complete integrated
  DMA plus resident overlap was already slower. A bulk DMA peak is insufficient.
- CPU-first and fixed-eight-worker concurrency are completed negative/neutral
  experiments. Extending CPU phase control needs a deadline for joint layer
  completion and new phase engagement, not just a longer throttle. The earlier
  transfer report gives that complete design; this report adds no new evidence
  that withholding CPU workers through all down traffic is beneficial.

## Verification and provenance

Scheduling/transport candidates require exact logits and the same ordered
routes, placement and packed bytes. Preserve fallback behavior for no demand,
all resident, CPU-mixed, disk response and invalid route. Check cancellation,
error joins and boundary/source ownership once a promising complete candidate
earns qualification. No relaxed numerical thresholds or changed backend
assignment should be hidden in a scheduling comparison.

Primary evidence: `docs/evidence/maya-three-costs/RESULTS.md`,
`retained-trace/deep-summary.json`, `deep-layers.csv`, `timeline.sqlite`,
`docs/evidence/maya-transfer-attribution/RESULTS.md`,
`docs/evidence/maya-transfer/RESEARCH.md`,
`docs/evidence/maya-demand/RESEARCH.md`, and
`docs/MAYA-THREE-COSTS-TRANSFER-RESEARCH.md`.

The new aggregations used saved repeat-1 token ranges only, asserted 5,376
route/GU pairs and identified the shared/router predecessor kernels on each
router's own stream. No kernels were run. The approximately 0.262 ms exposed
demand down and 0.36 ms exact route lead are stronger source/trace constraints
than the previous generic two-ms expert-arithmetic ceiling. A large next gain
requires a demonstrable transport, traffic or lead-time change.

## Read-only implementation review at `ebeb181`

The parent requested a review after implementing the paired GU bands. I read
the committed diff without building or running it. No CUDA correctness defect
was found in that bounded review: both paired-copy spans cover the native
gate/up halves exactly once; routed chunks use the full matrix stride and
absolute hq block; shared down ignores the routed row offset and runs only in
the first GU launch; prefix readiness follows the second-prefix join; and the
existing final down join protects both sources and the shared rank plan.

`launch_fetch_gu_band` calls `cudaGetDevice` and `cudaDeviceGetAttribute` at
host submission time. They read metadata and do not enqueue stream work or
implicitly synchronize the device, so their presence alone does not invalidate
stream capture. Successful capture/replay is the execution evidence. Replays
do not execute those host queries again. A setup-time/per-device cache would
avoid repeated uncaptured submission overhead, if it becomes material; copying
the older single global `sms` cache can instead choose the wrong count in a
heterogeneous multi-device process. This is not a demonstrated current cost.

One portability hazard was identified: the HIP specialized `gu_type == 16`,
4096-by-2048 RDNA3 path returns through `launch_expert_gu`, whose x-grid still
uses full `n_ff / GU_ROWS`. The shared GU kernel now adds `gu_row_first`, so
enabling bands there would repeat full first-band work and let the second
launch access chunks 32–95 rather than 32–63. This path is outside the CUDA
Q2_K/Q3_K/Q4_K campaign. Reject GU bands on HIP or update that specialized
grid before claiming support; no HIP qualification is inferred. The finding
was sent to the parent for its implementation decision.
