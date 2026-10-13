# Transfer critical path: complete next mechanisms

2026-10-13. Read-only source/evidence research against `4f2f6e8`; source
references below refer to that revision. No inference, build, GPU experiment,
benchmark, or source mutation was performed for this report. Saved CSV/JSON
files were read and aggregated on the CPU. This report is the only file written.

Scope: the retained GSQ-RCO 3.0-bit Q4-attention pack, c=1, current RAM backing,
transition eviction and captured-only 32-block component overlap. Preserve all
required experts, packed bytes, arithmetic and rank-order reduction. No MTP,
token speculation, K4V, pruning, altered quantization or quality shortcut.

## Findings that should govern implementation

The available **exact same-layer arithmetic is small relative to transfer**.
Latest warm arithmetic is 9.223 ms/token, but routed gate/up plus shared down
accounts for only 1.245 ms and routed down for 0.792 ms. The other arithmetic
largely precedes routing and cannot hide that layer's actual demand without
earlier knowledge of the route. Existing overlap already hides 0.594 ms.

Thus an ideal same-layer pipeline has at most approximately
`1.245 + 0.792 - 0.594 = 1.443 ms` of additional overlap at unchanged arithmetic
duration. This is an optimistic aggregate bound, not a schedulable guarantee.
CPU completion can become the critical path as GPU work finishes earlier.
Captured component layers contain 0.935 ms of GU and 0.592 ms of down; the
whole-fetch layers contain 0.310 ms of GU and 0.200 ms of down. A captured-only
resident-GU change with the original single down launch therefore has only
about **0.341 ms of extra overlap ceiling**, not several milliseconds. Exposing
down computation during down transfer is the natural complete expansion.

There is a second, independent cause: **CPU expert co-execution reduces host
transfer bandwidth**. The live CPU-first ablation recovered 2.721 ms of fetch
time but increased wall time by 4.405 ms. Fixed eight workers did not improve
throughput reliably. A viable new host-memory mechanism must retain useful CPU
concurrency and minimize the joint finish time; restoring copy-only bandwidth
is not itself success.

Current warm GPU fetches are 383.000 MB/token and recorded CPU expert payloads
are another **564.156 MB/token** on CSV (564.187 MB on web), from the existing
churn audit. These payload counts are not physical DRAM transaction counts.
CPU caches, fabric and coherence still affect actual traffic. Merely moving a
GPU miss to CPU does not remove the shared host-memory workload.

The highest-confidence implementation direction is a **captured exact MoE
pipeline**, initially resident-GU plus demand-prefix, then output-row-band
down streaming if the first stage is insufficient. The largest causal transfer
headroom without reducing bytes is phase-aware host-memory scheduling, but its
net gain remains uncertain. Component caching and prediction are larger policy
projects and should earn an offline opportunity screen before live work.

## Evidence and source contract

Primary evidence:

- `docs/evidence/maya-decode-latest/RESULTS.md` and
  `latest/deep-summary.json`: warm wall 22.871 ms; fetch 10.088 ms;
  fetch-only 9.494 ms; overlap 0.594 ms; GPU arithmetic 9.223 ms;
  CPU-answer wait 2.956 ms; no GPU activity 1.156 ms.
- `latest/deep-transfers.csv`: 105.950 MB whole-expert transfers in 2.831 ms;
  184.700 MB GU prefixes in 4.974 ms; 92.350 MB down components in 2.284 ms.
  Mean prefix-ready to GU and down-dependencies-ready to down gaps are under
  one microsecond. Launch-after-readiness is not the missing milliseconds.
- `latest/deep-layers.csv`, aggregated by `component`: component layers have
  GU 0.935/down 0.592 ms; whole-fetch layers GU 0.310/down 0.200 ms.
- `docs/evidence/maya-transfer-attribution/RESULTS.md`: actual-address mapped
  replay 51.95 GB/s alone, 45.16 with CPU, 50.20 with routed GPU work, 43.21
  with both. DMA replay 43.61 GB/s. CPU-first live fetch is 7.542 ms versus
  10.263 ms with the same demand, placement and logits.
- `docs/evidence/maya-churn/RESULTS.md` and `audit.json`: approximately 70%
  resident selection hits; immediate refetch is small; strict frequency
  admission worsened repeated demand by 11–12%. No obvious short-term cache
  thrash mechanism remains to remove.
- `docs/evidence/maya-demand/RESEARCH.md`: broad GU/down overlap increased
  arithmetic 9.283 -> 11.347 ms and inactivity 1.166 -> 2.063 ms. Batched
  pulls and measured per-layer CPU plans did not beat their simpler baseline.
- `docs/evidence/maya-transfer/RESEARCH.md`: complete DMA + resident overlap,
  including expanded backing, lost to backing alone. Prediction reduced
  demand but added hundreds of MB/token and lost end-to-end. Lossless
  compression saved only 3–7% in measured samples before decode cost.
- `docs/MAYA-RESIDENCY-OVERLAP.md`: bounded down copies and graph restriction
  repaired the broad-overlap contention/submission defects. They did not
  make transfer cost disappear.

Relevant current source:

| Source at `4f2f6e8` | Contract relevant to the next mechanism |
|---|---|
| `include/strata/kernels/glm_fast.hpp:192–215` | `MoeDev` has fixed rank-index plans, sources, weights and CPU state. |
| `src/kernels/cuda/glm_fast.cu:3063–3084` | CPU ranks are the coldest RAM misses selected by the existing plan. |
| `glm_fast.cu:3086–3155` | GPU hits have a plan pointer and zero fetch source; CPU ranks have a null plan; promoted misses still have a nonzero fetch source. Disk misses initially have a plan but zero source. |
| `glm_fast.cu:3248–3274` | `moe_wait` publishes disk sources and table changes before any consumer. Classification using zero fetch source is safe only after this fence, unless route explicitly supplies a resident mask. |
| `glm_fast.cu:3307–3334` | Copy kernels loop across experts, use contiguous uint4 transfers and a configurable block cap. They currently publish readiness only at whole component completion. |
| `glm_fast.cu:3351–3420` | GU blocks are independent per rank and 32 output values. Shared down is the special `ei == k` branch. Each rank writes its own `hq` rows. |
| `glm_fast.cu:3429–3486` | Down output rows are independent. Eight warps compute expert dots; the same block sums ranks in original plan order. This permits output-row-band decomposition without splitting the reduction. |
| `glm_fast.cu:4666–4695` | Shared-down launch conventions differ when shared pointers are null. Preserve them; do not access `fetch_src[k]` for the shared block. |
| `src/core/glm_fast_path.cu:4650–4775` | Route, disk wait, prefix/down fork, original GU/down and CPU answer join. Existing overlap excludes prediction and EXL3/unaligned formats. |
| `glm_fast_path.cu:3656–3718` | c=1 service executes CPU work synchronously, publishes the answer before reuse scoring, then retires the route. |
| `glm_fast_path.cu:1664–1789` | CPU GU/down are row jobs with unchanged per-row dots; final CPU sum is in original expert order. Current jobs are contiguous and deliberately not tiny. |
| `src/core/glm_fast_state.hpp:94–157` | Workers' generation-tagged ticket ownership. `set_active` controls entry to a batch, not worker claims already in progress; it is insufficient for mid-copy-phase arbitration. |
| `glm_fast_path.cu:3911–3952` | Boundary waits compute and service, then serializes table edits. Current slot reuse assumes all consumer work joined compute. |
| `glm_fast_state.hpp:313–408` | Resident/spare/draining/lent/landing VRAM and RAM ownership states; demotions own both endpoints until events land. |
| `glm_fast_path.cu:3106–3118` | Error teardown joins compute, copy and component side stream before graphs and pinned sources are released. |
| `src/core/glm_layer_graphs.hpp:18–71` | Layer graph keys include CPU plan and residual pointers; replay explicitly restores request accounting. Extend keys if pipeline shape changes at runtime. |
| `glm_fast_path.cu:4795–4799`, `4843–4887` | Current graph path captures recurrent layers; MLA work has position-dependent host arguments. A separate MoE suffix graph can avoid recapturing dynamic attention. |

## Ranked mechanisms and minimum complete implementations

### 1. Captured resident/prefix/down-band pipeline

**Priority:** implement the complete pipeline in bounded stages. The resident
part alone is an enabling experiment with a small ceiling. Down-row streaming
opens the larger exact overlap opportunity while preserving the current down
sum. Neither stage adds expert dot products.

**Stage A: minimum resident-GU + demand-prefix implementation.**

1. Keep route and `moe_wait` on compute. Prediction remains off. After the
   wait, `fetch_src[rank] == 0` plus `plan_ptr[rank] != 0` identifies a true
   ready GPU rank, including the all-resident and CPU-mixed cases. A nonzero
   source identifies a demanded GPU rank, including an answered disk miss.
   Because classification occurs after `moe_wait`, no new mask allocation or
   route publication field is required.
2. Add a scalar GU phase to the launch argument or copied `MoeDev`: ordinary,
   resident-only, fetched-only. Ordinary must remain the default for other
   callers. Check phase only after the shared-expert branch and retain the
   existing null-plan guard. This avoids a shared block indexing source 8.
3. Record route/source-ready on compute, wait for it on the copy side stream,
   and start the prefix copy there. While it runs, compute resident GU using
   the same dot code, same rank indices and same `hq` destination. Execute
   shared down exactly once in this resident invocation.
4. Join prefix-ready on compute. Start bounded down copy on the side stream,
   and run fetched GU with shared pointers null. Keep the original down join,
   original `moe_down` and original single CPU-answer addition.
5. Capture the entire fork/join on recurrent layers. Initially retain the
   old path elsewhere. A graph launch must account for exactly one route.

The prefix now competes with resident arithmetic. **Do not fork the existing
170-block prefix unchanged and call that a fix**: the former broad-copy
contention defect returns. A 32-block overlapping prefix is a reasonable
first complete design, but its bandwidth must be measured. A natural expansion
is a bounded initial prefix tile during resident GU, followed by the full-grid
remainder after resident completion. This limits the low-grid interval rather
than slowing every byte; it needs an explicit resident-ready dependency and
disjoint byte ranges, not two racing kernels over the same destination.

Keep all rank-index activation storage. No scatter of ranks, replacement GU
math, shared recomputation or second quantization. Phase changes alter which
rows execute, not their equations. A launch counter alone is insufficient:
count executed resident/fetched expert rows and prove their sum equals the
original required GPU rows on every route. The two GU launches may still run
longer than one because of launch/block overhead. That is a rejection reason,
even though useful arithmetic is unchanged.

**Stage B: exact down output-row-band streaming.**

This is preferable to computing per-expert down partials into a new tensor:
each output row already computes the entire original rank-order sum in one
block. Split output rows, not experts or the summation.

- Divide `n_embd` into two aligned contiguous row bands initially. After
  prefix-ready, copy each down band across all fetched ranks on the existing
  side stream, recording separate band-ready events. The byte offset is
  `Ly.down_off + row_start * d_row`, and length is `row_count * d_row`.
  Native Q2_K/Q3_K/Q4_K row sizes satisfy the existing 16-byte alignment;
  validate this rather than assuming arbitrary formats do.
- Add `row_start` and `row_count` to a down-band launch. The existing dot
  function, warp-to-rank mapping, four-row grouping, `m += w[i] * part[i]`
  order, optional `cpu_flag` contribution and shared addition remain the
  same. Output rows store to their original `ffn` addresses.
- After GU finishes, main waits for band 0 and runs its down blocks while
  later down bands copy. Main then consumes remaining bands. At 4096 rows,
  two bands give approximately 512 blocks each, maintaining much more
  parallel work than one-expert or tiny-row launches.
- Join every band and keep CPU-answer addition after all GPU down output
  rows. The next layer must not read a partially produced `ffn`. GPU source
  and destination slots remain protected until the final join.
- Use distinct readiness events within an invocation and complete captured
  fork/join ownership. Reuse event handles only across serialized completed
  layer invocations, consistent with the existing graph lifetime contract.

Only if two bands engage and repay their cost should finer tiling be considered.
For small/no-fetch routes, a device-side ordinary band path or a safe unchanged
fallback avoids pointless extra work; route-dependent host graph rewrites would
recreate submission overhead. Do not globally change prefill or window paths.

**Stage C: capture the MoE suffix on uncaptured MLA layers.**

There are 11 whole-fetch MoE layers in this trace. Their GU/down totals are
0.310/0.200 ms, so extending exact overlap there has a modest ceiling. Capture
only the stable MoE suffix using the persistent `F->x`, `xq`, routing, shared and
FFN buffers after attention/normalization, rather than capturing variable
position-dependent MLA arguments. Include pipeline configuration/CPU-plan
identity in the graph key and preserve `expected` request accounting. Measure
the extra graph-launch boundary: the prior all-layer event path lost about
0.8 ms of inactivity, larger than this arithmetic opportunity.

**Counters and decisive experiments:**

- Per route: ordinary/resident/fetched GU row counts; shared-down count;
  total required GPU rows; actual prefix/down-band bytes; CPU/disk masks;
  mixed resident-and-fetch route count. No extra near-rank router work.
- Per band: ready, consumer start/end and residual copy interval. Trace graph
  nodes, actual block caps and copy/arithmetic interval intersection. Compare
  exclusive wall partitions, arithmetic union and whole token latency.
- For Stage A, use one fixed-input repeated request pair versus a fresh same
  binary retained path. Require exact logits, exact ordered routes/tier masks,
  unchanged demand, actual mixed-route overlap and unchanged CPU contribution.
  Do not retain it as a speedup if the added GU cost consumes overlap.
- If Stage A is weak but correct, test one Stage A+B combination against the
  complete simpler path. It is justified by the unopened down opportunity,
  not by a flag sweep. Reject on missing band engagement, arithmetic increase
  that cancels hidden transfer, higher inactivity, or no reproducible token
  benefit outside the existing approximately 4% timing variation.
- Only a useful combination earns one held-out web continuation and the
  existing long-context/prefill/lifecycle ownership qualification.

**Ceiling:** Stage A captured-only approximately 0.34 ms extra ideally;
Stage A+B captured-only at most 0.93 ms extra; all-layer GU+down ideal at most
1.44 ms extra. A practical gain of a few tenths to around 1 ms is plausible;
there is no evidence for several ms. Earlier CPU exposure and low-grid copy
slowdown can make the result zero or negative.

### 2. Route- and phase-aware CPU memory arbitration

**Priority:** strong causal reason to investigate, uncertain net outcome.
This changes CPU work issuance around actual transfer phases; it is not another
fixed worker-count experiment or blanket CPU-first mode.

The complete design needs a route-tagged transfer phase signal and CPU row-job
admission that responds inside a batch. Current `Workers::set_active` is not
enough: workers already in `work()` continue claiming jobs until the batch ends.

Minimum implementation:

1. Publish a small, cache-line-separated mapped phase record from graph nodes:
   route/source ready, prefix active/completed, down active/completed, plus
   route sequence. A completion marker must run after the entire copy kernel;
   one copy CTA finishing does not establish component readiness. Preserve
   system-fence publication and never let a stale route tag throttle new work.
2. Add a CPU-lane-specific controlled row scheduler, leaving default Workers,
   disk readers, prefill and windows unchanged. Permit only a bounded number
   of workers to claim contiguous GU/down jobs during a copy-critical phase;
   release all 16 when copy is finished or CPU completion becomes urgent.
   Put permit checks before claiming a ticket so no worker can own a job while
   waiting indefinitely for permission. Generation-tagged tickets, all rows
   exactly once and unchanged CPU reduction are mandatory.
3. Use the existing contiguous jobs initially. Do not replace them with tiny
   32-row jobs: the source records that such short streams lost substantial
   memory bandwidth on another tested machine. If responsiveness requires
   subdivision, bound the subdivision and measure its overhead.
4. Learn a small cost table over quant class and GPU-fetch/CPU-count classes
   from the existing concurrent path. The objective is joint layer completion,
   `max(CPU done, GPU down done)`, not minimum copy duration. A deadline override
   releases workers when projected CPU completion would become the exposed
   tail. No new router/dense GPU work is necessary.
5. No host CUDA API calls while a GPU readiness kernel waits for the same
   thread. CPU-only polling must be bounded and cancellation/error fallback
   restores ordinary concurrency; phase instrumentation failure must not park
   workers forever.

Record permitted/blocked jobs, controlled raw weight bytes, CPU start/GU/end
times, copy-phase boundaries, deadline overrides, and phase tag mismatches.
Keep CPU assignments and dots exactly the same. A specific useful policy must
show reduced **exposed layer completion**, not merely copy bandwidth restored
to 51 GB/s. Train on one fixed workload and require a separate workload before
retention. Compare one policy with baseline, then its pipeline combination only
if both independently engage. No broad quota/threshold sweep.

Reject if copy improvement is replaced by a longer CPU-answer tail, if permit
markers add appreciable GPU gaps, if smaller jobs degrade streams, or if the
policy cannot improve a held-out case. The live ablation bounds recovered fetch
time at about 2.7 ms for unchanged bytes on this trace, but recovering all of it
while keeping full CPU concurrency is not established. A realistic first
opportunity is approximately 0.5–1.5 ms net; even that is a hypothesis. The
phase controller is naturally complementary with earlier resident/down work,
which changes how much CPU work can be hidden and therefore changes its best
deadline. Gains cannot be added independently.

### 3. Earlier lossless weight prefetch without an extra GPU router

**Priority:** only after an offline precision/lead-time screen. This can, in
principle, overlap transfers with the much larger dense/attention interval
before actual demand. Same-layer pipelines cannot access that opportunity.

Use the existing causal same-layer next-route history as a host predictor of
the next token's layer demand; no additional GPU router GEMV is needed. This
is weight prefetch with exact fallback, not token speculation. The prediction
must not change expert requirements or skip misses.

Fix the specific former failure: predictions were published as resident before
routing, consumed spares, changed CPU/GPU placement and uploaded excessive or
CPU-served traffic. A new design stages at most a small budget of GU prefixes
in private, explicitly owned scratch, **without publishing them in `tab`**.
Actual routing and CPU allocation occur against the normal logical cache.
Only an actual GPU-fetched rank with matching immutable expert identity may
consume staged bytes. CPU-selected or unselected predictions count as waste.
Demand transfers have priority, and optional copies have a bounded outstanding
chunk rather than a long uncancellable full-expert kernel. Any late prediction
falls back without delaying demand. Account for scratch by deducting it from
the same pool budget or reporting a reduced pool; no free extra VRAM assumption.

Before implementation, replay saved routes and original CPU masks causally.
Measure prefix bytes that would match an actual GPU fetch, waste, confidence,
route forecast age and available future lead time. Existing high next-layer
selection precision does not establish GPU-fetch precision. The screen must
show a material number of useful bytes arriving early at a small waste budget
on a separate task. If it does not, stop. For a live experiment, counters must
include useful-on-time, selected-but-CPU, late, unused, cancellation, total
physical upload and demand delay. The baseline must be the complete current
pipeline, not an old transport path. Exact logical masks and logits are
required if staging leaves assignment unchanged.

This is the only proposed transfer mechanism here with access to several ms
of otherwise unavailable pre-route GPU work, but no saved result yet establishes
the required predictor/lead-time quality. There is no justified numerical gain
forecast. Previous 95.8% selection usefulness still increased total upload and
failed; it cannot be used as acceptance evidence for this design.

### 4. Component-granular GPU residency weighted by exposed cost

**Priority:** structural follow-up, not the immediate substantial-byte answer.
GU consumes two thirds and down one third of these native expert payloads.
Keeping only GU can fit approximately 1.5 times as many expert identities in a
given allocation, but it converts full hits into down transfers. This is a
capacity trade, not a 1.5-times hit-rate gain or free byte removal.

For fixed expert-use probabilities, both GU and down are required on every use.
Their expected bytes saved per cache byte are the same: `p * component_bytes /
component_bytes = p`. With a perfectly placed byte cache, caching the hottest
whole experts is consequently as good as component caching for raw traffic
apart from integer granularity. A component cache can help latency if prefix
arrival has much greater exposed cost than down arrival, or if its causal
admission policy is demonstrably better. That benefit must be measured, and
down-band streaming changes the cost model.

A complete design needs separate immutable GU/down tables and ownership,
component-aware allocation, exact CPU-tier source protection, format/alignment
handling, staging of absent components, promotion/eviction completion, and
prefill lending. Avoid making a prefix-only entry a full-ready `plan_ptr`.
Evaluate a cost-weighted causal simulator on saved route histories and hold out
a task before investing in this engine rewrite. Report total GPU bytes, CPU
weight bytes, prefix waits, exposed down waits and lost full-hit down traffic.
The current churn audit disproves simple frequency margin and recent-32 policy;
do not reintroduce them under component terminology.

No credible several-ms raw-byte ceiling is established for this design. The
hard bounds are the actual prefix traffic and added down traffic, not an
oracle from the old simplified whole-cache replay.

## Directions not justified by present evidence

- **Ordinary DMA replacement:** previous complete integration lost, and actual
  source-address replay gives 43.61 GB/s DMA versus 51.95 mapped. A new complete
  DMA design would need to repair host route submission/graph ownership and
  demonstrate bandwidth under the live CPU workload. A CUDA batch API or bulk
  microbenchmark alone does not meet that condition.
- **More copy blocks / parallel expert grids:** existing per-expert loop has
  contiguous uint4 access and roughly 52 GB/s isolated real-address capability.
  Batched-pull integration was already neutral. A work-mapping rewrite needs
  evidence of a specific inefficient request shape, not a generic occupancy
  claim. More concurrent CTAs recreate proven arithmetic contention.
- **Zero-copy expert dot products:** they retain PCIe bytes, hold arithmetic
  CTAs during host-memory stalls, and promotions would need additional copies.
  They remove a small VRAM staging read/write rather than the dominant link
  cost. A long fused kernel must not relabel transfer time as arithmetic and
  claim that the transfer interval vanished.
- **Compression:** measured lossless codecs remove only 3–7% of packed bytes.
  Perfect decompression would save at most roughly 0.3–0.7 ms of the present
  fetch interval; actual decode/extra reads cost time. This cannot be the main
  multi-ms mechanism without a new measured representation opportunity.
- **CPU-first, fixed eight workers, simple admission margins, recency, blanket
  prediction:** their complete negative results are already recorded. New work
  must repair the underlying concurrency, policy or usefulness limitation.
- **Pinned allocation locality alone:** real-address replay in the existing
  97.82 GiB tier reaches approximately 52 GB/s. Neither arena size nor address
  scattering alone explains the 38 GB/s live path. Changing host cache policy
  without measuring CPU read cost can make this worse.

## DS4 / Strata comparison

DS4's `ds4_cuda.cu:35685–35741` keeps an explicit demand generation ticket,
resident mask, early-mask state, rank mapping and ownership of resident GU
results. `29706–29747` establishes separate component readiness, and
`35744–35777` consumes the ticket by enqueuing a consumer dependency rather
than waiting for a host wakeup. These are useful correctness patterns. Maya's
already device-resident route/plan and captured graph make the extra mapped-ID
copy and DS4 activation quantization unnecessary. Do not port them.

DS4's measured scheduling bundle worked on a different model/engine and does
not override Maya's negative DMA/resident trials. The important transferable
idea is complete ownership and component readiness, not its throughput ratio.
Strata's gain-filtered full-expert admission is likewise not a transfer fix:
the equivalent simple Maya frequency-margin replay already worsened demand.

## End-to-end ceiling and qualification

At unchanged 383 MB/token, isolated mapped capability would imply about
7.37 ms fetch rather than 10.09 ms. Even combining that optimistic bandwidth
with hiding all 2.04 ms of expert arithmetic leaves approximately

`9.223 + 7.373 - 2.037 + 2.956 + 0.042 + 1.156 = 18.713 ms/token`.

This illustration is approximately 53.4 loops/s. It is not a prediction:
isolated bandwidth may not coexist with the required CPU work, all expert math
cannot necessarily be hidden, and CPU exposure changes. At theoretical Gen5
x16 transfer time 6.078 ms the same optimistic calculation is about 17.418 ms,
or 57.4 loops/s. Therefore **transport plus same-layer overlap alone does not
establish 60 TPS with the current bytes and exposed CPU tail**. Byte reduction,
less CPU exposure or less pre-route arithmetic is also needed. The parent
campaign can combine costs, but their gains are dependent.

For any retained transport-only change, compare exact logits and exact routing/
tier assignment on the same forced inputs. Require unchanged packed payloads,
same GPU pool and bounded RAM/scratch headroom. Trace engagement separately
from unprofiled TPS. Preserve old launch paths for unsupported formats/devices;
exercise mixed Q2_K/Q3_K/Q4_K routes, scratch and promotion destinations, disk
answer classification, STOP/recovery/QUIT, graph errors and prefill pool lending
once the complete candidate earns further qualification. Do not relax numerical
thresholds or substitute enabled flags for demonstrated work.

## Concurrent parent implementation checkpoint

The parent campaign implemented and measured Stage A while this research was
being completed. Its saved campaign log reports warm overlap
0.594 -> 0.957 ms, arithmetic 9.223 -> 9.615 ms, CPU wait
2.956 -> 3.507 ms and wall 22.871 -> 23.195 ms. The resident pipeline engages
but is not a performance gain. The arithmetic increase approximately consumes
the newly hidden transfer, and the CPU tail also grows. This supports the
bounded ceiling and joint-completion acceptance criteria above.

I read-reviewed the parent's committed down-band implementation at `c982082`
without building or running it. First-half copy offsets correspond to the first
2048 rows; second-half copy offsets correspond to the remaining rows. The
second side copy may overlap first-band computation, and its event joins before
the second consumer. Both bands retain the original per-row dot and rank-order
sum; CPU-answer addition remains after both. Next routing remains behind the
final join. Reuse of `ev_pred` for second-band completion is safe in this serial
stream order: its earlier side wait precedes the second record on that same
side stream, and the main stream joins before the next layer reuses it. Existing
error teardown joins the side stream. No correctness defect was identified in
this bounded source review; that is not execution/parity qualification.

The old detailed analyzer requires repair for these mechanisms:
`tools/analyze_deep_profile.py:65` treats every 32-block copy as down;
`:75` takes only the first GU/down kernel; `:76` assumes at most two copies;
`:77` identifies prefix by a non-32 block count; and `:82–84` assigns an entire
component payload to each such kernel. A bounded 32-block prefix and two down
bands invalidate these assumptions, readiness gaps and byte totals. Identify
explicit prefix/band metadata or reliable per-route launch order, account for
both GU/down consumers and each disjoint byte range, then reconcile to engine
demand. The whole-token activity partition uses all compute/fetch intervals
and remains suitable for the aggregate overlap conclusion.

## Complete extension: capture the MLA MoE suffix

Follow-up source review on 2026-10-13 considered the parent's working suffix
implementation, after `c982082`, without executing the engine. Eleven MLA MoE
layers remain outside the recurrent full-layer graphs: 3, 7, 11, 15, 19, 23,
27, 31, 35, 39 and 43. Their dynamic-position attention prevents reuse of the
whole layer, but their exact MoE suffix has a stable boundary: **after the
FFN-side HC and its host residual-pointer swap, capture only `fast_moe`**.
The router/shared projections, actual route, disk publication wait, bounded
prefix/down transfers, unchanged expert arithmetic and CPU-answer addition
belong inside that suffix. Position-dependent attention and residual state
ownership remain outside it.

This is a complete extension of captured-only component overlap to the MLA
layers, without predicting routes or changing bytes. It removes host event
submission from the overlap fork/join and permits the same exact resident-GU
and output-row-band machinery there. It does not make attention available to
hide an as-yet-unknown actual route.

### Measured opportunity budget

A read-only SQLite query against
`docs/evidence/maya-decode-latest/latest/timeline.sqlite` examined 128 warm
token windows: the second group of 128 NVTX `token seq=... pos=...` windows
with position at least 30. For each of the eleven layers, the suffix began
at the first GPU kernel after the FFN-side `hc3_kernel` preceding its router,
and ended at its first subsequent `moe_cpu_wait_kernel`. The trace contains
88 kernels and 77 internal inter-kernel gaps per token in these suffixes;
none carries a graph execution ID.

| Warm aggregate per token | Time |
|---|---:|
| Eleven suffix spans | 4.595456 ms |
| Gaps between their recorded kernels | 0.167683 ms |
| Entry gaps from preceding HC end to first suffix kernel | 0.029687 ms |
| MLA routed GU plus shared down arithmetic | approximately 0.310 ms |
| MLA routed down arithmetic | approximately 0.200 ms |

The query sums gaps between kernels, not memcpy intervals: another transfer
could occupy part of a gap. Therefore 0.167683 ms is an **optimistic internal
submission budget**, not proved removable device inactivity. The external HC
entry gaps should not be automatically credited to suffix capture. Per-layer
internal gaps are only about 0.0137–0.0166 ms.

At unchanged arithmetic, newly enabling perfect MLA expert overlap contributes
at most about 0.510 ms; adding the optimistic internal submission budget gives
about **0.678 ms/token**. Applied illustratively to the saved 21.950269 ms
bands-combination repeat, this implies roughly 21.272 ms or 47.0 TPS. These
are upper bounds with dependent gains: transfer/compute contention, CPU-tail
exposure and extra launches can consume the opportunity. This extension is
credible but cannot alone supply a several-ms gain toward 60 TPS.

The parent's completed saved `suffix-bands/summary.json` reports repeat
45.818367 TPS and 21.825309 ms. Its manifest excludes resident GU, so the
matching configuration comparator is `bands-no-resident` at 45.707457 TPS
and 21.878268 ms: only 0.110910 TPS or 0.052959 ms difference. The comparison
with `bands-combination` at 45.557529 TPS and 21.950269 ms changes resident GU
as well, and cannot isolate the suffix effect. All 39,649,280 forced-input
logits are exact, with identical fetch/demotion
bytes, hits, CPU expert counts, promotions, pool slots and RAM slots. This is
consistent with the small measured gap budget. A single comparison does not
establish repeat reliability or prove how much of the change came from newly
overlapped MLA work; parent tracing/repetition owns that qualification.

### Graph key, scratch and ownership audit

The working implementation uses a separate `moe_suffix_graphs` container and
key `{cpu_plan, F->x, F->ffn}`. Under the current ordinary-decode contract that
is sufficient: the activation arena is allocated once in `fast_setup`, and
`xq`, `rlog`, `sh_g`, `sh_u`, `sh_hq`, `hq`, `sh_out`, route/plan tables and
scratch share its unchanged lifetime. Router/shared weight pointers and model
geometry are also stable. Environment-selected pipeline shapes are static
for the process. `cpu_plan` can change through `set_pcie_share`, and inclusion
in the key invalidates and recaptures each affected layer.

Expert slot addresses are live device data reached through `tab`, `plan_ptr`
and `fetch_src`; they are not frozen graph arguments. Thus promotion,
demotion and `vision_lend`/`vision_reclaim` reallocation of the expert pool
tail do not inherently require suffix graph invalidation. The existing
boundary/lending synchronization and table publication still apply. If future
code reallocates the activation arena, model weights or route buffers in place,
or changes the scheduling shape in process, add an arena/config generation
to the key or explicitly clear both graph containers first.

The suffix owns one real route. `LayerGraphs::enqueue` invokes the body once
at capture, checks exactly one host request increment, and adds one request
on each successful replay. Failed/abandoned capture restores the prior
expected count. No host residual swaps occur in `fast_moe`; unlike recurrent
whole-layer replay, suffix replay requires no swap compensation. `max_pf=0`
and `n_ahead=0` remove mutable prefetch host state from this boundary.

All prefix/down side-stream work joins before its consumer and before the
suffix completes. CPU-answer addition follows both output-row bands. The next
route cannot overwrite the shared plan, `hq`, `sh_out` or output while an
earlier consumer still uses them. Separate containers avoid nested capture
inside the recurrent whole-layer graph. Teardown synchronizes compute, copy
and side streams, then destroys both graph containers before freeing their
fixed buffers and pinned sources. No new ownership defect was found for the
ordinary suffix in this bounded review; exact runtime parity remains the
parent's evidence, not a consequence of this audit.

### Guard findings and subsequent source repairs

Two source guard leaks were identified in the reviewed working tree:

1. `fast_layers` graph eligibility does not explicitly exclude
   `F->win.pipe_t >= 0`. `win_pipe_part` sets `pipe_t=t`, sets the thread-local
   `gf::set_skip(W.skip_d)`, then invokes `fast_layers` with no incoming hop
   on the first/single device. Single-device window execution can therefore
   satisfy the ordinary graph guard. The skip pointer is baked into router,
   projection and expert kernel arguments, but absent from graph keys.
   Reusing a graph captured in ordinary mode can lose window skipping;
   reusing one first captured in window mode during ordinary decode can keep
   a stale non-null skip flag. Recurrent full-layer graphs additionally freeze
   window-specific KDA row pointers. Add an explicit `F->win.pipe_t < 0`
   eligibility guard before either graph container is used.
2. `multi_rows` calls `fast_moe_experts` directly, disables/restores only
   profiling, and leaves `kda_graph_on`/`moe_suffix_graph_on` from a prior
   `fast_layers` call. The new suffix flag can then enable captured-only
   component overlap for uncaptured MLA row work, reviving the host event
   submission cost this restriction was intended to avoid. Its existing
   per-row stream joins appear to retain scratch safety; this is a
   performance/scope leak, not a demonstrated arithmetic corruption.
   Use an explicit caller capture role or scope/reset these flags for direct
   callers. The asynchronous service probe's route classification should be
   kept independently valid rather than relying on a transient host scope.

The parent subsequently repaired both in source while this review was active.
`fast_layers` now explicitly requires `F->win.pipe_t < 0` for graphs. A
transient `F->moe_graph_body` flag wraps only the actual `fast_moe` body that
is eligible for capture; `fast_moe_experts` uses it for the captured-only
overlap gate, and the wrapper resets it on both true and false returns.
Graph replay skips host code and uses the already captured schedule. Direct
row/MTP callers therefore cannot inherit ordinary decode's capture role.
The service probe retains persistent graph eligibility rather than the
transient host scope, so asynchronous reporting remains possible. These
repairs address the identified source defects; no additional experiment was
performed by this research agent.

The usual MTP caller does not introduce the same suffix-flag inheritance:
`mtp_il_ >= 0` disables ordinary graphs in `fast_layers` before `fast_mtp`
calls `fast_moe`. MTP remains excluded from this campaign. Source anchors for
this audit are `src/core/glm_fast_path.cu` (`fast_setup`, `fast_moe`,
`fast_moe_experts`, `fast_layers`, graph teardown),
`src/core/glm_layer_graphs.hpp`, `src/core/graph.cpp`,
`src/core/glm_rows.cu:multi_rows`, `src/core/glm_window.cu:win_pipe_part`,
`src/core/glm_prefill.cu:vision_lend/vision_reclaim` and
`src/kernels/cuda/glm_fast.cu:set_skip`. Line numbers are deliberately omitted
here because the parent's source is still changing.
