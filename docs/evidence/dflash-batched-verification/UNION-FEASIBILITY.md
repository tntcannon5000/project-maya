# Broader window-union placement: bounded feasibility assessment

2026-10-15. Source/evidence investigation only; no build or GPU execution.

**Decision:** do not implement a live union planner from the present evidence.
There is a larger theoretical opportunity than the narrow exchange screen,
but neither a measured large opportunity nor a universal rejection of union
placement. A small corrected-route diagnostic is the justified next step if
placement research continues. CPU-only repeated experts are particularly scarce
in the existing sample; treating all GPU repetitions as avoidable transfers
substantially overstates the opportunity.

## Evidence and limits

Primary counts below come from `assignment-screen/Q4_K_M/engine.log:28-44`,
with configuration/hash in its `manifest.json`, not from the similar earlier
`group-summary.json` run. The sample used one 40-token coding prompt, output
cap 64, full-layer T=3/5, pointer reuse, GPU22GiB/RAM94GiB and CPU plan
`011223344`. These group diagnostics synchronize every layer and perturb timing.

**These assignment-screen runs predate the deferred-demotion correctness fix.**
They describe those observed route/group counts, not established routing or
performance of the corrected current candidate. Their null pointers are CPU
proxies: CPU masks, initial RAM readiness and initial resident status were not
captured. Conditional deductions below assume complete evaluation, valid IDs,
RAM-backed misses and the source's enabled forward GPU-pointer reuse. Confirm
these assumptions against corrected routing before using a bound to reject a
current implementation. No conclusion here generalizes to other prompts,
widths, occupied contexts or cache states.

## What aggregate counts already imply

For each quant format, let R be routed entries, L logical expert-ID groups,
G GPU entries, H GPU ID groups, F distinct fetched ID groups. A group means a
logical expert within one layer/window, not an expert globally across windows.
Then C=R-G is the number of null/CPU-proxy rows, and U=L-H is the number of
CPU-only groups. Enabled forward pointer reuse means a mixed CPU/GPU group
has a null prefix followed by GPU uses (`moe_route_kernel`,
`src/kernels/cuda/glm_fast.cu:3264-3289`).

| T | Layer batches | R / L | G / H | C / U | F / fetched bytes |
|---|---:|---:|---:|---:|---:|
| 3 | 1,008 | 24,192 / 18,649 | 20,768 / 15,381 | 3,424 / 3,268 | 2,086 / 20,300,759,040 |
| 5 | 756 | 30,240 / 20,229 | 25,695 / 16,031 | 4,545 / 4,198 | 2,761 / 26,672,037,888 |

Copied bytes already equal unique fetched-ID bytes. Reassigning identical
currently fetched IDs to one physical owner cannot remove duplicate copies
that these counts do not contain.

### CPU-only repeated experts

Let M be mixed groups and O the mixed groups with exactly one null use. The
screen reports M=O=84 at T3; M=205, O=194 at T5. If Z is the number of null
rows in mixed groups, then Z=84 at T3 and
`O + 2*(M-O) <= Z <= O + (T-1)*(M-O)`, or 216..238 at T5.
CPU-only repetition excess is `E = C-Z-U`, the number of CPU-only rows beyond
one per CPU-only group. Each repeated group consumes at least one excess;
each has at most T-1 excess rows.

| T | CPU-only repetition excess E | Possible repeated CPU-only groups | Upper bound on their rows |
|---|---:|---:|---:|
| 3 | 72 | 36..72 | 144 |
| 5 | 109..131 | 28..131 | 262 |

The row bound is `2*E_max`, obtained when every repeated group has two rows;
higher multiplicity lowers the number of groups and total repeated rows at
fixed E. Applying the bounds per format strengthens the T5 lower group bound
to 28. At most 0.60%/0.87% of all routed rows belong to these previously
CPU-only repeated groups. Promoting all such groups to GPU is therefore not a
large newly exposed batching workload in this sample, and adds one packed
transfer per newly GPU-assigned expert.

### Resident repetition versus nonresident traffic

H-F = 13,295/13,270 GPU groups at T3/T5 did not require a new copy in their
window: 86.4%/82.8% of GPU groups. Under the stated assumptions these were
resident groups. A fetched destination may itself become permanent during the
window; classify it as initially nonresident rather than counting its later
residency as a free input.

There are 4,308/5,766 repeated GPU groups. For each format at most F of its
repeated groups can have fetched; summing `max(0, repeated_GPU_groups-F)`
gives at least 2,648/3,396 resident repeated groups, or 61.5%/58.9% of all
repeated GPU groups. These are lower bounds, not measured residency splits.
The logs do not identify the multiplicity of fetched groups, so they cannot
establish how much repeated nonresident GPU work could instead share CPU reads.
GPU pointer/ID group counts agree: there is no observed lost grouping caused
by duplicated GPU destinations in this sample.

## A broader optimistic byte bound

Consider RAM-only placement with initial residents fixed on GPU, the same
selected experts and **no increase in total CPU rows**, applied per format.
Every initially nonresident logical group must either be CPU-only or cross
to GPU once. If its baseline CPU-only count is U, candidate CPU-only groups
cannot exceed its allowed CPU rows C. Thus at most `C-U` baseline transfers
can disappear. Allowing row budgets to move freely across layers makes this
optimistic; real per-layer constraints, promotions and timing reduce eligibility.

| T | Maximum removable blobs Q2 / Q3 / Q4 | Weighted bytes | Fraction of baseline fetched bytes |
|---|---:|---:|---:|
| 3 | 131 / 15 / 10 | 1,385,496,576 | 6.825% |
| 5 | 290 / 32 / 25 | 3,094,609,920 | 11.602% |

Blob sizes inferred exactly from bytes/F are 8,257,536 / 10,813,440 /
14,155,776 bytes. Formula: `sum_format((R-G-L+H)*blob_bytes)`.
This allows replacing every CPU-only repetition and mixed prefix optimally
with GPU uses while moving distinct GPU singletons onto CPU, without the
narrow policy's two-swaps/layer restriction. It is an upper bound for this
fixed-row, fixed-resident class under the stated proxy assumptions, not a
realizable assignment or a universal bound on broader policies.

For scale only, applying these byte fractions to historical perturbed fetch
intervals 21.57/37.98ms would suggest about 1.47/4.41ms of proportional copy
time, before planner overhead. This is **not a time bound or predicted gain**:
bandwidth, launches, overlap, CPU starts/tails and GPU grouping change. Larger
CPU budgets or moving residents to CPU invalidate the bound; residents need
independently verified ready shadow backing. Moving repeated fetched experts
to CPU can also change the CPU row budget and needs its own cost model.

The corrected moderate-CPU experiment increased allocation to `011233445`
with two window lanes and merged rows. It engaged 16,092 CPU rows/15,141 unique
reads, but verification was 101.295ms versus the prior 99.192ms candidate,
with changed output history/acceptance. This fails the speed gate, not every
union policy. Component streaming's contemporary 58.841->58.081ms result
similarly shows why demonstrated transfer overlap is not equivalent to
critical-path savings (`STATUS.md`, `ROUND-UPDATE.md`, `WORK-LOG.md`).

## Diagnostic that would justify implementation

One bounded corrected-route T3/T5 diagnostic should record, per layer/format:
initial resident and ready RAM sources; exact CPU masks and selected IDs;
CPU-only, mixed and GPU-only multiplicities; fetched-expert multiplicities;
copy-owner/promotion dispositions; actual bytes; CPU service start/end and
remaining tail; GPU transfer and complete expert times. Export metadata after
the run or in explicit diagnostic mode, not a host readback in a timed planner.
Prove initial RAM ownership after deferred demotion submission. Reconcile the
new totals with live routes before trusting offline candidate evaluations.

Evaluate deterministic single-expert moves and bounded swaps with measured
format/multiplicity CPU and GPU costs, shared RAM-bandwidth contention, actual
service start and transfer overlap. Decode calibration and merged scheduler
alpha=0.2/0.3/0.8 are not calibrated window makespan curves
(`src/core/glm_fast_path.cu:2113-2114`). Require a conservative whole-round
opportunity above the prior 5% gate after overhead, with enough eligible layers
to matter. If that fails, record a measured feasibility rejection and stop;
do not sweep quotas. The current aggregates cannot supply that decision for
policies that change the CPU row budget.

## Smallest complete planner if the diagnostic passes

The existing full-layer schedule has all router logits before routes, so no
attention/recurrent reorder is required. Restrict initial implementation to
full-layer T3..8, complete top-k, ready RAM/resident inputs, expanded scratch,
no skip/prefetch/lookahead or staggered schedule. Any unsupported layer falls
back before live mutation. Required supporting work is concrete:

1. Share the exact selector and route-disposition logic; construct a bounded
   64-entry ID union without touching live counts or publishing requests.
   Validate baseline shadow masks/owners against actual routing, including
   count aging, CPU ties, spare claims and promotions.
2. Choose authoritative per-expert CPU/GPU placement from a measured makespan
   model, keeping residents GPU and baseline when predicted margin is small.
   Initially bound new GPU copies to two/layer plus explicit byte/scratch caps.
   A nominal per-token CPU table cannot always satisfy consistent expert
   placement; do not silently split experts to claim quota preservation.
3. Materialize ordinary token requests once in order, preserving exact route
   weights/counts/sequence/retirement. Disable legacy CPU choice and forward
   reuse overrides for forced assignments. Give one first-use GPU owner each
   source/copy and point all members to its pinned scratch or resident slot.
4. Reuse complete native paired GPU kernels and `win_cpu_merged`, which already
   groups by RAM pointer and retains per-token answer/sum order
   (`glm_fast_path.cu:2074-2180`). Serial route/wait/fetch is sufficient for
   the first placement experiment; union-fetch launches are a separate change.
   Preserve layer barriers: merged service holds ring pointers until last
   route (`glm_fast_path.cu:3686-3700`), and the 64-slot ring must not be lapped.
5. Preserve stable RAM/source and scratch lifetimes through CPU work, GPU
   compute, processed/expected retirement, cancellation and RAM release.
   Moving disk-only routes ahead requires a different response/staging
   protocol and is outside this RAM-ready first candidate.

Qualification must separate same-placement scheduling parity (exact logits)
from backend-placement numerical changes (full logits and greedy agreement
with explicitly reported differences). Exercise partial commits, reset/repeat,
pressure and cancellation before recommending a retained combination.

This is a finite layer-planning experiment, not a general router rewrite.
Its missing prerequisite is measured critical-path opportunity; neither the
narrow 2.1%/3.5% result nor resident GPU repetition supplies that evidence.
