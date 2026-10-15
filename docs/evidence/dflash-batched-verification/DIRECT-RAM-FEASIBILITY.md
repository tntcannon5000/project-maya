# Direct mapped-RAM native verification experts

2026-10-15. Source and saved-data audit only; no implementation, build or GPU
execution. **Recommend one bounded complete feasibility experiment.** The
mechanism is not established as fast, but its eligible work is substantially
larger than the rejected union-assignment mechanism and no source contract
requires an intermediate VRAM copy for transient native experts.

## What is new and how much can engage

Prior transfer campaigns tested mapped pulls versus DMA, component/resident
overlap, CPU scheduling and backing. `glm_transport_probe.cu` modes `pull_gpu`
and `pull_cpu_gpu` still copy weights into VRAM `target` before native GU/down.
Neither those modes nor the recorded H4/H5 experiments directly execute the
native expert dots on RAM pointers. No prior direct-weight arithmetic result
was found; rejection of integrated DMA does not reject this mechanism.

The corrected union CSV permits exact classification by copy-owner destination:

| Observed native transient work | T3 | T5 |
|---|---:|---:|
| Transient packed bytes | 6,903,693,312 | 13,124,567,040 |
| Fraction of all fetched bytes | 34.76% | 53.70% |
| Initially ready-RAM transient bytes | 6,828,982,272 | 12,934,840,320 |
| GPU multiplicity 1 / 2 / greater than 2 groups | 608 / 16 / 0 | 1,179 / 73 / 6 |
| Bytes in GPU-singleton transient groups | 6,730,285,056 | 12,318,474,240 |

These are bytes currently staged in scratch, **not removable PCIe traffic**.
Direct arithmetic still loads packed weights across PCIe and may issue more
physical transactions than a copy. It can remove scratch writes, subsequent
VRAM reads, and the copy-before-compute dependency for these experts. Do not
multiply these byte fractions by fetch time and present that as a speedup.

## Pointer and lifetime contracts

The existing copy kernel casts `fetch_src` to `const uint4*` and dereferences
it on the GPU (`glm_fast.cu:3533-3567`). The RAM allocations are pinned/UVA;
the NUMA registration path explicitly requires host-pointer support and uses
`cudaHostRegisterMapped` (`glm_fast_path.cu:291-346`). Thus the actual addresses
already consumed by GPU fetch are the relevant device-readable addresses;
do not substitute an arbitrary pageable host pointer or unverified alias.

Transient RAM-backed experts remain owned through the window. The service
touches their recency but does not free/reuse held slots within layer routing.
RAM victim selection and release occur at `fast_boundary`, after compute
stream synchronization and `processed >= expected`, under `mu`
(`glm_fast_path.cu:4360-4372`, `4480-4515`). A promoted source becomes
kRRelease, but even that release happens at the boundary. Leave permanent
promotions staged in VRAM for the first experiment; never put a RAM pointer
into the permanent GPU table or claim a spare was filled when it was not.

The first bounded candidate should use sources already ready/published in RAM.
Do not infer readiness from a nonzero CPU-side slot or pending demotion. A
later expansion could use post-`moe_wait` resolved sources, but disk staging
must be token-owned: the old shared eight-slot staging can be overwritten by
later routes before delayed direct arithmetic. Exclude disk/pending sources
initially rather than making readiness/staging expansion a dependency.

Direct reads must finish before boundary slot release, cancellation cleanup,
resize or teardown frees pinned allocations. Keep compute on the existing
stream and retain all service retirement. Make the compute drain explicit on
the direct path; `win_free` currently explicitly drains compute only when a
fetch stream exists, so do not rely on incidental `cudaFree` synchronization.

## Native kernels: feasible arithmetic, uncertain transport efficiency

The native GU/down kernels take their packed `blob` from the group pointer.
Changing that pointer's memory domain does not change activation quantization,
integer dots, float accumulation, warp reductions or combine rank order.
`moe_win_gu_reuse` and `moe_win_down_reuse` share each decoded weight call
across members (`glm_fast.cu:5174-5264`). The original singleton kernels also
accept an ordinary global pointer. This makes complete GU, h quantization,
down and combine available without implementing a different arithmetic path.

However, native K-quant loads are not the copy kernel's contiguous uint4
stream. Lanes walk K with stride32; Q2/Q3 read quant words plus byte scales,
Q4 reads quant words, packed scales and half factors. Lanes share/duplicate
metadata within quant blocks (`DecW<10/11/12>`, `rows_dot`). Coalescing,
system-memory transaction sizes and latency hiding must be measured. Do not
assume GPU caches will eliminate repeated host loads. Host-dependent CTAs can
occupy SM resources and delay otherwise ready resident work.

Mode2 reloads a group's weights for each pair of GPU members. The trace's
greater-than-two transient groups would add only 61,341,696 nominal packed
bytes at T5 (0.47% of transient bytes), and none at T3. This is not a physical
traffic estimate, but it rules out a large multiplicity multiplier as the
main reason to reject this particular sample. CPU-only groups remain CPU;
mixed groups preserve their existing CPU prefix and GPU members.

## Minimum complete remap and decisive test

1. Gate only CUDA native Q2/Q3/Q4, full-layer windows, expanded row scratch,
   complete evaluation, existing placement and reuse. Initially accept only
   ready-RAM sources destined for transient scratch, with no promotion. Keep
   ordinary decode, prefill and unsupported paths unchanged.
2. Change a transient owner's plan pointer **before its copy and before a
   later route searches earlier pointers**. Keep request CPU/fetch/promotion
   masks, route counts and service bookkeeping unchanged; suppress only that
   owner's actual GPU copy source and publish its mapped RAM plan pointer.
   Later occurrences then reuse that same pointer through the existing search,
   including its override of a nominal later CPU choice. Do not change backend
   placement. If remapping after all routes instead, replace every reference
   to the old scratch pointer; changing the owner alone breaks reuse. In the
   serial path this would also be too late to avoid already submitted copies.
3. Reuse existing grouped complete arithmetic, with singleton/pair handling.
   Unchanged component prefix/down fetches naturally skip cleared sources;
   remaining permanent copies retain their events and joins. There is no
   resident-only split or duplicated expert arithmetic.
4. Add explicit direct-group/row/nominal-byte counters and copied-byte counters.
   Existing demand/fetch masks describe logical routing demand, not physical
   copies after this change. Existing union analysis assumes every nonresident
   GPU group has a copy owner, and weight-audit guards admit only GPU slots;
   either extend these diagnostics to direct owned RAM or reject incompatible
   diagnostic combinations explicitly. Never report direct bytes as zero
   traffic or direct pointers as initial VRAM residents.
5. First compare complete staged-copy+GU+quant+down+combine against direct
   GU+quant+down+combine for actual Q2/Q3/Q4 model geometry and representative
   singleton/pair, mixed resident/transient groups. Use changing immutable
   blobs larger than cache, identical activations and exact outputs. Include
   the existing CPU co-execution when assessing timing; copy-only bandwidth
   was a misleading predictor in the prior campaign.
6. If the complete stage has material headroom, require same-placement full
   logits, partial commits/reset/repeat, then one matched coding T3/T5 screen
   against the best current staged combination. Check cancellation and bounded
   pressure only if its end-to-end result passes the existing 5% round-time
   gate. A neutral complete-stage screen ends the experiment without a format,
   block-count or CPU-quota sweep.

Unlike assignment, this candidate does not need to find a large new repeated
expert set or alter the CPU quota. Its unresolved question is whether native
packed arithmetic can sustain useful mapped-RAM bandwidth while doing the
required math. A source-only answer cannot decide that question.
