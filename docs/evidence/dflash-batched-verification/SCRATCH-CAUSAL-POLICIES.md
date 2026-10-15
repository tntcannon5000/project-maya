# Causal scratch-retention policy gate

Previous goal turn made concrete progress: exact direct-RAM replay rejected a
slower mechanism and bounded cross-window scratch reuse. This follow-up tests
causal policies on that same trace, not another model workload or a TPS claim.

All policies start empty, keep CPU/promotion decisions fixed, charge maximum
blob-sized slots and protect active consumers in the shared64-slot pool. The
new policies can see the current fully routed layer, never future layers.
No parameter fitting or grid sweep. Frequency weighting uses accumulated past
GPU-fetch count times bytes; GreedyDual adds eviction-based aging. Past counts
include the current demand. The second half is chronological continuation of
history, not an independent workload or unbiased holdout after policy selection.

| Policy | T3 total saved bytes | T5 total saved bytes | T5 second half |
|---|---:|---:|---:|
| Legacy LRU replay |10.40%|3.21%|3.11%|
| LRU, protect all current-layer hits |10.55%|3.74%|3.62%|
| Frequency times byte cost |9.28%|8.78%|7.38%|
| GreedyDual frequency/byte cost |10.02%|5.99%|4.74%|

Legacy results reproduce the prior independent simulation exactly. Six focused
checks pass: future-prefix independence, promotion/D2D/retirement, protected
current-layer hits, pending-source exclusion, live capacity and initial-resident
duplicate retirement. Real trace completeness/unique-owner assertions pass.

Best causal T5 result saves126.15MB/window overall,108.94MB/window in the second
half, with additional promotion D2D copies still needed. At an illustrative
40GB/s transfer rate,126MB is about3.15ms of copy work, before overlap and policy
costs; this is not saved critical-path time. T3 does not improve over recency.
This does not justify a live allocator/cache implementation aimed at5ms/round.
Preserve tools and defer integration; do not select a default from this one
short history or expand to another policy sweep without a new mechanism.

The next qualification issue is to attribute the existing ordinary-versus-window
output differences under identical fixed input and controlled backend. Prior
exact candidate/control tests establish equivalence within verification, not
necessarily equivalence to ordinary forward with hybrid placement.
