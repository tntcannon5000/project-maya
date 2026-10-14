# Verification component streaming

## Hypothesis

The verification path waits for full expert blobs before starting grouped
gate/up work. Native blobs contain gate, up and down components. Gate/up needs
only the prefix; down can arrive while gate/up computes. This can remove an
exposed transfer tail without changing expert choice or arithmetic.

This differs from the earlier full-blob asynchronous experiment, which mainly
overlapped copies with routing. It also avoids splitting resident/nonresident
arithmetic into extra launches, an earlier source of overhead.

## Complete dependency chain

1. Publish every row's route and resolve disk requests into stable per-token
   staging. RAM sources remain owned until the window boundary.
2. Copy all nonzero sources' gate/up prefixes into their original destinations.
3. After prefix completion, start down-component copies on the side stream;
   cap their worker blocks to preserve compute resources.
4. Run existing grouping and gate/up arithmetic on the compute stream.
5. Join down-copy completion before existing down arithmetic and combine.
6. Finish both components before any later layer reuses plans or scratch.

Keep full-layer native formats, eight routed experts, disjoint row scratch,
aligned component sizes and stable disk staging as eligibility requirements.
Ineligible paths retain existing behavior. Preserve the combined arithmetic API
for callers that do not need separate readiness phases. Error paths drain live
side-stream work. Immutable-byte diagnostics intentionally join early.

## Decision evidence

Existing GPU-only trace GU intervals of 6.18/8.85 ms at T5/T8 bound the possible
overlap; they do not predict a hybrid gain. Current paired GU may be shorter,
CPU completion may dominate, and transfer workers can slow computation.

Require exact fixed-input logits, including mixed formats and partial commits.
Use diagnostic source counts for actual component bytes (submitted row capacity
is not traffic), and Nsight interval intersections for actual fetch/GU overlap.
Then use one short coding screen with the same allocations and flags apart
from component scheduling. Reject if round time does not improve materially.
Do not infer a speedup from stream existence, enabled flags or overlap alone.
