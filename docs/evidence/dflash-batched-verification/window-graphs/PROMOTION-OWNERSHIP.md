# Competing promotion retirement

Status: opt-in `MAYA_PROMOTION_OWNER_FIX=1`, validation underway. This addresses measured duplicate GPU residents; it is not yet an explanation for the earlier intermittent logit mismatch.

## Observed problem

A background H2D upload reserves Landing slot A and pins RAM source R0. While R0 remains routeable, a demand (or predicted expert) request can upload the same expert into spare B. Background retirement previously published A unconditionally, leaving B resident but untracked. The bounded GPU-only fixture accumulated 34 duplicate/orphan residents.

## Implementation

Under the service metadata lock, both routed and predicted promotion writers mark pending uploads of the same expert to other slots as superseded. The marker is latched even if B later leaves VRAM. A and R0 stay reserved until the copy completion event succeeds. Retirement frees A without changing the demand owner's GPU mapping. It retains R0 as held backing when still current; otherwise it publishes the current ready RAM owner, or zero for an unavailable/pending replacement, before freeing R0. A new pending demotion can change `ram_of` before changing the device RTAB, so merely leaving RTAB untouched would be unsafe.

Retirement validates Landing and pinned-source ownership. An unexpected owner or unmarked competitor latches the existing fatal copy error, preventing a retry through partially retired metadata. No route, expert, quantization or arithmetic is skipped. Existing lending drains background work before reusing pool storage.

## Validation gates

1. Same fixed-input T3/T5 fixture, 6,793 occupied prefix, partial commits, two resets. Check graph-off/on exact logits with the ownership fix enabled in both.
2. Count superseded retirements, duplicate/orphan residents and invalid pinned sources; prove the path engages and duplicates cease accumulating. Compare GPU-only logits to the earlier retained control as well.
3. Test matched hybrid behavior and serving speed only after the correctness gate; keep any unreproduced original failure explicit rather than declaring it fixed.

The diagnostic is off by default. CPU metadata scanning costs and any cache/traffic changes need measurement before a default recommendation. Pending superseded uploads still consume the existing transfer budget until completed; this patch eliminates duplicate residency, not already-issued PCIe bytes.

## Initial full-model result

Both graph-off and graph-on retire 51 superseded uploads. Across all recorded boundaries, duplicate residents, orphan residents and invalid pinned sources remain zero. The corresponding pre-fix observation accumulated 34 duplicate/orphan residents. All 7,434,240 logits / 48 rows match exactly between current graph-off/on, with 84 captures and 420 actual replays.

Against the older ownership policy, the first 24 rows (first request) are exact; the second request differs after reset and batched prefill (max abs 2.32879). Cache layout has changed; this is not yet attributed. The next fixture records prefix logits separately. No claim that the original intermittent graph failure has been causally fixed.

Warm GPU-only window call means: T3 91.621 -> 87.724 ms; T5 148.532 -> 144.738 ms. These are diagnostic fixed-input call timings, not generated-token TPS, and do not include the hybrid backend. They are consistent with a modest dispatch saving, not a transformative speedup.
