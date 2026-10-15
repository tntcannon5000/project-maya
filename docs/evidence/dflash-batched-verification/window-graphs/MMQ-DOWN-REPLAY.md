# Captured MMQ down replay — 2026-10-15

## Finding

**The sampled expert61 down-output difference is caused by its MMQ dispatch/group
geometry, despite identical selected weights and quantized input.** This is now an
isolated production-kernel reproduction, not an inference from a flag or source alone.

| Check | Result |
|---|---|
| Selected down matrix bytes, first/repeated pass | Exact |
| Selected quantized activation bytes | Exact |
| First-pass production MMQ replay vs engine output | All 4,096 floats bitwise exact |
| Repeated-pass production MMQ replay vs engine output | All 4,096 floats bitwise exact |
| Zero unrelated weights/input rows, first-pass geometry | Selected output unchanged, bitwise |
| Zero unrelated weights/input rows, repeated-pass geometry | Selected output unchanged, bitwise |
| Between dispatch geometries | 3,349 float bit patterns differ; max abs 1.78813934326e-7 |
| Compute Sanitizer memcheck | 0 errors |
| Combined captures vs preceding diagnostic | Both byte-identical |

Together with the prior combine replay, this explains the sampled first-MoE
repeat-prefill difference as two grouping-dependent numerical effects. The specific
MMQ submechanism (tile selection versus Stream-K partial sums/fixup) is not isolated.
No requirement to distinguish those further before using this evidence for its scope:
batch geometry changes the result even with the selected expert data fixed.

## Method

`MAYA_PREFILL_DOWN_CAPTURE=1`, alongside the existing bounded MoE capture, captures
only the MMQ down batch containing the last token's first selected expert at layer3.
Store every batch down matrix, actual quantized input array, bounds, shape, and selected
output. Copy each matrix separately, so the diagnostic does not allocate another full
batch of host weights. Two raw captures are 634,879,564 and 19,685,740 bytes; kept local.
The same 6,793 occupied-token fixture runs two resets, GPU-only, graph off, with original
ownership policy for continuity. Manifest/log record settings; this is not a timing run.

The standalone `maya_mmq_down_replay` uses the actual `mmq::Context::run`. It repacks
down matrices contiguously, subtracts the batch's row base from bounds, and reconstructs
identity output indices. Exact self-replay establishes that normalization/repacking did
not change these sampled outputs. Quantized inputs retain column-block-major layout.
Identity indices include 512 padding entries for MMQ's tail-tile reads, matching the
production allocator's excess capacity. Source review caught that padding requirement;
it was fixed before the successful sanitizer run and checkpoint.

Then preserve geometry, selected expert weights and selected input row while zeroing
all unrelated matrices/rows. Output remains exact within each geometry. This removes
unrelated batch data as the explanation for the selected result. No host float emulation
or relaxed threshold is involved.

Build: `cmd /c tools\build-mmq-down-replay.bat`.
Replay: `build/maya_mmq_down_replay.exe <first.down> <repeated.down>`.
The captured pair passes `compute-sanitizer --tool memcheck --error-exitcode 9`.
Compact evidence, binary/capture hashes and sanitizer log are in `mmq-down-capture/`.

## Scope and decision

Retain the diagnostics and exact replay. Do not impose deterministic prefill kernels,
change arithmetic, relax thresholds or claim a TPS improvement from this finding.
Resetting a conversation need not recreate residency/group geometry, so controlled
graph comparisons must check starting-state equality rather than infer it from reset.

The old intermittent graph failure had no saved starting state. This result explains
the **current sampled prefill difference**, not that historical event. Controlled
current graph-off/on exact evidence and the historical limitation remain separately
recorded. Stop expanding this standalone diagnostic: return to campaign qualification
and the measured performance decisions. No additional throughput matrix is justified
by this result alone.
