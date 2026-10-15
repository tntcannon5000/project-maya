# First-MoE capture and production replay — 2026-10-15

## Result

The sampled repeat-prefill difference has **two components**, not just combine order.
Capturing layer 3's last prefill row and replaying its actual production CUDA combine
reproduces both captured outputs bit-for-bit. Swapping only the combine partitions
changes the result. Five of eight captured expert output vectors also differ before
combination. Their matrix-product origin is not yet isolated further.

This is causal evidence for the sampled combine effect. It does **not** retrospectively
attribute the historical graph-on failure, whose pre-verification state was not saved.

## Controlled experiment

- Same 6,793-token occupied prefix; captured last row of the 6,792-token batch before
  the final single-token forward. Same fixed continuation and two resets.
- GPU-only, pool16/RAM94, graph off, original ownership policy deliberately retained
  for continuity with the prior numerical diagnostic. This is not the recommended policy.
- Native Q2/Q3/Q4 candidate flags as recorded in `moe-combine-capture/graph-off-manifest.json`.
- Only layer3 selected; no `GLM_CB_DIR` global seam instrumentation.
- Host capture synchronizes this layer. Timings are **not performance evidence**.
- Raw `.bin` and `.f32` remain local; compact route summaries, replay results and
  fixture manifest/log are checkpointed. Raw capture hashes are in `routes.json`.

| Measurement | Result |
|---|---:|
| Route IDs, route weights, shared/initial FFN | Exact across resets |
| First-pass / repeated-pass combine groups | 3 / 4 |
| Expert outputs differing | 5 of 8 |
| Maximum individual expert output difference | 1.78813934326e-7 |
| Captured final FFN maximum difference | 1.49011611938e-7 |
| Production replay vs captured output, each pass | Bitwise exact |
| Same first-pass values, changed partitions | 5.96046447754e-8 max; 2,135 differing float bit patterns |
| First-pass values with repeated-pass partitions vs repeated-pass output | 1.34110450745e-7 max |

Route-order groups (indices into the eight router selections):

- First pass: `[3,5]`, `[0,4,7]`, `[1,2,6]`.
- Repeated pass: `[3]`, `[1]`, `[0,4,5,7]`, `[2,6]`.

Floating-point partial sums are added to the shared result once per group. This
changes association even though every selected expert remains included. Replaying
both sets of partitions on common values isolates this effect from the separately
observed expert-output differences. Neither error component is generally additive.

## Implementation and validation

`MAYA_PREFILL_MOE_CAPTURE=<path-prefix>` records integer route IDs/positions, float
weights, initial FFN, per-flush selected expert outputs, and final FFN. Capture is
before window reuse and after shared-expert computation. GPU-only prefill is required.
No capture allocation or CUDA synchronization occurs when disabled.

`maya_moe_combine_replay` reads captures, compacts sorted positions while preserving
route iteration and flush membership, and invokes the production `moe_combine_add`.
It rejects invalid shapes/routes, truncation, and more than eight routes (the kernel's
actual shared-array limit). Captures retain the chronological partitions.

- Build: `cmd /c tools\build-moe-combine-replay.bat` passed.
- `tools/test_moe_combine_replay.py`: 3 tests passed, including a production-kernel
  cancellation example proving partition sensitivity, K9 rejection, and truncation.
- Actual-model replay: both outputs exact, including all 4,096 floats per pass.
- Read-only source review confirmed capture lifetime and replay mapping. Its K>8
  parser finding was fixed before checkpointing.

## Decision and next discriminator

Retain diagnostic tools; do not change default arithmetic to force bitwise cold/warm
identity. Do not relax parity gates. Existing controlled graph-off/on exact evidence
remains valid, but reset alone is not a guarantee of identical numerical starting state.

If further attribution is necessary, capture the selected MMQ gate/up activation and
down input plus dispatch geometry on this same row. That distinguishes product/reduction
geometry from combine association without another throughput matrix. The original
historical failure remains a separate limitation until reproduced with prefix evidence.
Current performance evidence is unchanged: 56.78 short coding, 39.64 longer coding,
40.96 writing TPS in the previously reported bounded confidence-policy samples.
