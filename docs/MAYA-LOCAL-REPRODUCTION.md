# Maya reproduction and development baseline

## Scope

User-directed transition from DS4 to reproducing Project Maya, then developing on
Maya using lessons from DS4 and Strata. No engine optimization is applied before
measuring an unmodified baseline.

Source checkpoint: `20e27a23491eb08506d1c291def17a30dc89e92d` (v1.1.3).
Local branch: `research/maya-reproduction`.

## Reference

Issue https://github.com/mw00/project-maya/issues/43 and its attached report:
https://github.com/user-attachments/files/33244599/maya-report.txt

Reported v1.0.17, Linux, RTX 5090, Ryzen 7950X, 96 GB DDR5; Maya-M,
131072 context capacity, single GPU, no speculative decoding. Three short
greedy/thinking-off prompts, up to 256 output tokens: CSV parser 31.4 TPS,
photosynthesis 35.1 TPS, bouncing-ball HTML 31.9 TPS, mean 32.8 TPS.
Prefill: 707 TPS at 2048 tokens and 1982 TPS at 8192 tokens.

Local: native Windows, CUDA 13.2, RTX 5090, OC 9950X3D, 128 GB DDR5-6000 CL36.
Newer engine and different OS are explicit deviations. The current standard
benchmark has the same three decode questions, but its documentation-derived
prefill text changes with source versions. Context capacity does not represent
occupied decode context. The report also used an evolved expert usage profile
which is not supplied; cold local cache placement cannot reproduce that exactly.

## Execution plan

1. Build stock pinned Maya and download its published Maya-M shards to
   `G:/Models/Maya-M`, verifying publisher hashes. The pack indexes source GGUF
   data rather than duplicating the full model. Initial G: free space: 226.5 GB.
2. Use c=1, 131072 capacity, no vision or draft model, no expert pruning/skipping.
   Inspect generated config before inference; explicitly disable MTP.
3. Run the standard five-measurement benchmark and retain configuration, output,
   startup calibration, tier allocation and expert/cache/transfer statistics.
4. If materially behind the report, investigate version/platform/calibration and
   memory placement before modifying kernels. Compare calibrated hybrid versus
   GPU-only on a bounded shared fixture and check numerical differences separately.
5. Establish a reproducible launch and baseline checkpoint before architectural
   development. Preserve DS4 and Strata as references, not active inference engines.

## Status

2026-10-12: isolated checkout created; native prerequisite check passes. Stock
setup/build started. No local Maya inference or performance result yet.
