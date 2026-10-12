# Research and first screens

Sources retrieved 2026-10-12, copies in local `public-research/`.

- CUDA Best Practices Guide: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
  Pinned buffers, non-default streams and independent work enable overlap;
  Async alone is not a proof. Concurrent memory demand can still bottleneck.
- Fiddler: https://arxiv.org/abs/2402.07033 and https://github.com/efeslab/fiddler
  CPU/GPU orchestration avoids expensive expert movement. Reported speedups
  are on other setups, not predictions for this machine. Maya already has a
  calibrated CPU lane; investigate its marginal costs rather than duplicate it.
- ProMoE: https://arxiv.org/abs/2410.22134
  Intermediate results predict later experts; proactive caching is relevant
  to Maya's existing next-router path. Useful/timely bytes matter, not hit
  overlap alone. Published speedups do not validate our predictor.
- NVIDIA nvCOMP: https://docs.nvidia.com/cuda/nvcomp/index.html
  The old GitHub repository is archived and points to CUDALibrarySamples.
  GPU codecs are a possible implementation, not evidence quantized weights
  compress enough to justify one. A copy-saving representation must be lossless.

Local DS4: `cuda_stream_upload_batch` and `ds4_gpu_pinned_resident_gate`
in `DS4-glm/ds4_cuda.cu`. Measured retained bundle: 23.79→25.44 TPS on a
different quant, exact fixed-input logits, small prefill change. Graphs and
independent demand queues did not establish gains; do not copy indiscriminately.
Strata comparisons and retained/rejected paths are recorded in the preceding
MAYA-DS4-PORT-PLAN and local Strata/DS4 evidence, not inferred from marketing.

## Compression feasibility (H7)

Actual model payloads, all three expert quant classes (GGML 10/11/12), three
distributed layers per class, experts 0/137/287, separate gate/up/down matrices.
Every sampled decompression compared byte-for-byte with the original.

| Codec | Compressed/original | CPU decode GB/s |
|---|---:|---:|
| DEFLATE level 1 | .9390 | .505 |
| DEFLATE level 6 | .9342 | .535 |
| LZ4 default | .9712 | 3.516 |
| Zstd level 1 | .9332 | 1.829 |

These are single-thread CPU feasibility timings, not GPU codec throughput.
LZ4's optimistic savings against 14.32 ms is only 0.41 ms before decompression;
Zstd's is 0.96 ms. CPU decompression of ~500 MB would cost ~142 ms / ~273 ms.
Reject CPU-decoded compressed transport. Deprioritize a GPU codec for this
campaign's substantial-gain goal: even perfect decompression cannot save more
than the measured small byte fraction for these codecs. This does not rule
out a novel structured representation or another model's compressibility.

## Controlled screening fixture

CSV request's 30 prompt IDs + 128 saved output IDs, two repetitions in one
process. 131072 capacity; short occupied positions 30–157. Fixed 22 GiB expert
pool cap, 87.4 GiB RAM cap, CPU plan 011223344, same initial expert profile.
No inference sampling; all logits saved, per-token times exclude file writes.
Host logits download is included; absolute numbers differ from server sampling.
Baseline: 2693 GPU / 10555 RAM slots, ~87.33 GiB pinned expert RAM. Baseline
driver binary preserved as `build/maya_transfer_lab_baseline.exe`.

| Screen | First / repeated TPS | Status |
|---|---|---|
| Baseline | 30.52 / 33.03 | Reference |
| H3 table publication before bulk D2H submission | 30.55 / 32.98 | No throughput improvement; all 39,649,280 logits bit-identical. Needs trace before claims about overlap |
| H1 bounded 4 GiB RAM shadows, no new allocation | 31.72 / 34.43 | Provisional ~4% gain. At boundary 256, 1,895 avoided copies / 18.04 GB cumulative including prompt. No disk reads; all experts represented |

H1 changes placement/CPU assignment: max logit difference 5.5712, RMS .15365,
top-1 agreement 98.44%. **Not a parity pass.** Need CPU-off isolation, same
route/placement controls and ownership checks before retention. No threshold
has been relaxed. Counters are cumulative; do not divide the 18.04 GB value
by 256 and call it a decode-only statistic.
