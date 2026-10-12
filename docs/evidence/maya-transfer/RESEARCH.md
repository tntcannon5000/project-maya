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

## Subsequent screens (still provisional)

| Mechanism | First / repeated TPS | Repeated demand / D2H MB per token | Finding |
|---|---|---|---|
| Resident gate/up overlap | 29.74 / 31.52 | 403.5 / 339.8 | Exact logits; slower; trace overlap before rejecting mechanism |
| Admission count >=4 | 32.73 / 34.92 | 410.8 / 249.1 | Promising reduced churn |
| Existing prediction, max 2 | 27.67 / 28.98 | 319.2 / 462.3 | Demand drops but migration grows; total prediction-byte counters added for qualification |
| More CPU: 012233445 | 31.92 / 35.38 | 254.3 / 229.5 | Promising lower transfer demand; arithmetic placement differs |
| Admission + CPU + 4 GiB shadow | 33.15 / 36.16 | 258.7 / 166.2 | Combination works, not additive |
| Same combination without shadow | 32.98 / 36.12 | 257.7 / 185.5 | Small shadow has negligible additional speed benefit here |

GPU-only ownership isolation: baseline vs 4 GiB shadow, all **19,824,640
logits bit-identical**, 128 identical input forwards. This excludes mixed
CPU/GPU assignment as a confounder for that comparison. Stock hybrid between
its own repeats has max difference 3.877, RMS .1382 and 98.44% top-1 agreement.
Do not conflate that observation with a license to ignore arbitrary errors.

H5 isolated transfer test, byte-exact readback for each mixed-format payload:
DMA ~57–58 GB/s; mapped pull ~45–53 GB/s. This is a feasibility microbenchmark,
not end-to-end performance. Two first integrated completion schemes stalled
before producing scored tokens (host callback, then event polling). Both owned
processes were stopped. Submission ordering is being investigated; these are
incomplete implementation results, **not** performance rejections of DMA.

Strata's own prior early-shared overlap retest also failed to improve throughput
despite token agreement (GLM-CONTINUOUS-RND round 20). Its host timings cannot
be substituted for our CUDA timings. This reinforces the need to verify
independent work and actual critical-path reduction in Maya.

## Dependencies tested and stronger mechanisms

- Expanded H1 backing (101 GiB requested, 97.82 GiB actually pinned): 37.43 /
  41.24 TPS. Windows pinning fallback reduces Q2_K class from 10384 to 10059
  slots. Measured minimum physical availability 12.17 GiB, commit availability
  22.87 GiB. This is a bounded shadow tier, not a claim every model byte has
  a RAM duplicate. Repeated demotion traffic is below 1 MB/token.
- More CPU with expanded H1: 36.70 / 41.05; adding admission: 34.56 / 41.00.
  Their original gains are **not additive** once demotion is nearly gone.
- DMA + resident compute: 29.93 / 32.26. Expanded H1 + DMA + resident:
  36.74 / 40.09. Neither beats its simpler counterpart. Real DMA engagement
  is counted in bytes; explicit route submission fixes the earlier stalls.
- Original overlap trace: 1.037 ms/token fetch/arithmetic intersection,
  versus zero control. Arithmetic grows 9.154→10.257 ms, GPU-inactive gaps
  1.942→4.169 ms, wall 33.591→34.680 ms. Real overlap was insufficient.
- H3 refinement: table publication alone was incomplete scheduling because
  the next embedding upload still followed bulk D2H. `layers` scheduling
  queues copy-back after embedding submission; protected slots/events remain
  owned until completion. 34.34 / 37.00 TPS, all 39,649,280 logits exact.
  Its pending queue also has a fallback before another boundary and explicit
  drain before prefill lending. Trace and combination qualification pending.

Long-context harness correction: the first `baseline16k` attempt mistakenly
used `forward(vector)` for the prompt, which serially decoded it. It was
stopped and has no valid scored results. Corrected driver calls `prefill`
for the prefix and `forward` only for its last token, mirroring the server.
A separate baseline worktree at 615973d adds only the test driver/CMake target
(commit 4bf9d4c); engine sources unchanged. Its extra transfer CSV counters
are unavailable (zero placeholders), not zero-byte measurements. The 8K
fixture now contains frozen baseline Python source rather than repeated text.

## Final combination and qualification

These results supersede provisional/incomplete statuses earlier in this log.

- H5 submission handshake completed the DMA path: 30.41 TPS first request,
  exact fixed-input logits. DMA+resident compute: 29.93/32.26 TPS; adding
  expanded backing: 36.74/40.09, slower than backing alone. Rejected code is
  preserved on `research/maya-transfer-prototypes`, not discarded as untested.
- H6 with expanded backing: 34.51/36.53 TPS. Repeated demand 228.0 MB/token
  plus predicted 591.6 MB/token, only 375.9 MB useful (63.5%). Removing D2H
  does not rescue this prediction policy; wasted and early transfers remain.
- H3 trace: 31.23 ms wall, 2.33 ms D2H/arithmetic overlap. D2H duration itself
  grows to 12.31 ms under contention; the *wall* time improves. Do not present
  raw copy duration as exposed critical-path cost or add overlapping stages.
- H1+H3: 38.90/42.61 TPS short before cleanup; 35.52/37.27 at 8K occupied
  context. Batched prefill 3488.56/3479.68 ms vs stock 3563.05/3562.62.
  Relative to H1 alone (37.43/41.24), the scheduling contribution is modest
  but positive; relative to H3 alone (34.34/37.00), backing remains important.
- Combined trace first/repeat: 26.94/25.09 ms wall; D2H 50.02/0.711 MB/token.
  Repeat mapped pull 11.05 ms; GPU arithmetic 9.14 ms. Warm demotion is no
  longer the main bottleneck. No 60 TPS claim is supported.
- Cold GPU-only expanded backing with complete per-class sampled payload
  comparisons matched all 19,824,640 logits to warm GPU baseline exactly.
  Hybrid placement-sensitive differences are not a bit-exact parity pass.
- Final report is generated from raw records by
  `tools/report_transfer_campaign.py`; cleaned short and free-generation
  results and cancellation/teardown outcome are recorded there. No external
  publication, no daily launcher override, no altered model bytes.

## Validation scope

The most expensive traces/long-context checks target H1/H3 and the meaningful
H4/H5 overlap dependency; weak screens were rejected using end-to-end results,
byte accounting and correctness, without a full profiling/prefill sweep for
every variant. Compression is a measured sample across all three classes,
not whole-model compression or a GPU codec benchmark. No confidence interval
or universal numerical parity is inferred from two repeated requests.
