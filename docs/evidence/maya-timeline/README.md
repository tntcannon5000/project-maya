# Maya token-generation timeline

2026-10-12. Open `report/index.html` (also served locally on
http://127.0.0.1:8767/index.html). No performance optimization was applied.

## What was measured

- GSQ-RCO 3.0bit Q4-attention, stock Maya packed weights. RTX 5090,
  9950X3D, 128 GB RAM, c=1, 131072 context capacity, greedy, no MTP,
  suffix drafting disabled. Short coding/explanation/web prompts, 128 outputs.
- Nine unprofiled runs: three repetitions of each task after a warm-up,
  in one engine process. Prompt-cache disabled in the driver configuration.
- One Nsight CUDA/NVTX capture spanning a warm-up and three complete task runs.
  Analysis excludes warm-up and prompt forwards: 381 decoded forward passes,
  17,145 layer executions and 283,083 kernel executions. Output token 1 is
  sampled from prefill; the 127 later outputs each have a forward pass.
- Actual kernel timestamps and host-launch correlation identify each layer;
  NVTX ranges identify token wall time and CPU expert service. No CUDA events
  were inserted for this capture. Source additions are opt-in host labels and
  cudaProfilerStart; arithmetic, routing, transfer and ownership logic unchanged.

## Principal findings

| Average traced cost | ms/token |
|---|---:|
| Demand expert pulls from RAM | 14.318 |
| VRAM-to-RAM copies, predominantly whole-expert demotion | 7.171 |
| Dense projection kernels | 3.531 |
| GPU waiting for CPU answers | 2.764 |
| No recorded GPU kernel/copy activity | 2.011 |
| DSA/MLA kernels | 1.195 |
| Routed gate/up | 1.101 |
| Routing | 1.012 |
| Residual/norm | 0.815 |
| Routed down | 0.750 |
| KDA prep/recurrence | 0.468 |
| Embedding/head | 0.240 |
| Host-response wait | 0.106 |

Traced token wall means: CSV 36.047 ms, explanation 33.908 ms, web 36.515 ms.
The two weight-transfer directions together consume about 21.49 ms, roughly
60.5% of the mean token loop. This is measured activity, not a promise that
all of it can be removed. DMA and kernels are allowed to overlap; the token
CSV separately reconciles interval unions against wall time.

Demand payload is 463-534 MB/token. D2H payload is 385-416 MB/token.
The D2H copies are expert-sized (8,257,536 / 10,813,440 / 14,155,776 bytes),
and match `Glm5Model::fast_boundary` demoting GPU slots into RAM in
`src/core/glm_fast_path.cu`. Representative timelines show these copies
before layer computation. The old stage-event profiler began inside
`fast_layers`, after this boundary: it omitted this large cost.

CPU expert-service elapsed time is 15.8-17.7 ms/token, but most coincides
with GPU fetches. Only about 0.9-1.0 ms overlaps GPU arithmetic; 2.1-2.4 ms
overlaps GPU wait kernels. Never add CPU service to GPU elapsed time or
count a spin-wait overlap as useful arithmetic. NVTX CPU ranges include
thread-pool coordination; they are not CPU-core instruction samples.

This elevates **demotion/promotion churn and token-boundary scheduling**
alongside resident-compute/fetch overlap. Maya's additive tiers preserve
capacity, so dropping demotions without changing ownership is incorrect.
Potential experiments are selective retained RAM copies where capacity permits,
less churn, or safely overlapping demotion with other work. They are not
implemented by this profiling task.

## Unprofiled throughput and uncertainty

| Task | Weighted TPS over three runs | Individual-run range |
|---|---:|---:|
| CSV coding | 26.28 | 23.99-29.75 |
| Explanation | 29.14 | 28.04-29.78 |
| HTML/JS | 26.18 | 25.26-26.89 |

These 128-output tests differ in length from the preceding 256-output benchmark.
They are not evidence of an engine regression. Later requests share a process
and its learned expert placement; repeats are not independent cold starts.
CSV and explanation token sequences also differ across stock repetitions.

Nsight displaced nine expert slots (2731 stock, 2722 traced), and the traced
CSV output differs from stock's first CSV output. First explanation and web
outputs match. Therefore traced-versus-stock differences do not isolate profiler
overhead: instrumentation, placement and routing/output changes are confounded.
Do not scale each traced stage to manufacture an unprofiled stage breakdown.

Short occupied contexts are tens to hundreds of tokens. This is not a 128K
occupied-context measurement or a claim about long-context attention cost.
CPU context switches, per-core samples, PCIe hardware counters and OS scheduling
causality are not captured. No-GPU-activity gaps are identified without guessing
their cause. `miss_mask` counts host-service requests, not actual SSD reads;
the engine reports zero disk expert reads in these samples.

## Artifacts and reproduction

- `stock/requests.json`: output IDs, per-output arrival times, engine DONE counters.
- `trace-v2/requests.json`, `engine.log`, `manifest.json`: trace run conditions.
- `trace-v2/timeline.nsys-rep` and `timeline.sqlite`: native trace and export
  retained locally, not added to Git (approximately 0.55 / 1.54 GB).
- `report/tokens.csv`: every forward pass, transfer payloads, CPU overlap,
  wall-time reconciliation and stage durations.
- `report/layers.csv`, `layer_averages.csv`: actual and mean sequential layer
  spans and cumulative completion times.
- `report/kernels.csv`: all kernel starts/ends relative to token entry,
  durations, layer, category and full kernel name (local, about 52 MB).
- `report/copies.csv`, `cpu.csv`: transfer/CPU service intervals.
- `report/index.html`: nine selectable individual-token timelines; detailed
  sequential tables; slow-layer zoom; all-token layer heatmaps; run summaries.
- `report/checks.json`: all 45 layers mapped for every scored token; overlap
  partitions reconcile to wall time within floating-point rounding.

Driver: `.venv/Scripts/python.exe tools/profile_decode.py --mode stock
--out <directory> --repeats 3`; trace uses `--mode trace --repeats 1`.
Analyzer: `.venv/Scripts/python.exe tools/analyze_decode_timeline.py
docs/evidence/maya-timeline`.

The first Nsight launch attempt could not conduct the console handshake through
Nsight's buffered output; its owned engine was stopped, with no scored results.
The working driver runs inside the profiler process tree. A preserved
`build/strata-stock-profile-reference.exe` supplied all unprofiled results.
No inference process remains running; the lightweight report server is left up.
