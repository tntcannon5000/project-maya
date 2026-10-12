# Maya transfer campaign

Baseline: `615973d`, stock engine source `20e27a2`. Development branch:
`research/maya-transfer-campaign`. Preserve stock executable and timeline tools.
Model: existing GSQ-RCO 3.0bit Q4-attention pack, unchanged. c=1, no
speculative tokens, MTP, K4V, dropped experts or altered quantization.

## Facts and open questions

The prior 381-forward trace averages 35.49 ms/token: 14.32 ms mapped-RAM
pull kernels, 7.17 ms D2H copies. These are diagnostic observations, not
unprofiled timings. CPU service overlaps mostly pulls, so GPU compute overlap
must be measured separately. Prior stock throughput is 26.18–29.14 TPS by task.

`fast_service` marks promoted RAM experts kRRelease. `fast_boundary` clears
their RAM table and frees their slots after compute/service completion. Tier
GC also removes dual copies. Spare replenishment then demotes VRAM victims
into free/victim RAM slots. Weights are immutable: the copy is required by
exclusive tier ownership, not changing weights. Demotions already use a
nonblocking copy stream and events. Their observed serialization therefore
requires investigation; merely saying “make them async” is insufficient.

Stock allocation: ~22.30 GiB GPU expert pool, 87.38 GiB pinned tier, 107.2
decimal GB routed weights. Pool lending during prefill consumes extra RAM
capacity. Full duplication cannot be assumed safe from model size alone.

## Ranked hypotheses and bounded experiments

Savings below are optimistic ceilings against the old trace, not forecasts.
Every result starts pending; enabled flags do not qualify as engagement.

| Rank / ID | Mechanism and evidence | Plausible ceiling / dependency / risk | Decisive experiment |
|---|---|---|---|
| 1 H1 | Keep selected immutable RAM shadows; avoid reconstructing weights on eviction | Up to 7.17 ms and ~405 MB D2H/token; constrained by class-specific free capacity and prefill lending | Bounded shadow budget; count retained bytes, shadow eviction hits, avoided copies, disk reads and pressure. Validate ownership and fixed-token logits |
| 2 H2 | Demand reuse admission/hysteresis reduces promotion churn; existing promote-min path | Up to demotion cost, but may worsen hit rate and H2D. No arithmetic change | Compare existing reuse admission with stock on identical forced routes; demonstrate fewer promotions/demotions and net latency/traffic benefit |
| 3 H3 | Move demotion submission after table publication or isolate copy scheduling | At most exposed 7.17 ms; existing async copies mean dependency/driver scheduling may dominate | Trace CUDA API plus stream order, then reorder only if proven safe. Preserve source slot until copy event; measure actual D2H/computation intersection |
| 4 H4 | Compute resident gate/up while demanded weights arrive; separate component readiness | Hide at most available independent arithmetic (~1–3 ms), potentially more with DMA; extra launch/SM contention | Complete resident/missing masks, separate fetch stream, event readiness and original reduction. Prove overlap on mixed hit/miss layers and unchanged logits |
| 5 H5 | DMA rather than GPU mapped-memory pull; batched/component transfers | Up to 14.32 ms only in impossible zero-copy limit; actual gain limited by link and host publication | Actual mixed-format payload microbenchmark (GPU pull vs DMA), then route-driven complete transport if feasible; verify bytes, bandwidth, CPU overlap and service latency |
| 6 H6 | Predict next-layer demand and hide useful copies | Hide demand cost only for correct, timely predictions; waste can evict useful data and delay demand | Existing predictor with actual useful/unused/late byte accounting. One conservative policy and one dependency-aware combination, not width sweep |
| 7 H7 | Lossless compress immutable packed experts during transport | Bounded by measured compressed-byte fraction; decompression and memory pressure may erase gain | Sample every quant class and multiple layers/experts; lossless roundtrip, ratio and decode rate. Reject on measured inadequate savings or pursue GPU feasibility |
| 8 H8 | More work on CPU avoids transfers when marginal CPU cost is cheaper | Save pull bytes at cost of CPU tail; 16–18 ms CPU service already mostly hidden | Calibrated alternate split on same weights/forced tokens; measure reduced transfers against increased exposed CPU wait; assess arithmetic differences separately |

## Dependencies and measurement

H1 and H2 reduce transfer demand; test together only if both independently
engage. H3 may help H1 misses. H4 needs independent transport progress and
complete readiness; H5 may be necessary to avoid SM contention. H6 depends
on demand-priority/transport behavior. H8 must be recalibrated for retained
transport. H7 precedes a production codec decision. No exhaustive sweep.

Use fixed input tokens at short and representative longer occupied context, plus CSV/explanation/
web free generation. Record binary/config/weight manifest hashes, same initial
expert profile, cache slots, memory and CPU plan. Separate stock repeatability,
placement-sensitive CPU/GPU arithmetic, and scheduling-only parity. Preserve
thresholds; investigate failures. Prefill at 2K/8K once per promising bundle.
Exercise pressure, cold/warm transitions, repeated requests and shutdown.
Unprofiled runs establish TPS; traces establish mechanism and exposed latency.
Profiler slot changes are documented, not silently normalized away.

## Audit

- Campaign opened from the measured baseline; original branch and executable
  retained. No production optimizations yet. Ownership audit located explicit
  RAM release and GC, plus already-asynchronous D2H event ownership.

- Eight mechanisms were screened, with complete DMA/resident-overlap prototypes
  and measured compression feasibility. Findings and intermediate failures are
  in `docs/evidence/maya-transfer/RESEARCH.md`.
- Checkpoint `3cf31ef` preserves the prototypes on
  `research/maya-transfer-prototypes`. Checkpoint `84c4f17` removes rejected
  engine mechanisms, restores stock arithmetic kernels, and keeps bounded RAM
  backing, after-embedding demotion scheduling, counters and ownership checks.
- The long fixture was corrected from an invalid serial-forward 16K attempt
  to real batched prefill at 8K occupied context. Invalid output is excluded.
  Stock engine with the corrected driver is isolated at `4bf9d4c`.
- Final cleaned-build qualification adds independent free-generation tasks,
  cold GPU-only exact-logit/weight checks, STOP/recovery and QUIT. The complete
  results, dependency map, limitations and retained settings are in
  `docs/evidence/maya-transfer/report/RESULTS.md`.
