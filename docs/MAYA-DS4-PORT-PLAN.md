# Maya / DS4 comparison and base-decode port plan

2026-10-12. Target: 60 tokens/s, c=1, without MTP, token speculation, K4V,
expert skipping or pruning. Preserve Maya prefill and its memory-tier design.
This is a source audit plus one diagnostic profile, not an implemented speedup.

## Starting evidence

Maya stock source 20e27a2; local evidence checkpoint b346b9b. DS4 4daa992
(test-only driver atop the retained engine). Hardware: RTX 5090, 9950X3D,
128 GB DDR5-6000. Current target model is GSQ-RCO 3.0bit Q4-attention.

| Short benchmark | Maya target model | DS4 Antirez Q2 reference |
|---|---:|---:|
| Mean decode TPS | 29.8 | 30.67 |
| 2K prefill TPS | 1080 | 955.90 |
| 8K prefill TPS | 2280 | 819.38 |

Different weights: these are observations, not engine speedup ratios. The target
file fails DS4's fixed 46-block loader check because it declares 45 blocks.
Do not spend an open-ended effort porting that model into DS4 merely to port
scheduling back into Maya. Stock-versus-modified Maya on the same packed model
will be the controlled comparison. Maya's compatibility pack converts selected
small projections to BF16; expert weights are unchanged.

Our strongest DS4 scheduling bundle improved 16K occupied-context decode from
23.79 to 25.44 TPS (+6.94%), prefill 925.20 to 917.30 TPS (-0.85%), with exact
154,880-value logits after 128 forced tokens. It combined component staging,
resident gate overlap, mapped IDs and batched DMA. This does not predict a 7%
Maya gain, and certainly does not establish a 2x gain.

## New Maya diagnostic profile

One standard benchmark, same config, STRATA_GLM_NO_MTP=1 and
STRATA_GLM_PROF=64. Raw evidence: evidence/maya-reproduction/port-audit-profile.
Existing profiler records consecutive events on the compute stream and collects
them after completion. It skips its first 64 calls, then reports 858 profiled
steps, 33.15 ms/step. This spans the benchmark's fast-path calls, including
short-prompt processing; it is not an isolated fixed-context decode capture.
Per-layer events perturb execution (measured task TPS drops to about 24-27),
and the profiler disables experimental layer graphs. Never use this run for
throughput claims or compare its prefill TPS with the uninstrumented baseline.

| Measured stream interval | ms/profiled step | Share |
|---|---:|---:|
| RAM expert fetch | 16.000 | 48.3% |
| KDA input projections | 2.478 | 7.5% |
| Exposed CPU answer wait | 2.422 | 7.3% |
| Routed gate/up | 1.853 | 5.6% |
| Routed down | 1.425 | 4.3% |
| MLA | 1.404 | 4.2% |
| Route | 1.210 | 3.6% |
| Router/shared projections | 1.044 | 3.1% |
| KDA output projections | 0.896 | 2.7% |
| Host/disk response wait | 0.249 | 0.8% |

Normal scored tasks used 63-77 CPU experts/token, with 13.4-16.3 ms CPU work,
42-56 RAM fetches/token and zero disk expert reads. CPU work overlaps the GPU;
do not add the full CPU duration to stream time. Nor does removing the CPU wait
remove the CPU work. Disabling CPU execution would increase transfers.

## What differs and what to preserve

| Mechanism | Maya | Retained DS4 | Decision |
|---|---|---|---|
| Miss execution | Calibrated hybrid CPU/GPU | Predominantly upload weights and execute on GPU | Keep Maya hybrid |
| Memory ownership | Additive RAM/VRAM tiers, per-layer slots and demotion | Owned pinned expert source plus GPU cache | Keep Maya capacity/ownership model |
| RAM fetch | GPU kernel reads mapped RAM on compute stream | Upload-stream DMA, optional CUDA 13 batch submission | Prototype alternative transport |
| Resident experts | Gate/up starts after fetch stage | Early gate/up overlaps transfers | Highest-value scheduling port |
| Component readiness | Full expert blob fetched before gate/up | Gate/up and down have separate readiness | Second part of scheduling bundle |
| Routing control | Device tables, mapped ring, service thread | Mapped IDs and cache tickets | Preserve Maya; avoid duplicate machinery |
| Prediction | Optional next-layer router, max two copies, phase/rank controls | Top-6 retained, wider trials inconclusive | Test only after transfer path is improved |
| Prefill | Layer-major batches, tensor cores/MMQ, staging and borrowed pool | Pipelined staging/MMQ | Preserve Maya; no wholesale replacement |
| Arithmetic shortcuts | Own native kernels/CPU numerical contract | Several optional arithmetic variants failed parity gate | Do not port unresolved variants |

Source anchors: src/core/glm_fast_path.cu fast_moe_experts (route, wait, fetch,
gate/up, down, CPU wait); src/kernels/cuda/glm_fast.cu moe_fetch_kernel and
moe_cpu_wait_kernel; src/core/glm_fast_state.hpp mark/collect; DS4 ds4_cuda.cu
cuda_stream_upload_batch and ds4_gpu_pinned_resident_gate. Maya's startup PCIe
calibration uses cudaMemcpyAsync, whereas actual RAM-tier demand fetch uses a
GPU load/store kernel: revisit calibration after choosing the real transport.
An old source comment quoting 11.5 GB/s is not a measurement on this machine.

## Ordered implementation experiments

1. **Readiness-aware resident overlap.** Keep stock path as a runtime fallback.
   Publish resident/fetched/CPU rank masks and immutable pointers per route.
   Move fetch onto a separate stream (first retaining the existing copy kernel)
   and execute resident gate/up while it runs. Follow with missing gate/up and
   the original rank-ordered reduction. Preserve shared expert exactly once.
   Pin every referenced slot until all consumers finish; a CPU/disk/predicted
   expert must never be treated as ready early. Measure contention: a GPU copy
   kernel itself occupies SMs, so overlap alone need not improve TPS.
2. **DMA transport plus component readiness.** Reuse Maya's service thread and
   mapped route publication to submit actual-route RAM copies. Compare upload
   DMA with the kernel transport using the same ownership contract. Gate/up
   completion releases gate/up work; down completion releases down work.
   Integrate batching where CUDA supports it, with a portable fallback.
   Do not make host pointer publication equivalent to transfer completion.
   Never allow promotion/demotion to recycle the RAM source during DMA.
3. **Recalibrate hybrid split on the completed path.** Measure effective copy
   and compute cost under concurrent RAM traffic, rather than relying solely on
   isolated memcpy calibration. Compare automatic hybrid with one bounded
   alternate split; CPU-off is a diagnostic, not the assumed target design.
4. **Useful prediction rather than maximum prediction.** Measure predicted
   bytes, useful arrivals, late arrivals, unused copies and demand delay.
   Try conservative existing prediction controls on the completed bundle
   before extending prediction count. DS4 top-6 is not a portable optimum.
5. **Remaining dense/attention work.** Only after another profile: projections,
   MLA, fusion or graphs. Do not import DS4 arithmetic paths that failed its
   numerical gate. Extra cache slots and cache-policy tuning come after exposed
   transport is measured, not before.

The first two stages form a dependent scheduling experiment: test their
combination, not just isolated pieces that lack a supporting transfer path.
Known DS4 graph and independent-demand-queue results do not justify prioritizing
them. Maya already has device-side routing and optional prefetch; copying those
features again is not progress.

## Measurement and acceptance

- Freeze stock packed weights/config and record hashes before engine edits.
- Add a bounded decode-only trace with precise byte accounting and stream
  overlap. GPU pull kernels do not appear as ordinary H2D memcpy transfers;
  count their bytes explicitly. Track gate/up/down separately and expose
  readiness wait, CPU tail and promotion/demotion traffic.
- Use the existing three short tasks uninstrumented for a quick screen. Keep
  c=1, 128K capacity, temperature zero, no speculation and no expert dropping.
- Keep a fixed-token coding continuation at 16K occupied context to prevent
  different generated routes or output lengths masquerading as a speedup.
- For retained changes, compare same-input logits and per-rank results;
  scheduling-only changes should preserve arithmetic order. Establish baseline
  hybrid repeatability first and investigate differences instead of relaxing a
  gate. Exercise cold/warm cache, mixed hits/misses, eviction, cancellation and
  repeated requests. Check 2K/8K prefill and memory reclamation once per bundle.
- Keep separate commits for instrumentation, resident overlap, DMA/component
  readiness and calibration. Retain only useful changes; log negative results.
- Maintain default/fallback paths for other devices and mixed quant formats;
  these tests establish only this NVIDIA machine's performance.

60 TPS requires about 16.67 ms/token versus roughly 33.6 now. Even eliminating
the entire measured fetch interval would be an unrealistic upper-bound thought
experiment, not a forecast. Faster transfers, overlap and lower exposed compute
will likely all be needed. No evidence yet establishes that 60 TPS is achievable
with this model and hardware without speculation. Optimize the measured critical
path and retain prefill rather than promising that target from a simple port.
