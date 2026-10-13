# Missing-expert execution campaign — 2026-10-13

Start: retained `86c6ea0`, isolated branch `research/maya-missing-expert-cost`.
Previous confirmed repeated short/8K TPS: 45.90/42.25. This campaign gets no
credit for prior improvements. Preserved executable:
`build/maya_transfer_lab_missing_baseline.exe`.

Goal: reduce joint demand-upload and CPU-expert completion time, preserving
GSQ-RCO 3.0bit Q4-attention, c=1, all required experts, numerical qualification
and prefill. No MTP, speculative tokens, K4V, new quantization or publication.

Three delegated source investigations cover residency/traffic, transfer overlap,
and CPU completion/RAM contention. Only the parent runs inference. Use one short
fixed-input coding continuation for screens, plus held-out/8K qualification
where justified. Do not repeat the full historic suite for every experiment.

Known constraints: previous diagnostic profile was flat despite modest unprofiled
gains. Short warm traffic remains 383MB H2D/token and CPU logical payload 564MB/t.
Only ~2ms of expert computation can overlap same-layer demand. Frequency-only
admission, integrated DMA, resident-GU splitting, HC splitting and dense Q4 row
sharing already failed; revisiting requires a concrete mechanism change.

Retain immutable manifests, exact logits and traffic counters. Policy changes
can alter CPU/GPU arithmetic assignment, so distinguish ownership/scheduling
parity from backend-placement differences. No threshold relaxation.

## First implementation checkpoint

- Fresh preserved baseline: 41.40/45.71 TPS, first/repeated request.
- Paired gate/up row-band streaming: one paired-range pull per half; each half
  covers complete 32-row hidden quantization blocks. First GU runs during second
  prefix fetch; shared down runs once; CPU prefix-ready remains after full prefix.
  Same persistent bytes/layout and slot lifetime. Screen 41.77/46.88 TPS, all
  39,649,280 logits exact and all traffic/cache/assignment counters unchanged.
- Coarse single-expert CPU executor: one worker generation, work-conserving GU
  claims, last GU completion publishes unchanged whole-vector hidden quant;
  independent down/output stripes eliminate separate combine scheduling. No
  fixed worker-arrival barrier and no prefix-phase throttling for this path.
  Screen 41.81/46.87 TPS, all logits/counters exact; 5605 actual batches.
- Neither isolated screen establishes a reliable gain yet. Combining next.
- Research caveat: existing down fetch already overlaps nearly all GU on demand
  routes. Prefix streaming could merely relocate overlap; any gain needs traced
  contention/critical-path evidence, not counting overlap twice.

## Second checkpoint: decisive limits

- Combination screen 41.68/46.45 TPS, exact. Combined diagnostic warm loop
  22.898ms versus retained 22.880ms: flat. Exposed fetch falls9.107→8.399ms,
  but GPU arithmetic rises9.401→9.905ms and CPU wait3.160→3.462ms. Actual
  overlap1.765ms, no claim of additive gains or profiled net improvement.
- Four independent mapped-copy loads before stores:40.83/45.82TPS exact;
  no useful independent gain. Same bytes and graph topology.
- Online16 residency: matured12600 causal anchors, changed7582 victims.
  First uploads464.14→483.53MB/t, repeat383.00→367.94MB/t. Route-logged TPS
  40.00/46.13 is diagnostic, not controlled unprofiled confirmation. Full hybrid
  logits differ (max3.42, greedy99.22%) as backend placement changes. Not
  eligible for retention on warm-only bytes. GPU-only isolation follows.
- New substantial CPU hypothesis: paired Q2 rows in AVX512 retain independent
  original AVX2 integer/FMA/reduction chains, sharing activation loads across
  gate/up or adjacent down rows. Unlike changing reduction widths, widen over
  independent output rows. Requires exact standalone row tests and full-model
  logits before performance can qualify it. Q3/Q4 remain original fallbacks.

## Arithmetic and phase qualification

- Paired Q2 standalone tests passed exact dots (96 pairs across256/512/2048/4096),
  whole512x256 and4096x2048 split experts, clamp/SwiGLU, hidden activation bytes,
  misaligned valid data and odd down-tail ranges. Full-model Q2 screen is exact
  across39,649,280 logits,40.84/46.21TPS. No useful standalone gain established.
- CoarseCPU + pairedQ2 + four-load copy:40.82/45.39TPS exact; adding plausible
  mechanisms worsened the screen. Do not retain by summing isolated numbers.
- Keep8 CPU callbacks admitted until all demand bytes arrive, with copy-stream
  generation publication and existing work-conserving completion escape:
  41.49/46.30TPS exact. No major gain; late CPU completion remains the tradeoff.
- Online16 GPU-only isolation:9,912,320 logits exact, zero CPU experts. The
  hybrid difference is backend placement, not evidence of lost/corrupted copies.
  Reject its first-request regression and mixed-backend divergence for retention.

## Confirmation and engagement

- Fresh short confirmation: baseline41.38/46.29, coarse CPU41.73/46.68TPS.
  Warm gain0.84% is smaller than the1.29% movement between baseline runs.
- 8K occupied confirmation: baseline36.40/42.15, coarse CPU37.36/41.79TPS.
  Repeat regression0.86%; all39,649,280 logits/counters exact. No reliable
  general coarse-CPU gain. Do not change daily settings.
- All-mechanism engagement diagnostic:32 tokens,1344 routes,2688 paired
  prefix4 launches,2688 down-component4 launches,2688 GU/down launches each.
  Every phase release follows all fetches on their copy stream. Separate
  counter run verifies9,263,104 Q2 GU rows,18,495,360 paired down rows,
  30,848 fallback-tail rows and1682 coarse jobs (includes setup). First-request
  prefix2,478,080 logits exact. Second-request cache differs from128-token
  fixture, so no second-request parity claim against that reference.
- Nsight initially hid child stderr. Fixed the reusable harness with
  `--show-output=true`; used an unprofiled diagnostic counter run to recover
  actual helper engagement, without inventing missing trace counters.
- One remaining targeted check qualifies GU-only (not its failed combination)
  on short and8K fixtures before final retention decision.

## Final decision

GU-only confirmation41.83/46.73TPS,8K37.68/41.77TPS, all logits/counters exact.
Like coarseCPU, its small short gain does not generalize to repeated8K context.
Keep the previously retained configuration, all new flags off. No60TPS claim,
no default/daily configuration change, no external publication. Six complete
mechanisms plus combinations/ablations are preserved as opt-in research with
negative bounds and engagement evidence. No further broad benchmark sweep.
Final report: `docs/evidence/maya-missing-costs/RESULTS.md`.
