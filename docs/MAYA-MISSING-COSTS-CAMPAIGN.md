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
