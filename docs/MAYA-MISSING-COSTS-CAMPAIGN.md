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
