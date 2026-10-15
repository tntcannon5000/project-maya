# Confidence policy and complete adaptive graph coverage

2026-10-15. Existing upstream policy: minimum draft confidence0.95, no mandatory drafts, at most4drafts. Same GSQ-RCO Q4-attention pack, Q4_K_M drafter, CPU011223344, pool22/RAM94GiB,128K allocated capacity,128output-token cap. No expert skipping or numerical threshold change. This is not a newly invented acceptance policy.

| Coding workload | Ordinary TPS (current) | Confidence, old T3/T5 graph coverage | Confidence, complete T2..T5 coverage | Accepted/offered |
|---|---:|---:|---:|---:|
| Short,40prompt tokens |44.04|57.06|56.78|92/97 (94.85%)|
| Longer,6,793prompt tokens |40.83|39.05|39.64|47/48 (97.92%)|

Do not claim the width extension increased short-task TPS: it is0.5%lower, while the ordinary control itself fell2.0%. The longer1.49%gain is modest and not statistically established. Practical result is approximately57TPS short,40TPS longer;60TPS remains unmet. The prior forced-width long task reached only24-25TPS. Policy choice supplies the large recovery; graphs supply modest verification savings. Different policies can produce different histories, so ordinary-versus-confidence TPS is a workload comparison, not exact-history attribution.

## Why high acceptance is not enough

Short policy makes35rounds:3offer zero drafts,6offer one,4offer two,5offer three,17offer four. Longer makes83rounds:48offer zero,26offer one,5offer two,4offer three. Its97.9%acceptance is only47accepted drafts total, not97.9%of all128generated tokens. Drafter execution still costs about196ms across83rounds even when confidence rejects candidates. It cannot approach2x by accepting almost all of a small number of offered drafts.

## Complete implementation and verified engagement

The original bounded graph prototype bypassedT2/T4. The expanded graph gate now uses the same native body, live pointer/configuration keys, per-width request credits, side-stream fork/join, rollback and teardown for everyT2..T5window. No arithmetic modification. The longer policy previously bypassed1260MoElayer calls; now all1470eligible calls are captured/replayed:126captures/1344replays, zero bypass/errors. Short policy plus cancellation/recovery records168captures/1806replays, zero bypass/errors. Matching old/new output IDs, offered counts and accepted counts for every paired request. Actual pool remains2523slots.

Actual CUDA helper: all widths, dynamic pointers, invalidation, stream joins, exact credits, abandoned capture;8captures/16replays pass. Full-model matched hybrid T2/T4:5,575,680bitwise-exact logits/36rows plus309,760exact prefix values,6793occupied prefix, partial commits/two resets. Both arms retire33superseded uploads, no sampled duplicate/orphan residents or invalid sources. STOP, recovery and clean teardown pass.

Cold prefill short718.2 ->718.4ms, longer3234.8 ->3215.0ms. Bounded no-regression smoke, not full-context or statistical qualification. Historical intermittent original graph mismatch remains unattributed; current passing checks do not erase it.

## Decision and next work

Retain complete graph coverage opt-in; no daily launcher/default change. Use the existing confidence policy as the practical candidate instead of mandatory drafts for all workloads. No broader quant/width sweep. Before a PR/default recommendation: one bounded writing task (not a full suite), targeted ownership edge-case qualification, and attribution of the preserved original numerical failure. Focus further performance work on exposed verification cost and unsuccessful drafter overhead only if a concrete mechanism has material headroom; do not infer savings from acceptance percentage alone.
