# Maya demand-transfer and exposed-wait campaign

Start: retained `20af539`, branch `research/maya-demand-campaign`. Preserved
executables: `maya_transfer_lab_retained_r1.exe`, `strata_retained_r1.exe`.
Do not attribute previous gains to this campaign. Same model and packed
weights, c=1, 22 GiB expert pool, expanded RAM backing and layers demotions.
No expert skipping, speculative tokens, MTP, K4V or new quantization.

## Ranked, distinct hypotheses (provisional bounds, not promises)

| ID | Mechanism / evidence | Optimistic useful saving | Complete experiment and rejection criterion |
|---|---|---|---|
| D1 | Layer allocation from held-out route frequencies and avoided-byte value; existing code uses global ranking as artificial share | 10–25% demand bytes (40–100 MB, ~1–3 ms) if current partitions are poorly placed | Train on writing/web, hold out CSV and cross-task transitions. Byte-constrained slot allocator, same total pool. Replay exact routes with oracle/static bounds, then real engine. Reject if traffic or wall time does not improve |
| D2 | Recency-aware eviction rather than long-lived counts | Up to trace-measured avoidable re-fetches; estimate after replay | Existing LRU policy is a genuinely different eviction rule. Replay LFU/LRU/oracle, then engine on retained backing. Track evictions, reuse distances and routing differences; reject transient benchmark-only benefit |
| D3 | Stage GU and down transfers separately, overlap down pull with the single original gate/up launch | Hide at most gate/up duration or ~one third of pull time; likely 1–3 ms | Independent transfer stream, component-ready events, unchanged arithmetic/reduction and slot lifetimes. Prove down-fetch/GU overlap, exact GPU-only parity; reject overhead > hidden time |
| D4 | Batch independent expert pulls in one grid instead of serial per-expert grid loops | Bounded by current ~37 GB/s vs isolated 45–53 GB/s; ~1–3 ms | Fully mapped mixed-format payload copy kernel with same ownership and consumer barrier. Exact bytes/logits and actual timing. Reject bandwidth gain absent end-to-end benefit |
| D5 | CUDA layer graph replay reduces host submission gaps | No-GPU budget ~1.95 ms, practical fraction smaller | Existing Maya KDA graph path, verify captures/replays and route/ownership parity; compare trace GPU gaps and free generation. Not a generic arithmetic rewrite |
| D6 | Per-layer CPU split based on exposed wait/transfer balance | Up to CPU-wait time plus avoided fetch; measured bound needed | Train type/layer costs from traces/calibration, bounded per-layer plan; preserve expert selection and all computations. Compare backend-controlled parity separately from placement changes |
| D7 | Confidence-filtered expert prediction with demand priority | Only useful timely bytes count; previous ~36% wasted | First measure top-rank useful bytes/trace opportunity on retained config. A selective policy must beat its complete no-prediction counterpart; stop if traffic/lead time cannot support benefit |

Dependencies: D1/D2 change the best D6 split. D3 may need D4 or D5 to repay
submission costs. D7 must not starve demand and should follow cache choices.
Test promising complete pairs with targeted ablations, not exhaustive sweeps.

## Qualification

Primary baseline is retained candidate, not stock. Identical forced inputs
do not establish identical routes: save actual route IDs and masks. Separate
diagnostic captures from unprofiled measurements. Short and 8K occupied
contexts, writing/coding free generation and transitions; record actual
capacity and RAM headroom. Timing differences below previous ~4% variation
need targeted confirmation. Exact backend-controlled comparisons; no relaxed
thresholds for new differences. STOP/recovery/QUIT and prefill lending for
retained changes. Large raw artifacts remain local; no external publication.

GPU arithmetic receives one bounded kernel-duration breakdown. Pursue it
only if a substantial specific inefficiency is established. The dominant
first targets remain external transfers and exposed waits.
