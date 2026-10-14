# Batched verification work log

## 2026-10-14 — baseline profile

Parent c730fbb; plan checkpoint 22d03c0. Full-layer groups, scratch64, reuse enabled, single-node CPU batching off. Q4 drafter, 128K capacity, 40-token coding prompt, 192 output tokens. Profiled control 41.49 TPS; width2 41.73; width4 35.38. No retries.

| Per-window CUDA event interval (ms) | Width2 | Width4 |
|---|---:|---:|
| Full verify GPU timeline | 62.81 | 99.75 |
| Routing/wait/fetch combined | 33.00 | 56.60 |
| Expert computation plus shared down/group/combine | 9.33 | 12.72 |
| Exposed CPU answer waits | 3.37 | 5.14 |
| Other stages | 17.11 | 25.29 |

Host enqueue overlaps GPU work and is not additive. CUDA intervals can include host submission gaps. This is diagnostic, not unprofiled throughput. Routing/fetch accounts for more time than expert arithmetic; even eliminating all 9.33ms would not alone meet the ~19ms saving sought at width2. Therefore complete batching must address transfer/route work, not merely arithmetic. Finer counters/events are being implemented before deciding the next supporting mechanism.

Prototype direction: exact-order DP4A weight decode reused across tokens, mixed Q2_K/Q3_K/Q4_K, fallback preserved. Existing prefill multirow helpers do not directly cover all required formats; no tensor-core speedup assumed.

## First complete prototype

MAYA_WIN_GROUPED=1 uses native quantized DP4A arithmetic with decoded weights shared across members. Both gate/up and down are implemented for Q2_K/Q3_K/Q4_K; singleton groups and unsupported formats retain original arithmetic. No extra tensor conversion, quantization or launches. Dimensions must be divisible by 256 and T<=8. Register pressure and shared allocation also affect singleton CTAs in the specialized kernel, so source reuse alone is not a speedup claim.

Instrumentation now separates route / disk wait / fetch and shared down; group diagnostic reads actual slot membership and copied bytes. It deliberately synchronizes, so diagnostic TPS cannot be used as a performance result. Actual on-device branch counters are being added before the diagnostic run. Baseline executable preserved as build/strata-before-grouped-verification.exe.

Next gate: controlled GPU-only full-logit comparison on fixed windows, partial commits and eviction pressure. If exact, run one small end-to-end A/B and inspect group multiplicity. Investigate complete window scheduling only where the split profile supports it. Avoid assuming that all repeated CPU experts should move to GPU: that can increase transfer demand and destroy overlap.

## Grouped arithmetic qualification

Full-model grouped-off vs grouped-on comparison passed bit-exactly: 28,188,160 logits, 182 rows, zero mismatches, zero retries. Includes alternating T5/T8, partial commits, two resets/repeats and 16GiB expert capacity eviction pressure. Source arithmetic matches checkpoint 59d5b5f; test binary hash is recorded in manifests (built before diagnostic-only counters). This qualifies controlled GPU arithmetic, not hybrid-placement changes or a throughput win.

Follow-up scheduler design must address disk temporary staging: F->stage[m] can be reused on a later route, so merely placing fetches on a side stream would be unsafe. All-RAM-only gating would change the baseline configuration and may not engage; investigate bounded per-window staging or a precise source-retirement fence. Keep this separate from grouped arithmetic.

## Actual engagement and group diagnostic (64-token coding sample)

The GPU counters equal eligible repeated groups for every Q2/Q3/Q4 stage: 4342 repeated GU/down groups at T3 and 5816 at T5. Actual GPU entries/slot groups: 20776/15376 at T3; 25817/16034 at T5. This is ~26%/~38% fewer decoded weight-call memberships in the ideal reusable portion, not an equal speedup or DRAM-byte claim. Pointer groups equal GPU logical-ID groups: no observed pointer fragmentation. Fetch entries equal distinct fetched IDs in every quant/scope: no duplicate fetch-copy bytes remain within current placement.

Fine diagnostic intervals: T3 fetch21.57ms, route2.75, disk-wait0.67, experts11.61; T5 fetch37.98, route5.02, disk-wait1.04, experts13.81. Diagnostic synchronization/readback perturbs CPU overlap and adds ~4.1ms/window; do not compare these intervals directly to earlier baseline as speedup evidence. Clean end-to-end grouped run is next. Files in group-diagnostic/Q4_K_M preserve exact counts and bytes.

This evidence deprioritizes redundant-copy elimination. Prioritize a safe route/fetch overlap experiment and arithmetic profiling; union-aware placement remains proposed and requires a measured capacity/makespan opportunity, not assumed free savings.

## Clean A/B: first grouped kernel

Identical output histories/acceptance, no phase/group diagnostics:

| Variant | Control TPS | 2 drafts TPS | 4 drafts TPS |
|---|---:|---:|---:|
| Grouped off | 41.93 | 41.91 | 35.09 |
| Grouped on | 41.73 | 41.76 | 35.24 |

No material gain. Retain as opt-in research, not recommended/default. Compiled GPU resource audit found GU registers ~59/60/58 baseline vs128/128/127 grouped for Q2/Q3/Q4; no local-memory spills. At512 threads this plausibly halves active CTAs. Refine based on resource/multiplicity evidence rather than endless rerunning. Exact performance attribution still needs kernel timing/occupancy, not register counts alone.

## Ranked next mechanisms

1. Safe asynchronous per-token fetch submission: complete bounded prototype built; join before all consumers; extra864MiB pinned RAM protects disk fallback source lifetime. Preserve serial route/disk-response ordering. Pressure parity now tests64GiB RAM tier to exercise disk paths, not just warm all-memory behavior.
2. Grouped-kernel resource refinement: preserve singleton efficiency and smaller member groups; reject extra launch/grid overhead if it outweighs reuse. Await concrete resource-grounded design.
3. Union-aware assignment: documented but deferred until a bounded eligibility experiment shows available byte/makespan savings. Current fetches are already unique under existing assignments. Do not implement a large shadow-routing simulator without evidence of benefit.

Trace capture hook added to fixed-window test for four warmed windows only; intended to demonstrate actual stream overlap if async candidate is correct/promising. Throughput remains separately measured.

## Async pressure qualification failed — blocked from speed claims

With64GiB RAM tier and16GiB GPU pool, async-off/on differed: max5.3676, mean0.1619;181/182 greedy rows. First14 logit rows exact; divergence starts row14 (second token of third window). No retries; token-owned disk staging exercised1344 times across repeats. Normal-memory async also differs from older normal reference, but extra pinned allocation changes available RAM tier, so this is not a clean same-capacity isolation. Serial normal vs serial pressure differ already fromrow0, likely a capacity-dependent prefill path; do not conflate that with the same-config async discrepancy.

Next decisive isolation retains async allocation and staging but joins each token's fetch before advancing routes. Added rare ring-lap/skipped-sequence counters without changing behavior. No numerical thresholds weakened. No performance claim for async while qualification fails.

## Paired refinement and async isolation follow-up

Mode 2 separates singleton work and processes repeated groups in pairs, reducing compiled register pressure without changing quantized arithmetic. The controlled normal-memory GPU-only comparison passed all 28,188,160 logits bit-exactly (182 rows, max absolute difference zero). Evidence: pairs-parity/comparison.json. Clean coding widths 2/4 are being measured; compiled resources alone are not a speed claim.

Async JOIN_EACH isolation retains the same staging allocation and side stream, but waits after each token fetch. It agrees with pressure serial reference for 78 rows, then differs; fully asynchronous diverges at row 14. No nonzero ring-lap report was observed. Next run the identical serial pressure control once to test reproducibility before further ownership changes. Extra pinned allocation and different memory-tier capacities must not be conflated with scheduling effects.

Clean mode-2 coding sample completed: control 42.04 TPS, two drafts 42.26 TPS, four drafts 35.79 TPS. Prior clean grouped-off was 41.93/41.91/35.09 TPS. Small cross-run differences are not a robust gain; no recommendation/default change. Evidence: pairs-clean/Q4_K_M (same acceptance and output histories to be retained for attribution).
