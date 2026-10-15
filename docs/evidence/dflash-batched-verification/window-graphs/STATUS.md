# Current update: matched qualification passes, speed gate in progress

The sections below retain the historical failure and investigation. The competing-promotion fix is now implemented opt-in. Current graph-off/on comparisons pass 7,434,240 logits each under controlled GPU and matched hybrid execution. Hybrid prefix sidecars are also exact. Neither result establishes the cause of the original intermittent mismatch. See PROMOTION-OWNERSHIP.md and hybrid-owner-fix/qualification.json. A bounded experimental serving comparison is now allowed; production sign-off remains open.

# Native MoE CUDA graphs — correctness gate remains open

Implemented opt-in `MAYA_WIN_MOE_GRAPH=1`, with dynamic routing, complete component stream fork/join, replay request/component credits, ring capacity guard, safe capture rollback, fatal state after uncertain live launch failure, diagnostic bypass and buffer-lifetime invalidation. Production defaults and launcher unchanged.

The isolated CUDA test passed dynamic pointer changes, T3/T5 replay, side-stream join, key invalidation, all host credits, abandoned capture and disabled-key fallback. Build succeeds.

## Full-model evidence

Same binary, 6,793 occupied prefix tokens, 128K capacity, GPU-only plan, 16 GiB expert pool and 94 GiB RAM; paired groups, HC batches, component transfer, exact columns enabled.

| Test | Values | Exact | Maximum absolute difference |
|---|---:|---|---:|
| First graph-off/on, six T3/T5 windows × two resets | 7,434,240 | No | 2.012939453125 |
| Narrow prefix + first window, separate graph-off/on | 309,760 | Yes | 0 |
| Original graph-on repeated unchanged, compared to original off | 7,434,240 | Yes | 0 |

The first failed comparison matched all 48 greedy picks but is still a failure. The narrow direct result also matches the original off result. All three graph-on runs used the same binary. Passing rechecks do not resolve the original discrepancy. No throughput claim or hybrid/serving qualification is justified yet.

Graph engagement in the original full run: 84 captures, 420 replays, no capture/launch failures or bypasses. The 1,680 captured nodes equal 42 layers × (18 T3 + 22 T5 nodes). Ring-credit waits are substantial host throttling (427.284 ms and 411.842 ms across each six-window repeat); they overlap GPU work and are not additive GPU costs. Saved fixed-input call times are diagnostic only while numerical failure remains unresolved.

Read-only audit found no captured host job-array lifetime or missing component join. A possible background-versus-demand duplicate promotion ownership case is under investigation; it is a hypothesis, not an attributed root cause. Next step is targeted ownership evidence, not throughput reruns. Raw logits remain local.

## Targeted host ownership observation

Added opt-in `MAYA_WIN_OWNER_COUNTS=1`: after compute and service drain, under the existing metadata lock and before boundary edits, count duplicate residents, residents not matching `slot_of`, competing background/demand promotions, and invalid pinned promotion sources. It performs no extra GPU readback and does not change placement. It is diagnostic and remains off by default.

One original six-window/two-reset graph-on fixture demonstrated competing promotions followed by accumulating duplicates: 0 initially, 1/3/19/20/22/27 during the first sequence, and up to 34 in the second. Up to 16 pending promotions simultaneously competed with another mapped GPU slot. No invalid promotion sources were observed. This run's 7,434,240 logits exactly match the original graph-off reference, so duplication alone does not establish the numerical root cause.

Source mechanism: the background H2D destination is Landing while the source remains routeable RAM. A demand route can promote the same expert into another spare. Background retirement then unconditionally assigns `slot_of[key]` to its own destination, orphaning the demand copy. Existing published-pointer checks explicitly did not reject duplicate unreferenced residents. Both copies can contain correct bytes, but this wastes capacity and complicates subsequent demotion ownership.

Next work should preserve one authoritative owner through competing promotion retirement and conditional table clearing, including in-flight demotions and retained RAM backing. Do not blindly free a duplicate that remains a live transfer source. Diagnose the original numerical failure separately; do not label this discovered ownership case as its cause. Graph performance remains unqualified; no serving A/B was run.

## Completed graph speed gate (2026-10-15)

Current exact GPU/hybrid qualification is followed by two bounded coding workloads. Short40-token prompt: warm two-draft50.82 ->52.42TPS, four-draft51.22 ->52.91TPS. Ordinary control45.27/45.06TPS. Actual84captures/6258replays across the four speculative requests, zero bypass/errors; STOP/recovery and teardown pass. Every paired output history and acceptance count matches.

Reversed-order6,793-token coding prompt: ordinary40.87/40.96TPS; two-draft24.02 ->25.05TPS (+4.29%), four-draft23.57 ->23.78TPS (+0.91%, not a reliable gain from one pair). Cold prefill3232.9 ->3219.8ms, a bounded smoke check. Full paired histories/counts match. Forced drafting remains much slower than ordinary on this task because acceptance is only35.8%/30.7%.

Retain graph implementation opt-in; no default or launcher change. The short-task graph gain is modest, and60TPS is not achieved. Preserve historical intermittent failure as unattributed; production sign-off and forced ownership edge-case validation remain open. Do not repeat the same width/quant matrix.

Next decision: inspect and exercise the existing confidence-based policy with this complete combination, rather than add more unconditional drafting or another general kernel sweep. Offline first-draft counts show the long two-draft task accepts35/74 first drafts overall, but17/19 when confidence>=0.95; this is descriptive only, not a simulated policy/TPS result. Confidence gating changes histories and must be measured if used. At the forced policy's1.73emitted tokens/round,60TPS requires28.83ms total per round versus65.32ms verification alone; dispatch optimisation cannot bridge that gap. See window-graphs/SERVING-RESULTS.md and acceptance-budget.json.
