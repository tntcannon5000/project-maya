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
