# Maya three-cost campaign — 2026-10-13

## Starting point and scope

Branch `research/maya-three-costs` starts at retained checkpoint `4f2f6e8`.
Preserved executable: `build/maya_transfer_lab_three_costs_baseline.exe`.
Unrelated untracked `maya-bench.txt` is left untouched. No publication.

Current diagnostic warm loop: 22.871 ms/token, 383 MB/token demand,
10.088 ms fetch including 0.594 ms overlap, 9.223 ms GPU arithmetic,
2.956 ms CPU-answer wait, 1.156 ms without GPU activity. These are not
independent additive costs. Goal: reduce whole-loop time without sacrificing
prefill, arithmetic qualification or ownership correctness. 60 TPS requires
16.667 ms/token; it is a target, not a promised result.

Three read-only GPT-6.1 Sol investigations: transfers (xhigh), GPU computation
(xhigh), CPU service/waits (high). Agents do not run inference or experiments;
only the parent uses this PC's inference resources. Research notes are separate
from established measurements. Parent will rank complete proposals, identify
dependencies, and run sequential experiments.

## Measurement rules

- Same GSQ-RCO 3.0bit Q4-attention packed weights, c=1, 22 GiB expert pool,
  retained RAM shadows/demotion policy, transition eviction, captured-only
  component overlap, and KDA graphs. No MTP, token speculation, K4V, expert
  skipping, additional quantization or CPU-model-specific affinity work.
- Select one short fixed-input coding workload per mechanism; use a second
  held-out/occupied-context workload to qualify promising combinations.
  Do not rerun the entire historical suite for each change.
- Preserve a baseline executable and manifests. Separate Nsight diagnostics
  from unprofiled timings. Fixed input history is not proof of equal routing:
  compare logits/route attribution where relevant.
- Before judging a mechanism, establish its engagement and prerequisites.
  Reject unsupported/incomplete negative conclusions. Test substantial
  supporting expansions when analysis identifies a credible dependency.
- Exact parity for transport/scheduling changes. Arithmetic changes require
  explicit numerical investigation; do not loosen thresholds. Preserve CPU/GPU
  placement distinctions from earlier campaigns.
- For retained mechanisms, qualify prefill and affected cold/warm transitions,
  ownership, and cancellation/teardown. Test targeted ablations of promising
  combinations, not exhaustive parameter sweeps.

## Log

- Inspected retained source/history and prior failed overlap, admission,
  transport and CPU-assignment experiments. Created isolated branch and
  preserved baseline binary before source edits.
- Research launched in parallel; experiment execution remains serial.

## First mechanism screens (not final retention decisions)

| Screen | First/repeat TPS | Evidence / decision so far |
|---|---:|---|
| Fresh retained baseline | 39.01 / 43.83 | 383 MB/token repeated demand, fixed masks |
| Q4 two rows/warp | 40.01 / 42.62 | All 39.65M logits exact; mixed timing, needs kernel attribution |
| Early CPU answer | 40.21 / 44.41 | Exact, same demand/counters; modest improvement requiring combination/confirmation |
| Windows spin PAUSE | 40.55 / 44.75 | Exact, same demand/counters; test supporting CPU combination |

Resident-GU diagnostic (profiled, not directly comparable to table): repeat
22.871→23.195 ms, overlap .594→.957 ms, arithmetic9.223→9.615 ms,
CPU wait2.956→3.507 ms. Mechanism engages; no speed gain. Implementing the
down-row-band extension to exercise full component readiness, and combinations
with CPU improvements, before rejecting the whole overlap direction.

## Dependency map and a priori bounds

- Q4 paired rows → reusable activation-only dmin work; unchanged weight bytes,
  k accumulation and reductions. Dense Q4 already moves ~4.185GB/token at
  ~1.3TB/s; research bounds ideal saving below0.9ms, realistic much smaller.
- Resident GU needs captured fork/join, bounded prefix-copy occupancy, immutable
  rank/source mapping after disk readiness, shared down exactly once. Sources
  and slots cannot be reused until both streams join. Masking retains rank hq.
- Down row bands extend that pipeline: preserve each output row's dot and rank
  accumulation, copy each byte exactly once, start first output half before
  second-half copy completes. Default callers keep full-row execution.
- Early CPU answer needs stable request snapshot and deferred processed
  retirement. Disk and multi-token windows keep the original path.
- CPU expert-ready pipeline replaces two pool generations with a synchronous
  ready-task graph, not asynchronous unowned jobs. Last GU task quantizes h,
  release/acquire publishes it to down tasks, final down completion permits
  per-output-row ordered combine. Prefill/multi-token CPU kernels unchanged.
- Windows PAUSE is a spin hint, not a fence/yield or a CPU affinity policy.
  Preserve all existing memory ordering and wall-time spin budgets.
- GPU early compute and CPU changes can compete for RAM differently: judge
  whole token loop, not sum isolated savings. No new quantization or skipped
  computation in any experiment.
