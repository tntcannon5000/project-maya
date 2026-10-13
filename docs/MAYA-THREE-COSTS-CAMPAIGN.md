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
