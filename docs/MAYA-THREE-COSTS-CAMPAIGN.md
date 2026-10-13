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

## Second checkpoint

All fixed-input screens below retain exactly all 39,649,280 logits, demand
bytes, cache counters and CPU assignment totals of baseline. These are single
screens, not confidence intervals or final retention decisions.

| Mechanism / dependencies | First / repeated TPS | Interpretation |
|---|---:|---|
| Early CPU + PAUSE | 39.97 / 44.54 | Simpler CPU support reference |
| Expert-ready CPU pipeline + early + PAUSE (profiled) | 38.51 / 43.11 | No diagnostic loop improvement; not directly comparable to unprofiled TPS |
| Prefix-phase CPU gate + pipeline + early + PAUSE | 40.60 / 45.06 | Small possible gain; need actual-gate counters and combined evaluation |
| Resident GU + two down row bands + early + PAUSE | 40.88 / 45.56 | Strongest initial screen; needs ablation/confirmation |
| Warp top-k + cooperative publication + early + PAUSE | 40.98 / 44.14 | Mixed timing; require stage attribution before retention |

CPU readiness review found no c=1 publication/deadlock defect. Phase gate is
eligible only for multi-expert CPU jobs (~17.15 of42 routes/token), although
readiness kernels launch for all42; do not describe it as controlling every
CPU expert. Added once-per-gated-callback counters rather than hot spin-loop
atomics to prove engagement in the combination trial.

Repaired the detailed analyzer: grid32 no longer implies a down copy, and
two GU launches/two down row bands are accounted independently. Reconciled
383MB/token on existing resident and CPU-pipeline traces. Large raw traces
and logits remain local; manifests and compact results are checkpointed.

## Complete overlap extensions and bounded arithmetic follow-up

- Down bands without resident GU:40.42/45.71TPS, exact. Resident split is
  unnecessary in this screen; omit its extra launch from the simpler candidate.
- Captured MLA MoE suffix plus bands:41.53/45.82TPS, exact;11 captures,
  3465 replays. Matching ablation saves only0.053ms, within variation. Capture
  excludes dynamic attention, positions and residual swaps; uses immutable
  FastState input/output buffers, cpu-plan key and exactly one route accounting.
- Review found and fixed inherited eligibility leaks: exclude pipelined window
  tokens, and scope captured-only overlap to the actual MoE capture body so
  multi-row/pre-fill consumers cannot inherit a prior decode mode.
- HC3 exact three-stage split plus simpler overlap/CPU candidate:
  39.98/44.83TPS versus41.53/45.82 without split; exact logits. Kernel boundaries
  replace both grid spins, same32-block reduction and ordered arithmetic;
  mixed values temporarily use the separately allocated x buffer. Sinkhorn
  runs only in block0 in phase2; its parallel replication was not a32x latency
  saving. Negative screen, not a retained optimization.
- Full diagnostic combination (Q4 rows, warp publication, CPU pipeline/phase
  gate, down bands, suffix graphs): repeat23.144ms diagnostic loop, overlap
  1.288ms, fetch10.519ms, arithmetic9.393ms, CPU wait3.410ms. All logits exact.
  More overlap is real but not a whole-loop win. Q4 row sharing costs3.181ms
  versus original3.211ms: negligible practical gain. Warp route family falls
  ~0.976 to0.793ms, a real but small opportunity. Need unprofiled scheduling
  ablation and simple combined candidate before the final decision.

## Selection and qualification checkpoint

- Scheduling combination without Q4 row sharing:41.60/46.49TPS, exact;
  7167 expert-pipeline batches and51625 gated callbacks (includes setup).
- Same GPU changes without CPU pipeline/phase limit:41.76/45.53TPS, exact.
- Expanding pipeline/phase coverage to one-CPU-expert routes:
  41.26/46.07TPS, exact;12717 pipeline batches and75757 gated callbacks.
  Coverage expands as intended, but no further gain: single-expert extension off.
- Fresh back-to-back baseline confirmation:40.47/44.36TPS; selected
  candidate40.72/45.90TPS. Warm saving0.756ms/3.47% TPS, first-request
  gain0.60%. Do not present46.49TPS as a guaranteed result or sum standalone
  improvements. Original baseline screen43.83 and confirmation44.36 bracket
  visible run variation.
- Selected configuration: early CPU answer, PAUSE, two down bands, MLA MoE
  suffix capture, warp route/publication, multi-expert CPU task pipeline with
  prefix gate8. No resident GU split, Q4 paired rows, HC split or single-expert
  pipeline. `run_three_costs.py --candidate` reproduces these opt-ins.
- Hardened the CPU readiness interface to carry immutable per-job pointer/seq;
  asynchronous windows/calibration use the existing CPU path. No global
  readiness context can be accidentally read by a later window job.
- Detailed CPU-stage NVTX spans are now separately opt-in via
  MAYA_CPU_STAGE_TRACE. This avoids adding many new host range calls when
  comparing ordinary whole-token diagnostics against the preserved baseline.
  Earlier detailed traces retain their actual old instrumentation manifests;
  unprofiled comparisons remain the headline evidence.

## Completed qualification

- 8K occupied corpus: baseline first/repeat 36.38/40.82 TPS; candidate
  37.61/42.25 TPS. All 39,649,280 logits exact, counters identical. Repeated
  prefill 3561.222 vs3558.099ms: no material change.
- New warm diagnostic:22.880ms vs previous22.871, effectively flat. Overlap
  rises0.594→1.271ms, but fetch/arithmetic/CPU wait grow. This trace proves
  overlap and routing mechanisms, not the full unprofiled throughput gain.
  Report both; do not manufacture per-stage attribution or add isolated gains.
- All10,752 layer-route masks match in order. Expert IDs are not in these
  NVTX fields; full-logit comparisons complement them.
- Same-binary baseline-flag/candidate-flag lifecycle sequence:64-token warmup,
  STOP after8,32-token recovery,128-token web output, QUIT. Every output ID
  identical; both processes exit cleanly with shadow ownership verification.
  Held-out web38.78→39.01TPS (+0.59%), not a reliable speed claim from one run.
- Cold tiers (warming disabled):761 scored disk reads;16 scored tokens,
  2,478,080 logits exact and every non-timing counter exact. CPU lane is off
  in this cold configuration, so no hybrid parity extrapolation is made.
- Bounded8-token HC proof trace:544 launches of each phase (68/token), with
  1,239,040 prefix logits exact. Phase2 alone0.968ms/token versus the entire
  original HC family~0.81ms; complete implementation is correctly rejected.
- Retain selected opt-in combination, not rejected prototypes. Source,
  experimental checkpoints, compact evidence and raw local traces preserved.
  No external publication; no inference process intentionally left running.

Final tables and figures: `docs/evidence/maya-three-costs/RESULTS.md`.
