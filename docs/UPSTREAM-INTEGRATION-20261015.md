# Upstream integration — 15 October 2026

## Scope and checkpoints

- Local baseline: `2ce956f`, preserved as `checkpoint/pre-upstream-20261015`.
- Integration branch: `integration/upstream-20261015`.
- Upstream target: `07c3dc9` (main, includes v1.1.11 and eight subsequent commits).
- Preserved executable: `build/strata-pre-upstream-20261015.exe`.
- Existing untracked diagnostic captures remain untouched.

## Measurement

Two coding prompts, 256 output tokens each, 128K allocated context, c=1,
same packed GSQ-RCO 3.0bit Q4-attention weights and Q4_K_M drafter.
Each process runs a warmup, ordinary decode, then confidence DFlash (maximum
four drafts, confidence threshold 0.95). Ordinary decode retains the drafter
allocation for equal available memory. This tests decoding disabled versus enabled,
not a separate no-drafter memory configuration. No GPU profiling capture enabled;
the existing round timing counters and one-second device sampling are retained
identically on both sides. Exact commands, environments, token IDs and binary hashes
are saved in each evidence manifest. Source merging overlaps the old executable's
baseline run; compilation does not overlap inference.

## Merge decisions

- Retain our transport diagnostic target and add upstream MTP source.
- Keep ownership auditing and deferred demotions; incorporate upstream resident checks.
- Preserve native CUDA gate/up and down band offsets while adopting upstream multi-row
  HIP indexing. Shared-down output partitions remain independent of routed band offsets.
- Incorporate upstream all-resident wait/fetch omission, disabling our transfer overlap
  when there is nothing to fetch.
- Keep verification component fetch and graphs; upstream alternative multi-token path
  cannot select component fetch because it returns before that path's component joins.
- Keep upstream DFlash all-CPU residency adaptation and Windows pool budget correction.

## Checklist

- [x] Preserve rollback branch and executable.
- [x] Measure both baseline workloads.
- [x] Build integrated source and resolve compiler/semantic issues.
- [x] Repeat matching workloads; compare outputs, prefill, acceptance and expert capacity.
- [x] Run focused validation of merge-sensitive code.
- [x] Investigate regressions, checkpoint and summarize.

## Primary results

| Coding workload | Mode | Before TPS | Merged TPS | Change |
|---|---|---:|---:|---:|
| Short, 40 prompt tokens | Ordinary | 40.72 | 41.13 | +1.03% |
| Short, 40 prompt tokens | DFlash confidence | 50.95 | 51.33 | +0.75% |
| Longer, 6,793 prompt tokens | Ordinary | 38.26 | 37.90 | -0.94% |
| Longer, 6,793 prompt tokens | DFlash confidence | 40.17 | 41.73 | +3.89% |

Each measured request emits 256 tokens. These are bounded single samples, not a
statistically established speedup or an unconditional no-regression guarantee.
Ordinary performance is within about 1%; no material regression demonstrated.
The longer DFlash improvement also changes accepted/offered drafts (106/113 to
113/118) and generated continuation, so it cannot be attributed solely to faster
verification. Short DFlash's output IDs match, but acceptance is 177/185 versus
179/188. Cold prefill: short 729.3 to 716.6 ms; longer 3240.8 to 3209.1 ms.

Upstream's Windows budget check trims the pool from 2523 slots/20.63 GiB to
2504/20.48 on the short run and 2502/20.46 on the longer run. RAM tier remains
approximately 94 GiB. Ordinary outputs diverge under these changed placements;
CPU/GPU execution is not assumed numerically interchangeable. A separate
old-capacity ablation is recorded below rather than disabling the new safeguard
in the retained configuration.

## Focused correctness and integration review

- `glm_memory_test`, `maya_promotion_owner_test`, `glm_window_graph_test`: all pass.
- Old versus merged diagnostic binary: **12,080,640 logits bitwise identical**,
  maximum absolute difference 0. Controlled GPU backend, 16 GiB pool, native mixed
  quant weights, 40-token prefix, two repetitions with partial window commits.
  Alternating widths 5/8 exercise graph-eligible T5 and the wider fallback path.
- Added full-residency state to the verification graph identity: upstream can
  remove transfer launches while fully resident, then a later prompt can lend
  expert slots. Replaying that graph after residency changes would be unsafe.
  This affects the full-residency opt-in case; our streamed model remains false.
- GPU diagnostic parity does not establish unrestricted hybrid backend parity,
  and these checks do not validate AMD or multiple GPUs.

Evidence: `docs/evidence/upstream-20261015/` contains manifests, output IDs,
round timings, device samples, build logs, unit-test log and comparison JSON.

## Capacity ablation and final build

Temporarily setting `STRATA_GLM_POOL_CHECK=0` restores 2523 GPU slots and 11365
RAM slots, matching the old build. First-request output matches the old baseline
exactly; repeated ordinary output still differs at token 37, while DFlash output
matches. Therefore capacity alone is not a complete causal explanation for all
hybrid differences. Controlled GPU logits above establish unchanged arithmetic in
that tested scope; adaptive hybrid residency and repeated prefill remain distinct.
No parity threshold was changed. The diagnostic-only pool-check override is not
retained in launch configuration.

This ablation measured 39.29 ordinary / 50.23 DFlash TPS. Since the ordinary sample
is lower, a targeted back-to-back repeat of the original short A/B follows rather
than claiming all differences are noise without additional evidence. STOP at eight
tokens and subsequent recovery passed, with clean process shutdown. Final rebuilt
binary includes the residency graph-key fix; all-resident remains false here.

## Back-to-back confirmation and verdict

Old then final merged binary, same short prompt/settings: ordinary **41.19 to
41.14 TPS**, DFlash **50.80 to 50.26 TPS**. Both report 2523 GPU slots in this
repeat; the new budget safeguard did not need to trim on this load. The earlier
warmup, ordinary and DFlash output IDs all match in this final A/B, and DFlash
acceptance is identical (177/185). The earlier
lower old-capacity diagnostic does not reproduce as a sustained ordinary slowdown.
These bounded samples support no material regression, not a universal guarantee.
No reliable speedup is established. The apparent longer DFlash gain has different
routing/acceptance/output and is not a clean kernel-speed attribution.

Retain latest upstream with our performance paths and upstream memory protection.
Daily launcher settings are unchanged; temporary diagnostic flags were process-local.
All test engines exited. No external publication.

Commits: `c7a7fc1` integration merge; `eb85fff` residency-aware verification graph
invalidation. Roll back source via `checkpoint/pre-upstream-20261015` and use the
preserved executable if necessary. Raw logit tensors remain local; compact evidence
and this report are committed separately.
