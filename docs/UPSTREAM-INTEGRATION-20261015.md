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
- [ ] Measure both baseline workloads.
- [ ] Build integrated source and resolve compiler/semantic issues.
- [ ] Repeat matching workloads; compare outputs, prefill, acceptance and expert capacity.
- [ ] Run focused validation of merge-sensitive code.
- [ ] Investigate regressions, checkpoint and summarize.
