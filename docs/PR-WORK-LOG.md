# PR work log: reduce expert transfers and decode scheduling overhead

- Target upstream release: Maya v1.1.7 (`1e225b9`), latest published release checked on 14 October 2026.
- Integration branch: `integration/maya-v1.1.7-retained`; merge checkpoint `736ac46`.
- Preserve the measured retained baseline at `80d4abb` and the unfinished concurrent-graph investigation at `6be3dc8` on `research/maya-c4-graphs`.
- Do not include the concurrent MoE graph experiment in this integration: its numerical qualification remains unresolved.

## Proposed PR title

**Improve decode throughput with RAM-backed experts and lower-overhead scheduling**

## Retained improvements

- Keep bounded RAM backing for GPU-resident immutable experts, avoiding unnecessary GPU-to-RAM copies when those experts are evicted. Respect physical/commit headroom and pinning limits; retain ownership checks and release rules when prefill borrows the GPU pool.
- Defer remaining expert demotions until useful work can run, while retaining copy-completion events and preventing premature slot reuse.
- Add an optional causal expert-reuse eviction policy, combining observed transitions with usage frequency. This changes placement, never the required expert set; it is not an oracle or speculative token decoding.
- Capture eligible single-sequence MoE work in reusable graphs to reduce repeated host submission. Keep dynamic attention outside the captured suffix and preserve graph/workspace lifetimes.
- Overlap down-weight fetches with gate/up computation on eligible captured layers, using explicit stream dependencies and row-aligned down halves.
- Publish CPU work earlier, pipeline ready experts through CPU gate/up and down stages, and carry readiness as immutable per-job state rather than shared mutable context.
- Add optional CPU polling pauses and a worker admission limit during demand fetches to reduce CPU/RAM contention.
- Add an optional warp-cooperative routing/publication path while preserving routing order and arithmetic semantics.
- Preserve fixed-input replay, transfer accounting, ownership verification, and token/layer timelines so performance claims can be traced to actual behavior.
- Merge upstream fixes and features unchanged where possible, including reset-stream ordering, Windows power-throttling opt-out, improved RAM pin-cap allocation, model-file guards, exact multi-token CPU rows, and INT8 prompt-cache conversion.
- Keep the local 192K benchmark target behind the existing diagnostic build option; ordinary/CPU-only builds should not acquire this target.

## Hardware-sensitive changes: keep optional, do not copy our preset as defaults

- **RAM backing and expanded tier:** our preset requests 24 GiB of backing within a 101 GiB RAM budget. Duplicating weights can crowd out unique experts on lower-RAM machines, increase disk traffic, or hit OS pin/commit limits. Existing headroom checks reduce risk but do not establish a benefit on those machines.
- **CPU/GPU split `011223344`:** tuned with this model and a 9950X3D/RTX 5090. Slower CPUs, faster GPUs, NUMA placement and different quant formats can change the best split. Preserve upstream calibration/defaults for other users.
- **Eight CPU callbacks during fetches:** the `MAYA_CPU_PHASE_LIMIT=8` choice trades CPU completion against memory bandwidth. It may underuse larger CPUs or contend on smaller ones; it does not pin work to a CCD or identify physical cores.
- **32-block copies and two down bands:** depend on GPU SM count, PCIe/host-memory bandwidth, transfer/kernel contention and quant geometry. More overlap is not automatically faster. The overlap path rejects EXL3, incompatible alignment and enabled expert prefetch; our preset is not valid for every upstream mode.
- **Captured-only overlap and suffix graphs:** qualified for our single-device ordinary decode configuration. Do not extend claims to MTP/DFlash, concurrent graphs, AMD or multiple GPUs without qualification.
- **Transition eviction:** workload-dependent; can change CPU/GPU assignment and hence hybrid floating-point results. Short repeated coding prompts do not establish universal gains across task transitions or model families.
- **Warp routing:** measured on NVIDIA; AMD wave behavior and compilation/runtime portability remain unqualified.
- **Polling/CPU pipeline:** PAUSE behavior and worker scheduling vary by CPU/OS. The pause opt-in specifically affects Windows x86; the existing non-Windows behavior remains in place.
- **Local CUDA binary:** built for SM120 on this RTX 5090. This executable is not a portability test or a generic binary for older GPUs; other devices need their own supported build.
- **Fixed 22 GiB GPU pool, expert profile, 192K capacity and local paths:** benchmark configuration, not product defaults. Smaller cards and foreground workloads need capacity/headroom planning.
- **Diagnostic overhead:** counters and dormant experimental branches remain in the research source; opt-in does not mean zero code-size or dispatch overhead. Validate an all-flags-off build before claiming stock-equivalent default performance.

## Exclude from a clean upstream patch

- Concurrent MoE graphs and their failed parity evidence: preserved on the separate research branch.
- Rejected research paths still present behind disabled flags: paired Q2 AVX512 rows, coarse single-expert CPU execution, four-load copy variants, gate/up bands, resident-GU splitting, Q4 paired rows, split HC, and 16-route residency. These are rollback/research artifacts, not retained improvements.
- Machine-specific launch configs and absolute paths as defaults; retain a portable example only if useful to reviewers.
- Bulk raw benchmark logs, logits, hardware telemetry, generated images and old research dossiers. Preserve them locally; include a concise results table and reproduction instructions in the eventual PR.
- Do not claim broad AMD, multi-GPU, EXL3, speculative-decoding or multimodal validation from this CUDA/native-weight campaign.

## Validation

- Build and focused CPU tests: `native_split_test` and `native_rows_exact_test` passed after the merge.
- Short serial-prefill GPU-only decode: all 39,649,280 before/after logits are exactly equal, including the repeated request; live ownership and sampled immutable backing bytes pass verification.
- Long batched-prefill GPU-only control is not bit-stable across repeats in the old binary, merged binary or stock v1.1.7. Turning fused SwiGLU off does not resolve it. Report this as unresolved exact long-prefill qualification, not a parity pass or proven merge regression. All 256 greedy choices matched between stock and merged in this diagnostic.
- Hybrid results differ when residency/backend assignments change; the short fixture matches exactly until its first changed expert-assignment counters. Do not claim exact hybrid parity.
- Cancellation after eight tokens, recovery generation, repeated requests and process teardown passed on the merged server.
- Normal-budget short warm decode is 1.6-1.8% lower in two pairs; repeated 8K is flat; repeated c=4 is 41.47 -> 43.04 TPS (+3.8%). At approximately matched RAM capacity, short warm decode is 31.37 -> 32.23 TPS (+2.8%). These are bounded paired results, not a universal zero-regression claim or additive gains.
- Fresh before/after inference tables: `evidence/maya-release-integration/RESULTS.md`.
- Historical stock-vs-retained numbers are separate from this merge's incremental effects; do not add gains from individual experiments.
- No PR opened or external content published. This document is a PR description draft and scope audit; the research branch still requires removal of rejected prototypes and bulk evidence before publishing a clean patch.
