# Warm verification submission-gap diagnostic

Four fixed-input windows (T=3/5/3/5), after 6,793 occupied prompt tokens, with the retained opt-in combination and hybrid CPU plan 011223344. These are Nsight diagnostic timings, not unprofiled throughput. Window boundaries exclude fixture readback, file writes and partial commits between calls.

| Window width | Span ms | Traced activity union ms | No traced activity ms | MoE-region gaps ms | CPU-answer wait kernels ms |
|---|---:|---:|---:|---:|---:|
| 3 | 62.960 | 56.980 | 5.980 | 3.329 | 11.179 |
| 5 | 95.157 | 87.890 | 7.267 | 4.037 | 7.492 |
| 3 | 66.928 | 61.247 | 5.681 | 3.249 | 4.984 |
| 5 | 95.883 | 89.192 | 6.692 | 3.847 | 8.004 |

All 7,130 gap-ending activities correlate to a CUDA runtime API, and every associated API had returned before its gap began. Missing correlations are explicitly counted; none occur for these gaps. The earlier approximately 42 ms host enqueue measurement therefore cannot be counted as removable token latency. These gaps do not demonstrate host submission starvation. CPU wait kernels are included in busy time and must not be labeled useful arithmetic.

Per-layer CUDA graphs may reduce device dispatch/dependency overhead. Even eliminating every measured MoE gap would save only about 4–5% of these window spans; actual savings may be smaller, and Nsight changes overhead. The bounded next experiment captures the complete native MoE stage, including its side-stream join, while preserving live routing and CPU service credits. It excludes attention and positional DFlash staging. It must prove replay engagement and exact same-backend logits before one targeted throughput comparison. No performance default changes are justified by this trace alone.

The database and raw logits remain local. `submission.json` records the database SHA-256, interval statistics and largest gaps; `manifest.json` records configuration and binary hash. Three synthetic analyzer tests verify complete correlation, missing-data reporting and late-submission accounting.
