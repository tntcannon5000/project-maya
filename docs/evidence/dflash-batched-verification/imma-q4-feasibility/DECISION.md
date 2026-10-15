# Exact Q4_K tensor cores: defer integration

The complete isolated prototype preserves the existing integer subtotals and FP accumulation order. All240 synthetic cases pass against both actual multi-column and singleton engine references; six additional actual-model shape cases also pass exactly, including output sentinels. No target arithmetic, production dispatch or numerical thresholds were changed.

| Model shape N x K; T5 | Calls/round | Existing us, two samples | Trial us, two samples |
|---|---:|---:|---:|
|4096 x8192|34|30.042 /30.148|28.356 /28.401|
|4096 x12288|3|41.565 /46.285|44.157 /45.325|
|4096 x16384|11|53.671 /58.919|48.444 /56.315|

Two-sample mean differences, multiplied by occurrence counts, total about0.098ms/round. This is an isolated estimate, not an observed whole-model speedup, and includes a losing/variable dense-down case. Even favorable endpoint selection is below0.2ms/round. T3 was substantially slower in the preceding screen. Existing weight-reusing DP4A is already competitive; preserving exact FP reduction order consumes much of the apparent tensor-core opportunity.

Decision: retain reproducible standalone experiment and negative result; do not integrate a target dispatcher or broaden model benchmarks for this candidate. Actual shapes matter: first screen used N8192, whereas the model's large-K dense matrices use N4096. No60TPS claim; this cannot materially close the12.48ms/round sample-specific gap.
