# Corrected hybrid component-streaming attribution

Measured 2026-10-15; same binary, RAM94GiB, requested GPU22GiB, 128K allocated context, 40 occupied prompt tokens, 128 output tokens, two draft tokens. Both runs use HC batching and paired expert kernels. These are phase-profiled diagnostics, not clean throughput qualification.

| Per verification round | Component off | Component on |
|---|---:|---:|
| Host verification elapsed, ms | 59.417 | 57.431 |
| GPU event timeline, ms | 58.56 | 56.60 |
| Fetch + expert stage span, ms | 34.50 | 33.02 |
| Exposed CPU wait, ms | 3.00 | 3.11 |
| Disk-service wait, ms | 0.82 | 0.98 |
| Profiled generated TPS | 45.93 | 47.50 |
| Ordinary control TPS | 44.50 | 43.40 |

Both runs completed 45 rounds and accepted 83/90 offered drafts. All corresponding warmup/control/speculative output token IDs match. This does not independently prove identical internal routes or numerical states.

Component-on expert sequence: prefix fetch17.15ms -> gate/up6.09ms (down transfers run concurrently) -> exposed down-readiness join6.56ms -> down/combine3.22ms. Total33.02ms. Off fetch26.00ms + experts8.50ms =34.50ms. The sum identifies a1.48ms stage-span reduction; it does not isolate bandwidth changes or arithmetic contention. The join is the remaining transfer tail, not the complete down-copy duration. Exact overlap requires stream intervals.

Host enqueue43.51/41.98ms overlaps device execution; adding it to GPU event times would double-count. Host final wait15.16/14.73ms similarly is not an additional independent GPU cost. The prior GPU-only T5/T8 capture's40.938ms overlap across four windows cannot be subtracted from these hybrid T3 rounds.

The clean contemporary comparison remains45.23 versus44.70TPS (1.2%), insufficient for a robust default change. Diagnostic pair suggests a small benefit, not the earlier apparent10-11% gain. No60TPS claim. The interrupted hybrid-phase-off run produced no completed requests and is excluded.

## Next decisions

- Collect corrected initial ownership and route metadata before implementing a broader union planner. Historical fixed-CPU-budget upper bounds6.825%/11.602% bytes are optimistic and predate the ownership fix.
- Test a standalone exact Q4_K integer tensor-core implementation against the actual existing multi-column kernel. Existing projections already reuse weights. Preserve original floating-point accumulation order; approximate MMA is not an acceptable substitute.
- Dense candidate stage envelope is8.38ms including shared work;1-3ms saving is a hypothesis, not a prediction. Neither avenue currently supports a near-doubling expectation.
- Keep component streaming opt-in. Broader lifecycle, longer occupied context and matched prefill qualification are required before recommending it.

## Distance to 60 TPS on this exact sample

128 emitted tokens over45 rounds means2.844 emitted tokens/round. At60TPS, the full generation allowance is47.407ms/round. The component-on sample uses2694.8/45=59.884ms/round: about12.48ms/round must still be removed, with the same emitted-token yield. Verification57.431ms plus draft2.054ms plus commit/injection~0.35ms explains nearly all of it. This is a sample-specific budget, not a universal ceiling; acceptance and routing vary by workload. Saving1-3ms in dense projections alone would not reach60TPS.
