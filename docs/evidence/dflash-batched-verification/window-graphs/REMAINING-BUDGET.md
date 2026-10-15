# Remaining verification budget — 2026-10-15

Source: existing complete-graph confidence-policy requests, not a new throughput run.
`tools/analyze_verify_budget.py` preserves source hashes and per-width timing totals in
`remaining-budget.json`. No new inference suite was run for this analysis.

| Recorded sample | Short coding | Longer coding |
|---|---:|---:|
| Occupied prompt tokens | 40 | 6,793 |
| Output tokens | 128 | 128 |
| Measured TPS | 56.78 | 39.64 |
| Decode wall time (ms) | 2,254.5 | 3,229.4 |
| Verification/token-forward time (ms) | 2,171.1 | 2,988.3 |
| Drafter time (ms) | 69.3 | 195.9 |
| Other elapsed time (ms) | 14.0 | 45.2 |
| Saving required for 60 TPS (ms) | 121.2 | 1,096.1 |
| Required saving from all verification/token-forward time | 5.6% | 36.7% |
| Required saving if only multi-token verification changes | 5.8% | 65.1% |
| Optimistic TPS if zero-draft attempts cost nothing | 56.92 | 40.93 |
| Optimistic TPS if all drafting costs nothing | 58.58 | 42.19 |

These counterfactuals subtract time while holding recorded histories and other costs
fixed. They are optimistic budgets, **not measured speed predictions**. Graph capture,
cache/routing changes, workload variation and altered acceptance are not simulated.
The recorded token cap can stop before every accepted token is emitted; use the real
128-token wall-time denominator, not the sum of accepted-plus-one per round.

## Source findings and prioritisation

`decode_dflash` already sends zero-draft rounds through ordinary `forward({y})`.
There is no redundant one-column window to eliminate. In the longer sample, 48
zero-draft rounds consume 1,304.5 ms of target forward time and 101.8 ms of drafting.
Avoiding unsuccessful draft attempts alone cannot materially close the target gap.

Short coding remains a plausible incremental performance target; long coding would
need substantially cheaper verification, more useful drafts, or faster ordinary decode.
Do not infer that high conditional acceptance alone provides a near-2x speedup. Do not
implement another generic drafter micro-optimisation or unconditional wider window on
the basis of these figures. Prior grouped, transfer, placement and graph findings still
apply; this analysis does not credit them twice.

## Ownership qualification update

The promotion test now includes four actual asynchronous 1 MiB upload cases alongside
24 production metadata cases. A bounded CUDA host callback holds the old upload queue;
an event query proves it is pending while RAM and destination reservations remain held.
A CPU reader checks the immutable backing during that interval. After release, the
completed upload is byte-checked before invoking the production retirement method.

Cases cover retained old backing, ready replacement backing, pending replacement
backing, and an evicted/reused newer GPU winner. Old source and losing destination are
then overwritten after retirement; the winner/reused winner slot retains its exact
expected bytes. All four pass. Callback timeout is a failure, not a passing fallback.
The fixture drains streams before teardown and allocates only 3 MiB each pinned RAM
and device memory.

Scope: this tests real DMA plus the exact production metadata transition under its
caller contracts. It does **not** run the full boundary/service scheduler, prove arbitrary
thread interleavings, or retrospectively attribute the historical numerical failure.
No inference arithmetic, default flags or parity thresholds changed.
