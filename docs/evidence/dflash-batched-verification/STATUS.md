# Verification campaign status

Research branch: `research/dflash-batched-verification`. Daily configuration unchanged. No external publication.

## What is established

- Expanded scratch/reuse remains the starting checkpoint; this campaign does not claim those previous gains.
- The first grouped expert implementation and paired refinement passed 28,188,160 exact logits in their initial controlled tests. Clean coding differences were small.
- Matched-staging asynchronous fetch passed 28,188,160 exact logits with all experts warmed. Nsight proves fetch/route overlap, but clean hybrid verification-round times barely improved.
- Major dense projections already batch weights across token columns. A wholesale missing-GEMM explanation is unsupported.
- Narrow CPU/GPU assignment swaps offer only 2.1%/3.5% optimistic byte reductions at T3/T5; below 1.2% for the transient subset. That planner is deferred.

## Clean coding samples

These are short 192-output-token samples with 40 occupied prompt tokens and 128K capacity, not long-context evidence.

| Configuration | Control TPS | Two drafts TPS | Four drafts TPS | Interpretation |
|---|---:|---:|---:|---|
| First grouped-off reference | 41.93 | 41.91 | 35.09 | Original campaign RAM configuration |
| First grouped kernel | 41.73 | 41.76 | 35.24 | Same output history; no material gain |
| Paired grouped kernel | 42.04 | 42.26 | 35.79 | Same output history; small cross-run difference |
| Matched 94GiB RAM, staging, serial | 41.12 | 39.80 | 36.08 | Separate matched-capacity comparison |
| Matched 94GiB RAM, staging, async | 41.09 | 40.65 | 32.75 | Acceptance/history changed; no reliable scheduling gain |

Matched serial/async verify-round means were 67.262/66.885 ms at two drafts and 105.430/105.257 ms at four. Do not interpret their TPS difference as pure scheduling savings.

### After deferred-demotion correctness fix

The earlier rows predate the fix and are historical, not current controls.

| Configuration | Control TPS | Two drafts TPS | Four drafts TPS | Verify round ms (two/four) |
|---|---:|---:|---:|---:|
| Corrected control | 33.80 | 39.73 | 38.19 | 66.340 / 101.709 |
| Corrected HC + paired experts | 35.48 | 40.96 | 39.11 | 64.395 / 99.192 |

Output histories and acceptance match. Small 2.5-3% verification changes are
not a robust gain given ordinary control variation. Both remain experimental.
Bounded pressure qualification now also passes 6,040,320 exact logits against
the corrected all-memory reference (six windows, RAM64GiB/GPU16GiB).

## Current correctness gate

HC phase batching plus paired experts now passes the corrected GPU-only control exactly: 28,188,160 logits, 182 rows, partial commits and repeated requests. Earlier failures were traced to missing deferred-demotion submission. The old control also changes after the fix (max logit difference 6.4494, all greedy choices unchanged), so it cannot remain the correctness reference.

Isolated HC arithmetic passed all 128 cases/576 rows exactly. The immutable-pack audit localized the first combined mismatch to stale RAM in a pending demotion. GPU bytes exactly matched RAM; scratch addresses were correct. Batched window execution omitted submission of boundary-deferred demotions, so CPU service could wait on an old/unrecorded event and serve stale bytes. Added submission before batched layer routing. The corrected bounded audit passed all 42 MoE scopes: 1,599 expert/pointer pairs and 14,556,266,496 bytes match immutable pack. Clean corrected-baseline performance comparison is next; no HC speed claim yet.

Separately, serial disk-pressure tests were not reproducible. Four ownership-check boundaries passed published-pointer checks and five retained-copy byte comparisons, but do not resolve that issue. A background-transfer API-error handling fix is checkpointed; no observed API failure establishes it as the cause.

## Decisions

- Keep performance experiments opt-in; no new default speed improvement is claimed.
- Preserve negative results and raw logits locally; commit only compact evidence.
- Qualify combinations only after their isolated correctness gates. Avoid another quant/width sweep.
- After a justified candidate exists, check writing, longer occupied context, prefill and cancellation/recovery.

Detailed chronology: `WORK-LOG.md`. Long-term plan: `docs/DFLASH-VERIFICATION-PLAN.md`. Checklist: `docs/DFLASH-VERIFICATION-CHECKLIST.md`.
