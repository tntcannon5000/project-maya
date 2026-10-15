# Scratch-retention feasibility: defer serving implementation

2026-10-15. CPU-only replay of the corrected `union-diagnostic/routes.cuda0.csv`.
No inference, builds or GPU work. Reproduce:

```powershell
python tools/analyze_scratch_retention.py docs/evidence/dflash-batched-verification/union-diagnostic/routes.cuda0.csv --out docs/evidence/dflash-batched-verification/SCRATCH-RETENTION-SCREEN.json
```

**Decision:** no demonstrated causal mechanism currently delivers the requested
5ms/round. Defer a live scratch cache until a predictor demonstrates conservative
whole-round headroom. This distinct retention opportunity is larger than narrow
assignment, but ordinary recency performs poorly and a future oracle is not an
implementation. Do not retry mapped arithmetic, IMMA or CPU quota sweeps.

## Why verification still fetches substantial bytes

Within one layer/window, uploads already equal unique fetched-ID bytes. GPU
pointer reuse shares that upload across all later GPU members. Repeated paths
are predominantly resident; there is no large missing same-window deduplication.
Transient scratch is reused by the next layer, however, so a later window can
fetch the same never-admitted expert again. The older resident-loss/return churn
audit does not bound that category: its loss requires earlier residency.

T3/T5 transient uploads average288/772MB per window. Their maxima are488MB and
1,082MB, so even the latter exceeds the64×14,155,776-byte scratch allocation.
Retaining that entire allocation and simultaneously claiming another complete
workspace is invalid. The screen models a shared64-slot allocation, with each
slot conservatively charged the maximum blob size regardless of quant type.

## Existing-trace results

Percentages are saved H2D bytes divided by all baseline fetched bytes. All
ready-RAM GPU fetches, including later promotions, can reuse an earlier
transient upload. CPU assignments and permanent promotion decisions stay fixed.

| Retention model | T3 | T5 |
|---|---:|---:|
| Fixed tail, LRU (40 / 24 retained slots;24 / 40 workspace slots reserved) | 5.284% | 0.000% |
| Fixed tail, farthest next use | 15.875% | 12.628% |
| Unified64 slots, LRU with live-consumer protection | 10.404% | 3.211% |
| Unified64 slots, farthest next use | 18.363% | 21.349% |
| Weighted future interval bound, fixed tail | 16.152% | 13.241% |
| Weighted future interval bound, unified64 incl. transient workspace | 18.465% | 21.739% |
| Unlimited retention repeated-key ceiling | 19.317% | 28.966% |

Farthest next use is an optimistic policy simulation, not a weighted-byte
optimum: quant blobs have different byte costs. The already-computed weighted
interval solver maximizes saved bytes for its stated model, reserves the
current layer's transient workspace and checks slot occupancy. Its promotion
source lease ends ideally at layer entry, and lookup/admission/D2D are free;
it is an optimistic interval-class bound, not a universal cache/latency theorem.

Unified LRU keeps total active+retained slots at most64; maximum live
cache/temporary consumers are7/13. The fixed-tail model requires D2D admissions:
its LRU run adds6.249GB/12.935GB of cache-admission copies over the trace,
before promotion copies. Those costs are excluded from the byte-saving metric.
Promotion hits still need a GPU-to-GPU copy into the unchanged permanent
destination. The unbounded retain-all simulation peaks at278/581 slots; these
are not minimum capacities for attaining its ceiling and are infeasible within
the current64-slot budget.

## Assumptions and complete-mechanism requirements

- A key is `(layer, expert ID)`. Sources must be initially ready in RAM;
  pending/disk-staging sources are excluded. Promotion or observed initial
  residency retires its duplicate retained key. Actual recorded initial tiers
  and CPU/promotion masks remain authoritative.
- Replay assumes no unrecorded inference overwrites retained scratch between
  captured windows. Main-cache changes at boundaries are represented by the
  recorded tiers/promotions. This is one short coding history, not evidence on
  writing, longer context or changed cache states.
- Lookup must happen after legacy CPU placement and promotion disposition;
  putting cached keys in the ordinary resident table would alter both. A
  transient hit must publish its pointer before later within-window reuse;
  promotion hits retain the permanent destination and use an explicit D2D
  source. All current-layer consumers remain pinned through computation/copy.
- A fixed tail preserves existing row workspace but requires admission copies.
  Unified leasing requires a bounded device allocator, cache identity and live
  pinning; it must handle ordinary/fallback scratch use and cancellation before
  releasing slots. No additional GPU allocation is assumed in the screen.

The reproducible script includes the simulations actually used, source hashing,
complete-row/unique-owner checks, live-slot assertions and weighted workspace
reservation checks. No new toy-fixture suite or serving implementation was
added in this bounded screen.

The weighted upper bound leaves a possible wider-window opportunity, but the
causal LRU result does not approach it. There is no measured predictor or
complete mechanism supporting a5ms/round improvement. Retention remains a
future measured-policy gate, not a justified next kernel implementation.
