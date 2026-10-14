# Verification campaign update

## Outcome

Retain the deferred-demotion correctness fix. New arithmetic and transfer
scheduling prototypes remain opt-in: none establishes a substantial speedup.
No daily launcher changes, publication or claim of 60 TPS.

| Contemporary coding sample | Ordinary TPS | Two-draft TPS | Verification ms/round |
|---|---:|---:|---:|
| HC/paired, component streaming off | 43.01 | 44.70 | 58.841 |
| HC/paired, component streaming on | 43.03 | 45.23 | 58.081 |

Identical output history and 87.14% draft acceptance. Difference is about1.2%,
not a robust gain from one matched pair. Earlier apparent10-11% gains were
mostly run variation. Four drafts measured43.84 TPS with streaming; no matched
contemporary width-four confirmation, so no claim of improvement there.

## Work completed

- Corrected deferred VRAM-to-RAM submission before batched verification. A
  missing submission let CPU service return stale RAM after an old event wait.
- Immutable pack audit checked14.56GB across1,599 expert copies in42MoE layers.
- HC/paired and component scheduling each matched28,188,160 logits exactly.
- Bounded disk-pressure component test matched6,040,320 logits and exercised
  304 token-owned disk staging uses.
- Component transfer trace proves40.94ms gate/up overlap across four GPU-only
  diagnostic windows. This is real overlap, not an equivalent time saving.
- Actual pressure traffic:22.707GB prefix and11.354GB down,3,627 source entries.
- Moderate CPU allocation/merged rows engaged but did not improve round time;
  reject as a recommended setting. KDA persistence deferred on small measured
  upper bound. No parameter sweep.
- Writing at3,239 occupied tokens completed with STOP/recovery and clean exit
  on the corrected HC/paired path. Ordinary39.98 TPS versus two-draft32.02 at
  54.1% acceptance illustrates workload dependence.

## Limits and remaining work

The component prototype has exact-logit and pressure evidence but no separate
component-specific cancellation/long-input qualification; it remains disabled.
No matched cold-prefill performance comparison or near-capacity context test in
this round. Widths/settings are short screens, not production benchmark claims.
The main remaining issue is complete verification-round cost, not draft
acceptance alone. A further large change requires a fresh critical-path case;
repeating weak sweeps is not justified by these results.

Long-term plan and checklist remain in docs/DFLASH-VERIFICATION-PLAN.md and
docs/DFLASH-VERIFICATION-CHECKLIST.md; detailed evidence is in WORK-LOG.md.
