# Campaign requirement audit — 2026-10-15

Objective: develop and validate efficient multi-token verification, profile costs and
reuse, implement complete grouped computation/supporting scheduling where justified,
verify correctness/end-to-end throughput, and preserve planning/evidence/rollback.

This audit separates the completed bounded R&D deliverable from production release
claims. It does not turn a failed historical test into a pass or treat60TPSas measured.

| Requirement | Finding | Evidence |
|---|---|---|
| Profile costs and actual reuse before choosing work | Complete | Phase attribution, group/union diagnostics, CUDA API gap correlation, remaining-budget.json; detailed decisions in plan/work log |
| Complete grouped native computation | Complete for target's Q2/Q3/Q4 path | GU/activation/down/combine implementation, isolated tests, corrected exact full-model evidence |
| Supporting scheduling and combinations | Complete bounded evaluation | Matched async, component readiness/overlap, HC, exact columns, native T2..T5 graphs; meaningful rejected CPU/IMMA/direct-RAM/placement/retention experiments |
| Prove engagement | Complete in measured scopes | Transfer counters and Nsight intervals; graph capture/replay/bypass records; exact replay tools |
| Correctness qualification | Current controlled qualification passes | GPU and matched-hybrid logits/prefix data, partial commits/resets, immutable audit, pressure checks,24metadata+4DMAcases |
| Numerical differences investigated | Sampled differences explained; historical limit preserved | Causal combine/MMQ dispatch replay; hybrid backend-placement investigation; original graph failure still unattributed |
| End-to-end coding/writing and longer occupied context | Complete bounded samples | Serving/confidence/writing summaries and matched manifests; current-build short smoke |
| Prefill/lifecycle | Complete bounded checks | Cold prefill smoke; STOP/recovery, helper rollback and clean teardown |
| Memory/capacity accounting | Reported with limitation | Same2,523expert slots; staging/RAM configuration; device-wide samples. Incremental graph bytes not isolated |
| Planning/checklist/report/checkpoints | Complete | DFLASH-VERIFICATION-PLAN/CHECKLIST/RESULTS, chronological work log, compact evidence, rollback commits |
|60TPSaspiration | Not achieved | Best current short smoke57.76TPS; longer/writing~40TPS. No success claim |
| Production/default/PR sign-off | Not given | Historical numerical attribution and hardware/generalisation limits remain explicit |

Read-only independent source/evidence review found no additional missing bounded
experimental correctness, engagement or lifecycle gate. It identified stale checklist
items already covered and the graph-memory reporting limitation; both are now reflected
in the consolidated report. No full suite or benchmark matrix was repeated. The final
current build ran one short workload with cancellation/recovery, and exact output/count
comparisons match the previous recorded request schedule.

All changes stay local/opt-in. The original historical graph mismatch cannot be
retroactively attributed without its missing starting state; current passing checks
do not prove that old bug fixed. The production-specific unchecked items remain open
on the checklist. They do not justify silently claiming production readiness, nor do
they require unbounded repeated diagnostic runs to report the experimental outcome.
