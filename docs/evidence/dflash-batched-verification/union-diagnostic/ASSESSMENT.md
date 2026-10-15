# Corrected window-union placement diagnostic

2026-10-15. Source: `routes.cuda0.csv`, validated by
`tools/analyze_window_union.py`; results in `analysis.json`. Binary/environment
and request history: `Q4_K_M/manifest.json`, `requests.json`, `engine.log`.
This run includes the deferred-demotion fix and supersedes the older proxy
counts for this sample. It uses one 40-token coding prompt, 128K capacity,
64-output cap, hybrid CPU plan 011223344, full-layer windows and GPU-pointer
reuse. Synchronization/file output deliberately perturb timing.

**Decision: defer a live RAM-ready union planner under the existing CPU-row
budget.** Even an oracle with free token-to-token budget movement, no planner
overhead or admission restrictions can remove only 4.442%/10.255% of eligible
copy bytes. There is no measured whole-round gain, and CPU/GPU placement costs
remain additional constraints. This is a measured feasibility rejection of
that candidate as the next substantial speed mechanism, not a universal
rejection of placement with different CPU budgets or resident-on-CPU policy.

| Metric | T3 / two drafts | T5 / four drafts |
|---|---:|---:|
| Windows / layer batches | 24 / 1,008 | 17 / 714 |
| Initially RAM-ready eligible layers | 989 | 684 |
| All copied bytes | 19,860,946,944 | 24,438,374,400 |
| Eligible copied bytes | 19,176,947,712 | 22,932,357,120 |
| Optimistic removable bytes | 851,902,464 | 2,351,824,896 |
| Ceiling / eligible copied bytes | 4.442% | 10.255% |
| Ceiling / all copied bytes, fallback unchanged | 4.289% | 9.623% |
| Eligible layers with any optimistic saving | 85 | 206 |
| CPU-only repeated groups / rows | 61 / 124 | 148 / 302 |
| Actual repeated GPU groups resident / total | 4,184 / 4,359 | 5,147 / 5,446 |

The last row is 95.99%/94.51% resident: most currently exposed GPU reuse does
not involve a transfer. Only 175/299 fetched groups contain multiple GPU
rows. CPU-only repeated rows are 0.51%/1.06% of all routed rows. The trace
validates one physical GPU destination and at most one copy owner per logical
expert. It does not reveal a large duplicated-transfer workload.

## What the byte oracle means

Within each layer, hold initial residents on GPU, retain every selected expert
and allow no more CPU rows than observed. Each nonresident expert put entirely
on CPU costs its logical multiplicity in CPU rows and avoids one packed blob.
All blobs in a layer have the same size. Sorting multiplicities therefore
maximizes the number of all-CPU experts under this relaxed budget. The saving
is `(oracle_CPU_only_groups - baseline_CPU_only_groups) * blob_bytes`.

This is stronger than the earlier aggregate C-U bound because it uses actual
per-layer multiplicities. It also allows more freedom than the proposed
bounded live planner: no per-token quota, no two-move limit, no extra-copy cap,
no planning cost, no CPU/GPU cost curve or shared-bandwidth contention, and no
future promotion effect. Its result is a copy ceiling, not an achievable time
saving. The whole selected route/cache state is held fixed.

For scale, the separate corrected T3 phase sample in
`../hybrid-phase-summary.json` reports a 26.00ms fetch interval and
59.417ms verification. Applying the all-copy fraction proportionally gives
about 1.12ms, or 1.9% of verification, before overhead. That is an illustration,
not a time bound: layers, bytes, launches and overlap differ. No contemporary
matched T5 phase decomposition establishes a T5 makespan benefit. A 10.255%
copy ceiling must not be presented as a 10.255% verification gain.

## Miss-mask interpretation and pending demotions

`miss` is the device's request for a **host response because no source is
published in the device RAM table**. It does not count storage reads.
`glm_fast_path.cu:3999-4027` handles kRDemote by waiting for its D2H event and
returning that RAM slot without reading disk. The same host-response path can
also use a lookahead RAM slot. Only its later read branch performs storage I/O.

All 19 T3 miss rows and all 31 T5 miss rows have initial GPU=0, initial
published RAM=0, RAM state kRDemote, and an unretired demotion. They occupy
19/30 layers; those layers correctly fail the initial-ready-RAM gate. Their
`ram_owned=0` means no published RAM pointer was validated, not that the
pending physical slot is unowned. The CSV does not separately export the
pending slot address or an actual-storage-read flag. The engine's per-request
statistics report 0.00 disk reads/token, consistent with the demotion service
path. Do not label these rows or layers as proven disk reads.

The analyzer now reports `host_response_required`, replacing its overly strong
`disk_miss` label. Eligibility and byte ceilings are unchanged. As a generous
counterfactual, allowing the pending layers' sources to be immediately usable
raises the whole-trace copy ceiling only to 4.372%/9.817%; it does not uncover a
large excluded opportunity or establish legal early CPU access.

## Gate and validation limits

No live routing policy or kernel arithmetic changed. The analyzer passes its
metadata checks and four CPU-only tests, including exhaustive subset checks of
the cardinality optimum on 25 generated traces. Warmup/control/T3 output IDs
match the corresponding first 64 tokens of `hybrid-phase-off-resume`; this is
useful engagement/history evidence, not full-logit parity. No diagnostic TPS
or phase interval is accepted as clean performance evidence.

Do not build a planner or sweep quotas on these counts alone. Reopen union
placement only with a concrete additional opportunity outside this bound and
a measured conservative whole-round case above the existing 5% gate, including
CPU start/tail and shared bandwidth. Existing moderate-CPU allocation already
failed that speed gate; changing the CPU budget remains an unproven different
experiment, not a consequence of these copy savings.
