# Profiling and analysis tools used on our Maya fork

This is the investigation toolkit, not a benchmark-suite guide. Paths below are
relative to the repository. Most scripts use the local `.venv` and some contain
machine/model-specific paths: inspect their arguments/configuration before reuse.

## Capture what actually executes

**NVIDIA Nsight Systems** is the main timeline profiler. Our harnesses capture
CUDA APIs, GPU kernels, copies, NVTX ranges and individual CUDA graph nodes where
configured. Captures are restricted with CUDA profiler start/stop calls and
exported from `.nsys-rep` to SQLite for programmatic analysis. Typical options:
`--trace=cuda,nvtx --sample=none --cpuctxsw=none`, with a profiler-API capture range.
`--show-output=true` preserves child-process diagnostics.

- `tools/profile_decode.py`: serving-oriented capture harness and SQLite export.
- `tools/run_transfer_lab.py`: controlled input driver/orchestration used by the
  transfer investigation; records configuration, binary/weight identities,
  per-token counters and memory information.
- `tools/trace_verify_windows.py`: captures a small set of warmed verification
  windows using a saved parity manifest.
- Engine NVTX instrumentation identifies tokens, layers, route/service work and
  selected CPU stages. `MAYA_CPU_STAGE_TRACE` adds detailed CPU-stage ranges;
  enable it only for diagnosis because instrumentation adds overhead.
- `nvidia-smi` sampling records device-wide memory, clocks, utilization, power
  and temperature. These are context for a run, not per-kernel attribution.

## Reconstruct the token timeline and draw the stage bars

| Tool | What it answers |
|---|---|
| `tools/analyze_decode_timeline.py` | Classifies kernels into fetch, gate/up, down, attention, dense projections, routing, waits and other stages; produces token tables and visual reports. |
| `tools/analyze_transfer_trace.py` | Reconciles fixed-input token spans, copy/computation overlap and uncovered gaps using interval unions. Produces aggregate and individual-token timelines. |
| `tools/analyze_deep_profile.py` | Adds per-layer readiness, mixed-quant component byte accounting and effective transfer throughput to the broad stage breakdown. |
| `tools/inspect_trace.py`, `tools/trace_copy_summary.py` | Inspect trace contents and copy activity when investigating unexpected categories or missing events. |

These scripts use Python, SQLite, NumPy and Matplotlib. The useful outputs include
CSV/JSON summaries, first-versus-repeated-request splits, and fast/median/slow
token plots. Examples live in `docs/evidence/maya-decode-latest/latest/` and
`docs/evidence/maya-transfer/trace-retained/`.

**Do not add overlapping stage durations to estimate token time.** Use interval
unions/intersections to separate compute-only, fetch-only, overlap, exposed waits
and periods with no recorded GPU activity. A GPU gap alone does not identify its
CPU cause. A CPU range's duration is not necessarily exposed token latency.

## Explain RAM-to-VRAM time and cache churn

- `src/core/glm_transport_probe.cu` and `tools/summarize_transport_probe.py`:
  replay actual expert-address transfers under controlled CPU/GPU concurrency.
  Compare copy-only bandwidth with concurrent inference-like contention and
  check transferred bytes. Graph-local GPU timer stamps handle internal events
  that were not measurable through the initial captured-event approach.
- `tools/transfer_transport.cu`: isolated transport feasibility comparisons.
  Useful for a ceiling; not proof that integrated DMA will improve decode.
- `tools/analyze_demand_routes.py`, `tools/replay_recent_residency.py`,
  `tools/glm_tier_replay.py`: inspect route reuse and replay placement/eviction
  alternatives. Separate causal policies from perfect-future oracle bounds.
- Engine counters track demand/predicted bytes, demotions, backing hits, promotions,
  GPU residency, CPU experts and disk reads. Counters plus completion events are
  needed to distinguish submitted traffic from traffic exposed on the token path.

Effective payload GB/s is bytes divided by the measured interval. It is **not** a
direct PCIe bus-utilization measurement. Mapped-RAM GPU fetch kernels combine
memory requests, execution and contention; their duration is not pure DMA time.
Replay may use changed same-format RAM contents and synthetic inputs, so it does
not independently establish model-output parity or whole-engine TPS.

## Inspect speculative verification costs

- `tools/analyze_verify_submission.py`: separates gaps inside verification from
  fixture readback/commit gaps; avoids blaming the engine for harness overhead.
- `tools/analyze_verify_budget.py`: computes optimistic savings bounds from
  observed rounds while retaining their recorded acceptance/history.
- `tools/analyze_window_union.py`: studies expert reuse across window positions,
  CPU/GPU assignments, ownership/readiness constraints and possible copy savings.
- `tools/analyze_scratch_retention.py`: investigates reuse across scratch lifetimes.
- DFlash round logs distinguish offered/accepted drafts, no-draft rounds, draft
  time and verification time. Acceptance percentage alone hides how often the
  confidence policy declines to draft at all.

## Check that faster execution is still correct

- `tools/check_verify_scratch.py`: fixed-input comparisons using selected binaries,
  repeated requests, partial window commits and saved float32 logits.
- `tools/check_window_forward.py`: compares ordinary forward with the first row
  of a verification window, including prefix state where configured.
- `tools/check_three_costs_routes.py`: checks recorded routing evidence; route
  masks alone must not be described as a comparison of full expert IDs.
- `tools/analyze_moe_products.py`, `tools/analyze_moe_combine_capture.py`,
  `tools/test_moe_combine_replay.py` and the MMQ/down replay helpers: localize
  differences to selected weights, inputs, intermediate products or reduction
  grouping rather than accepting a generic “floating point” explanation.
- CUDA Compute Sanitizer memcheck plus focused ownership/graph tests: investigate
  invalid access, slot lifetime, stream joins, graph invalidation and failure paths.

## Practical workflow

1. Save the binary/configuration, actual cache capacity, token inputs and model
   identity. Record occupied context separately from allocated capacity.
2. Capture one representative warm decode segment; separate cold/first/repeated
   requests. Reconcile the timeline before proposing an optimization.
3. Rank opportunities by removable **critical-path** time or avoidable bytes.
4. Use a targeted counter, replay or ablation to establish the proposed cause.
5. Check numerical results and ownership under controlled backends; then measure
   ordinary execution without the diagnostic profiler.
6. Preserve raw captures locally and commit compact findings, manifests and
   rollback checkpoints. Reject improvements that only make a microbenchmark
   faster, change cache capacity unnoticed, or move work outside the timed region.
