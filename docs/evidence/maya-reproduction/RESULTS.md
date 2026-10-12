# Maya reproduction: results and verdict

Date: 2026-10-12. No engine, kernel, packer or server source changed.
Source: stock Maya v1.1.3 at `20e27a23491eb08506d1c291def17a30dc89e92d`.
Hardware: RTX 5090, OC Ryzen 9950X3D, 128 GB DDR5-6000 CL36; native Windows,
CUDA 13.2, driver 610.88. GPU power limit reported 600 W.

## Published result versus this machine

Reference: https://github.com/mw00/project-maya/issues/43 and attached report.
Reference engine v1.0.17, Linux, Ryzen 7950X, 96 GB DDR5-6400, RTX 5090.
Both use Maya-M (116.0 GB), one GPU, 131072 context capacity, greedy sampling,
thinking off, and a 256-token limit on each of the same three decode questions.
All three downloaded shards passed the setup's published SHA-256 checks.

| Measurement, TPS | Published | Local stock | Local report settings |
|---|---:|---:|---:|
| Python CSV parser | 31.4 | 25.0 | 25.0 |
| Photosynthesis explanation | 35.1 | 27.7 | 27.4 |
| HTML/JS bouncing ball | 31.9 | 25.5 | 25.3 |
| Mean decode | 32.8 | 26.0 | 25.9 |
| Prefill, 2048 tokens | 707 | 1141 | 960 |
| Prefill, 8192 tokens | 1982 | 2314 | 1805 |

Report settings: `STRATA_GLM_PCIE_SHARE=0.10`,
`STRATA_GLM_RAM_HEADROOM_GB=4`, `STRATA_GLM_RESERVE_MB=256`.
Also explicitly set `STRATA_GLM_NO_MTP=1`; stock single-GPU Maya already runs
without MTP. No DFlash, expert pruning/skipping or optional KDA graphs.

**The published decode result was not reproduced.** Local stock decode is about
21% lower. The local stock prefill observations exceed the published figures,
but the standard benchmark constructs its prefill corpus from repository docs,
so source revisions change its text. This is reproduction of the benchmark
recipe, not a byte-identical historical experiment. The report's learned expert
placement profile is also unavailable; current Maya uses its supplied profile.
Context capacity is not occupied decode context: these decode prompts are short.

## Calibration and diagnostic observations

One stock `maya.py --calibrate --no-start` completed. Its final confirmation kept
the automatic settings, reporting 27.8 TPS on its separate calibration prompts.
That figure must not replace the benchmark's three-task mean.

- Stock startup: 16 CPU threads; 0.171 ms per CPU expert, 0.157 ms PCIe estimate.
  CPU plan for 0..8 RAM experts: `0 0 1 1 2 2 3 3 4`.
- Stock measured decode: 47.6–61.0 CPU experts/token and 68.4–81.9 RAM fetches/token.
- Report-setting run used the published CPU plan `0 1 2 3 4 4 5 6 7`:
  131.5–146.1 CPU experts/token, 10.1–13.8 RAM fetches/token, 22.5–25.1 ms/token
  reported CPU-lane time. Decode still averaged 25.9 TPS.
- Published CPU startup estimate was 0.096 ms/expert, with roughly 18–22 ms/token
  CPU-lane time during the reported tests. Our CPU path is measurably slower in
  these logs despite stronger nominal hardware; the OS, engine revision and
  placement differ. This is a lead, not a proven explanation of the entire gap.
- No disk expert reads occurred in the local benchmark decode samples.
- Stock expert pool: 20.49 GiB, 2536 slots; pinned RAM tier: 85.86 GiB.
- Calibration's intermediate fewer-thread results regressed (12: 23.6, 11: 21.4,
  8: 17.5 TPS). It did not confirm a better configuration. No manual tuning loop
  or implementation followed.

## Comparison with our DS4 setup

Fresh tests used the retained DS4 fast build and its known-good environment,
with a 700 MiB expert-cache reserve to match stock Maya and 131072 context
capacity. Source and executable hashes/settings are in `manifest.json`.
DS4 used our existing antirez Q2 file (96,505,816,384 bytes); Maya used Maya-M.

| Workload, TPS | Our DS4 + antirez Q2 | Stock Maya + Maya-M |
|---|---:|---:|
| Python CSV parser | 30.22 | 25.0 |
| Photosynthesis explanation | 30.21 | 27.7 |
| HTML/JS bouncing ball | 30.33 | 25.5 |
| Mean decode | 30.25 | 26.0 |
| Prefill, 2048 tokens | 829.76 | 1141 |
| Prefill, 8192 tokens | 911.03 | 2314 |

The decode prompts were supplied to DS4 as the same literal rendered chat strings
used by Maya's standard benchmark. Greedy, no speculation, up to 256 outputs.
DS4 ran each task in a fresh process with a cold dynamic expert cache; Maya's
standard test retains its cache and first generates a warm-up answer. Neither
table establishes model quality or token-for-token output parity.

Prefill used the same current documentation corpus and matched token counts,
but DS4 measured cold prefill in separate processes; Maya's standard test warms
prefill first and uses an offset of 997 tokens for its 8K sample, whereas DS4's
8K sample starts at the corpus opening. Timing and cache-state differences are
explicit limitations. Do not quote these ratios as isolated engine speedups.

Historical DS4 16K-occupied teacher-forced evidence remains 917.30 prefill TPS
and 25.44 steady decode TPS. It is not directly comparable to these short-chat
decode numbers; the fresh DS4 short-task measurements above are the relevant
practical comparison here.

## Same-model attempt

The existing stock packer was invoked with our exact DS4 file and
`--compat-bf16`, writing only to a new `pack-maya` folder. It failed closed:

> blk.0.kda_q.weight is Q4_K: not on the BF16 upgrade list nor the glm5-next
> native set (GLM5NEXT_NATIVE_OK); refusing to pack it silently

The source model uses DS4 tensor names/layout; Maya accepts `attn_q.weight`
on that path and has no stock alias for this tensor. No original GGUF was
modified, no weight conversion workaround attempted, and no compatibility code
implemented. The partial pack is not usable. Same-model throughput therefore
remains **unmeasured**, rather than zero or a failed performance result.

## Verdict

Maya is not demonstrated to be faster overall than our engine. Our existing DS4
Q2 setup won these decode tasks; Maya-M led the prefill observations and ran a
larger, different quant. Its published 32.8 TPS decode result remains unreproduced
on this Windows setup. Adoption solely for decode speed is not justified by this
test. Maya remains interesting for prefill and its mature GLM infrastructure,
but an engine-only verdict requires the same model, matched cache state and
matched timing boundaries. Engine work remains deferred per the user's request.

## Artifacts

- `stock/`: original benchmark, configuration and engine log.
- `report-settings/`: second benchmark with the report's environment overrides.
- `calibration.json` and `maya-maya-m.log`: completed stock calibration.
- `ds4-short-tasks/`: fresh DS4 outputs, startup/transfer logs and prefill CSVs.
- `manifest.json`: source/binary identification and environment settings.

All benchmark model processes were allowed to exit; no server was launched.
