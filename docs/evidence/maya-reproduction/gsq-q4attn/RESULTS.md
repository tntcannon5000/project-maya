# GSQ-RCO 3.0bit Q4-attention cross-engine attempt

2026-10-12. No engine implementation changes. One stock Maya benchmark and
one DS4 load attempt, sequentially on the RTX 5090 / 9950X3D / 128 GB system.

Source: `G:/Models/GLM-5.3-Flash-3.0bit-Q4Kattn/GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf`
(113,585,695,168 bytes).

Maya stock `tools/iq_pack.py --compat-bf16` accepted the file: 1257 tensors,
306 served natively, 657 converted small tensors (0.32 GiB), arena 0.92 GiB.
Expert bytes unchanged; source GGUF untouched. Thus Maya runs its standard
BF16-compatible packing of the source, not literally every original dense dtype.
See `maya/compat-bf16.json` for conversion details.

Settings: c=1, 131072 context capacity, greedy, thinking off, no MTP,
one unscored 64-token warm-up and three tasks capped at 256 output tokens.
Standard Maya prefill warm-up followed by 2048/8192-token documentation prompts.
Stock cache/profile settings, 700 MiB reserve. No tuning loop.

| Metric | Maya GSQ-RCO Q4-attention | DS4 same file |
|---|---:|---:|
| CSV parser decode TPS | 28.2 | Load rejected |
| Photosynthesis decode TPS | 32.7 | Load rejected |
| HTML/JS animation decode TPS | 28.6 | Load rejected |
| Mean decode TPS | 29.8 | Unavailable |
| 2048-token prefill TPS | 1080 | Unavailable |
| 8192-token prefill TPS | 2280 | Unavailable |

Maya allocated 22.30 GiB GPU expert pool (2731 slots), 87.38 GiB pinned RAM
tier; no disk expert reads during measured decode. This is about 15% faster
decode than the previous stock Maya-M mean of 26.0 TPS, in one observation.

DS4 used the existing `maya-warm-benchmark.exe` with the environment from the
parent manifest, reserve overridden to 700 MiB, MTP disabled, and arguments:
`--cuda -m <source> --ssd-streaming --ssd-streaming-cold
--ssd-streaming-cache-experts 2800 --ctx 131072 --temp 0`.
It exited 1 before inference: `expected block_count=46 for GLM 5.3 Flash, got 45`.
The glm5-next loader has a fixed block-count check in `ds4.c`. This is the first
compatibility blocker, not proof that changing that check would make the rest
of the tensor layouts and quantizations compatible. No bypass attempted.

Previous warmed DS4 Antirez Q2: 30.67 TPS mean decode, 955.90 / 819.38 TPS
prefill at 2K / 8K. Those remain different-model references, not a controlled
engine comparison. Same-model comparison remains blocked. Both processes exited;
no inference engine was left running.
