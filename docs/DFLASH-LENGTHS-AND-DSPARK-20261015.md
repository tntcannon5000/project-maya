# DFlash2 length screen and DSpark compatibility

Same merged engine and retained flags, GSQ-RCO 3.0bit Q4-attention target,
Q4_K_M DFlash2, c=1, 128K allocated capacity, 6793-token coding prompt,
256 generated tokens per request. One warmup, then ordinary control and
speculative requests. No inference implementation or daily settings changed.

## Forced lengths

| Drafts forced per round | TPS | Accepted/offered |
|---:|---:|---:|
| 0, ordinary | 38.38 | — |
| 1, preceding one-draft screen | 37.58 | 78.3% |
| 2 | 29.65 | 49.2% |
| 3 | 25.59 | 37.2% |
| 4 | 23.58 | 31.6% |
| 5 | 23.17 | 33.9% |
| 6 | 18.60 | 23.6% |
| 7 | 17.64 | 22.8% |

The one-draft value comes from the immediately preceding screen (ordinary
37.66 TPS), not the current process. Forced widths run in order 2,4,7,3,5,6
to avoid a monotonically increasing order confound, but this is still a bounded
screen, not randomized repeated trials. Adaptive cap7 afterward gives 40.65 TPS,
94.6% acceptance, 93 offered drafts across 167 rounds. Most rounds therefore
do not offer a wide block. Wider forced verification has predictable costs:
mean verification 63.55 ms at width2, 92.29 at width4, 142.20 at width7.
Our complete verification graphs support T2..T5; larger widths use the fallback.
These numbers describe the implementation we actually have, not a universal
limit on wider drafting. Outputs/routing can differ between hybrid runs.

## DSpark: relevant, but not supported by this engine

DSpark is DeepSeek's confidence-scheduled semi-autoregressive speculative
decoding approach. A separate trained drafter models dependencies within a
proposed block using a Markov head, and a confidence head estimates acceptance.
The target still verifies proposals. It does not replace target verification
or directly fix its RAM/VRAM cost. DFlash2 itself also has a candidate path
selector; DSpark is not simply adding dependency handling to independent DFlash2
predictions, and Maya's existing confidence cutoff is not a DSpark implementation.

A matching checkpoint exists:
https://huggingface.co/RedHatAI/GLM-5.3-Flash-speculator.dspark-preview

Its published configuration/card specifies a five-layer Qwen3-style backbone,
BF16 safetensors, 64-wide heads, 64 attention and KV heads, eight draft tokens,
auxiliary target layers 20/28/32/36/40/44 plus final45, rank256 Markov head and
confidence head. The preview was served with a specific vLLM development build
and Speculators plugin, not claimed compatible with all stable vLLM versions.
Reported HumanEval acceptance length is 4.74, but that is not a TPS result for
our GPU/offload/quant configuration and is not comparable to our single prompt.

Maya's actual `GlmDflash::load` requires `general.architecture=dflash`, DFlash2
selector/grouped-convolution tensors, and hardcodes head dimension128. Our
current drafter reads target layers6/15/25/34/43. No DSpark execution path was
found in Maya's source. The matching checkpoint repository contains config,
model card and BF16 safetensors, not a ready Maya-compatible GGUF.

Consequently no DSpark TPS run was performed: feeding those weights to DFlash2
would not implement DSpark. A real port needs weight loading/conversion,
target-feature wiring, 64-wide-head draft kernels, Markov/confidence scheduling,
then draft-reference and target-output qualification before benchmarking. The
existing target verification infrastructure is a potential reuse point. Given
our measured verification bottleneck, better confidence/acceptance could help,
but savings are unproven and the extra drafter capacity/compute must be measured.

Sources checked 2026-10-15:
- https://github.com/deepseek-ai/DeepSpec
- https://huggingface.co/RedHatAI/GLM-5.3-Flash-speculator.dspark-preview
- https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2

Source snapshots and benchmark manifests/results are under
`docs/evidence/dflash-lengths-20261015/`.
