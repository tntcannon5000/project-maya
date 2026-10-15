# One-draft MTP versus DFlash

Final merged engine, c=1, 131072 allocated context, same 6793-token coding prompt,
256 generated tokens, greedy decoding. One warmup and one measured ordinary
control precede each speculative request. No engine changes or launcher changes.
Retained transfer/verification flags preserved; requested GPU pool 22 GiB and RAM
tier 94 GiB. Different drafter allocations mean different actual cache capacities.

| Configuration | Ordinary control TPS | One-draft TPS | Acceptance |
|---|---:|---:|---:|
| Q4_K_M DFlash loaded | 37.66 | 37.58 | 112/143, 78.3% |
| Q4_K MTP block loaded | 35.05 | 36.85 | 111/143, 77.6% |

MTP improves its matched control by 5.1%, but is 2.2% below the ordinary
non-MTP control. One-draft DFlash is effectively flat. These are bounded single
samples, not an architecture-wide conclusion. Earlier adaptive DFlash with up to
four drafts reached 41.73 TPS on this prompt; it was not rerun in this screen.

## Mechanism

| Per drafting round | DFlash | MTP |
|---|---:|---:|
| Draft | 2.25 ms | 1.02 ms |
| Verification | 44.83 ms | 47.03 ms |
| Commit / cache update | 0.41 ms | 0.32 ms |

MTP engaged for 143 one-draft rounds plus one initial zero-draft round, advancing
1.78 tokens/round overall. Its verification graphs performed 42 captures and 5964
replays, no capture/launch failures. It uses the external 4,588,592,064-byte
`GLM-5.3-Flash-MTP-Q4_K.gguf`: 210 MB dense plus routed Q4_K/Q5_K experts.
Its default restricted draft vocabulary was used; no emitted tokens were outside
that vocabulary. This restriction is upstream draft behavior, not target quantization.

DFlash: 2502 GPU slots / 20.46 GiB, 11366 RAM slots. MTP: 2487 GPU slots / 20.78
GiB including 57 draft-layer slots, 11192 RAM slots. MTP adds expert competition
despite a slightly larger overall GPU pool. Speculative-run disk reads were
0.13/token (0.32 ms/token) for MTP versus 0.02/token (0.04 ms/token) for DFlash.
The slower MTP verification cannot be attributed to draft arithmetic alone.

Cold prefill 3220.7 ms DFlash versus 3164.6 ms MTP; no apparent prefill penalty
in this bounded sample. Ordinary/speculative output IDs differ in both sessions;
this is a hybrid-backend viability screen, not a new numerical parity qualification
or identical-route causal attribution. No numerical thresholds changed.

The initial engagement parser incorrectly expected the length histogram to begin
with `1:` and flagged the successful MTP run after clean shutdown. It was corrected
to allow the initial `0:1` round and rechecked against the saved log, without rerunning
inference. All 143 drafting rounds were confirmed width one. The raw engine log,
manifests, IDs, counters and timing samples are preserved under
`docs/evidence/mtp-one-20261015/`.

Verdict: no reason to switch from adaptive DFlash to this forced one-draft MTP
configuration. MTP drafts cheaply, but acceptance is similar and added expert
memory pressure/verification cost consumes that saving. No defaults changed.
