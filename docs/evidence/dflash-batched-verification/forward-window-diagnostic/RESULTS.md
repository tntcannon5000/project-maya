# Ordinary forward versus verification: bounded numerical attribution

One fixed coding input,6793 occupied prefix tokens,128K allocated context.
Fresh processes use identical packed weights, RAM94GiB, requested GPU pool16GiB,
64row scratch and retained grouped/HC/component/exact-column flags. Each pair
uses the same prefix construction: batched prefill of6792tokens, then ordinary
forward for the final prefix token. It records that prefix logit row followed
by six forced-input steps (two in the ranking ablation).

Window steps alternateT3/T5 but commit only the first token. Ordinary forward
advances that same input once. This avoids comparing different generated
histories. It deliberately exercises discarded lookahead and partial-commit
state. This is a numerical diagnostic, not TPS or acceptance qualification.

| Pair | Compared logit values | Exact rows | Maximum absolute difference |
|---|---:|---:|---:|
| GPU-only ordinary vs window-first |1,084,160|7/7|0|
| Hybrid, cold-count ranking |1,084,160|2/7|0.722715|
| Hybrid, ranking disabled, two steps |464,640|2/3|0.985607|

The first two exact rows are the prefix output and the first forced input.
All sampled greedy picks match within each pair; that does not make the
remaining full logits exact or establish universal greedy equivalence. The
previous longer serving run did diverge under free generation. No numerical
threshold has been changed and hybrid output parity is not declared passed.

The GPU-only pair has no drafter loaded. Both modes of each hybrid pair have
Q4_K_M DFlash loaded, so capture and graph eligibility match within each pair.
The hybrid diagnosis sets WIN_ALLCPU=0, leaves CPU-plan override unset, and
uses the same011223344 plan. Different backend allocation histories are the
phenomenon being investigated, not assumed identical because plans match.

## First observed divergence

At logical input position6794, the hybrid first differing logit row, layer3
selects the same ordered expert IDs in both paths:
`176,16,225,256,165,36,77,17`. The nonresident mask is253 in both cases.
Ordinary computes165 onCPU and256 onGPU; the window reverses those choices.
The other CPU experts17,36,77 stay the same. Prefix and first-input route IDs
and fetch/miss/CPU/promotion masks match for all42MoE layers.

The route kernel increments `dcnt` for every selected expert, including future
rows that will be discarded; those counts rank the coldest RAM experts forCPU.
The previous window's two discarded layer3 rows increment expert165 twice and
256 once. This is reconstructed from route records and the kernel source, not
a direct counter-memory dump. `window_commit` restores/replays recurrent state
and advances position; it does not rewind these route counts or residency.
Thus an unchanged CPU plan can choose different experts at the next boundary.

The existing `STRATA_GLM_CPU_COLD=0` ablation removes the layer3 discrepancy.
It does not restore whole-path parity: at layer4 residency/fetch masks first
differ, and at layer6 CPU assignment differs while ordered IDs still match.
The first ordered-ID difference moves to layer20 (layer7 with cold ranking).
This supports two concrete mechanisms: speculative count history and residency
history both change later backend assignments. It does not establish that
these explain every value in the earlier128-output free-generation sample.

CPU native expert activation quantization uses its CPU format/path, while GPU
experts useQ8_1. Switching backend can therefore change logits even when packed
weights and required experts are unchanged. Within a fixed backend, source
review finds matching combine order; CPU completion timing does not pick a
floating-point summation order. Same-mask mismatches would still need input,
weight and source-byte inspection rather than being excused as rounding.

## Decision and limits

The GPU first-row/partial-commit result closes a specific gap in the previous
candidate-versus-verification-control evidence: actual ordinary forward also
matches exactly in this selected controlled-GPU case. It is not a full-window
all-row test, a DFlash proposer acceptance test or near-capacity validation.

Do not change CPU ranking, cache policy or production defaults from this
ablation: no performance win was measured, and disabling ranking does not
solve equivalence. Do not silently label all hybrid drift a new kernel bug,
either. New scheduling/arithmetic changes still require controlled-backend
qualification. Achieving exact ordinary/hybrid speculative equivalence would
need backend assignment/quantization consistency beyond restoringKDA state;
reverting counters alone is insufficient because residency also diverges.

Artifacts: manifests identify binaries and all flags; comparison.json records
per-row full-logit differences and margins; route-comparison.json aligns the
actual ordered route IDs/masks by logical position/layer. Raw logits remain
local. Route logs and compact results are checkpointed. All processes exited.
