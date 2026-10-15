# Pre-verification state investigation

2026-10-15. Bounded diagnostics of the preserved historical graph mismatch. No production arithmetic, ownership policy or launcher changes. Fixture-only state capture and comparison tooling added.

## State capture

The fixture now optionally records the engine's conversation snapshot after prefill/final-prefix forward and before the first verification window. Each snapshot contains152,633,344bytes of recurrent state/conv history and86,118,912bytes of indexer/open-pool/DSAcache payload, plus220header bytes. Four snapshots remain local, not committed. Snapshot allocation/copies perturb memory/timing; these are never TPS evidence. The comparison tool validates layout, rejects truncated/mismatched inputs, compares hashes/bytes and checks finite bitwise logits. Four synthetic tests pass.

Matched old-policy GPU-only graph-off/on (RAM94GiB,pool16GiB,6793occupied prefix,T3/T5,sixwindows,tworesets) has:

- Both238,752,476-byte serialized prefix snapshots exact across graph-off/on.
-309,760prefix logits and7,434,240verification logits exact across graph-off/on.
- Current output exactly matches the original historical graph-off artifact. It does not match the original failed graph-on artifact(maxabs2.012939).

The historical failed run did not save prefix state. This rerun cannot retroactively attribute its cause, and disabling the ownership fix here does not make the old policy safe. It is a diagnostic matching the earlier configuration.

## Repeat-request divergence is measured before verification

Within either arm, resetting and prefilling the same prompt again changes the prefix logits(maxabs2.682236) before any verification graph is run. Serialized recurrent layers0,1,2match exactly across resets. The first differing recurrent layer is4(the next recurrent layer after MoE layer3), maxabs5.2452e-6;31of34recurrent layers differ downstream. This is pre-verification variation, independent of the graph-off/on contrast.

The existing GLM_CB_DIR diagnostic was then used in one graph-off fixture. Saved its prefill seams before final-prefix forward could overwrite their filenames. Sampled seams are the last prefill row, not every activation row.

| Last-row seam | Across resets | Maximum absolute difference |
|---|---|---:|
| Layers0-2: attention norm, mixer, FFN norm/output |Exact|0|
| Layer3 attention norm/mixer/FFN norm |Exact|0|
| Layer3 FFN output |Different|1.490116e-7|
| Layer4 attention norm |Different|8.940697e-8|
| Layer4 mixer |Different|1.154840e-6|
| Layer4 FFN norm |Different|1.726300e-5|
| Layer4 FFN output |Different|1.136310e-3|

Instrumentation caveat: GLM_CB_DIR synchronizes stages and disables prefill tail skipping, and affects some decode graph use. Its prefix outputs differ from the snapshot-only fixture(maxabs2.820549). Therefore it is a separate controlled diagnostic of repeat-prefill behavior, not an exact reproduction of the original failure. Its within-run comparisons use identical instrumentation in both repeats.

## Interpretation and bounded next experiment

Established: graph-off/on verification is exact from identical captured state in the new pair; repeat-prefill numerical state differs without verification graphs; sampled first-MoE input matches while its output first differs by~1.5e-7.

Source-grounded mechanisms: resident/prestaged/staged expert sets change output-window flush partitions, hence floating accumulation grouping in moe_combine_add. MMQ expert counts/max-row shapes can also change stream-K split/reduction order. Layer3is the first routed-expert layer, making this a concrete localization consistent with those mechanisms. It is not yet a proof that all variation is harmless rounding, nor the cause of the old unrecorded failure.

Next decisive experiment, if needed for sign-off: capture just layer3's last-row route IDs/weights, individual expert outputs, shared output and actual flush partitions. Replay the captured combine using the production kernel, then apply cold/warm partitions to common expert outputs to distinguish combine-order changes from MMQ product changes. No broad test suite, stochastic retry loop or threshold relaxation. Preserve the original failure and keep graphs opt-in until that distinction is settled.
