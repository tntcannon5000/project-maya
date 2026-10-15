# First-MoE product localization — 2026-10-15

The remaining selected expert-output differences first appear after **down projection**.
All eight captured FFN inputs and post-fused-SwiGLU vectors are bitwise identical across
the two reset/prefill passes. Five expert down outputs differ, with maximum absolute
difference 1.78813934326e-7. This narrows the investigation beyond the prior combine replay.

## Measurement

Same diagnostic fixture as MOE-COMBINE-INVESTIGATION.md: 6,793 occupied prefix tokens,
GPU-only, pool16/RAM94, two resets, graph off, original ownership policy deliberately
retained for diagnostic continuity. Capture the last row of the 6,792-token prefill at
layer3 only. Model formats and settings are in the committed manifest.

Extended opt-in capture stores each selected expert's input and post-SwiGLU vector,
plus MMQ dispatch shape: total rows, maximum expert rows, number of experts, GU/down
quant types, fused-GLU flag and dimensions. Records are written before down overwrites
the output window. No arithmetic change; host synchronization means no TPS claim.

| Route / expert | First-pass experts / max rows | Repeated-pass experts / max rows | Input and post-SwiGLU | Down max abs difference |
|---|---:|---:|---|---:|
| 0 / 61 | 124 / 418 | 4 / 93 | Exact | 1.788e-7 |
| 1 / 237 | 4 / 262 | 79 / 477 | Exact | 1.490e-7 |
| 2 / 217 | 4 / 250 | 4 / 156 | Exact | 2.980e-8 |
| 3 / 10 | 24 / 477 | 24 / 391 | Exact | 0 |
| 4 / 121 | 124 / 418 | 81 / 209 | Exact | 0 |
| 5 / 5 | 24 / 477 | 81 / 209 | Exact | 0 |
| 6 / 214 | 4 / 210 | 4 / 156 | Exact | 5.960e-8 |
| 7 / 44 | 124 / 418 | 4 / 144 | Exact | 7.451e-8 |

All captured products use quant type12 (Q4_K), fused GU/SwiGLU, nff2048 and embedding4096.
Full values and binary hashes are in `moe-product-capture/products.json`; raw sidecars
remain local. Both combined captures are byte-identical to the prior diagnostic's
captures, including routes, partitions, individual outputs and final output. Extra
intermediate capture therefore did not change these sampled results. Production
combine replay still reproduces both final vectors exactly.

## Source mechanism and limits

`src/prefill/moe_mmq.cu` supplies product expert count, total rows and max rows to
the compiled ggml MMQ dispatcher. The generated `build/strata-ggml-cuda/mmq.cuh`
chooses J from column geometry. Fused GU uses an ordinary tiled launch and returns;
non-fused products can use Stream-K with shape-dependent tile/SM partitioning and
partial-sum fixup. Different grouping can thus alter down reduction association even
when each mathematical expert product is unchanged.

This is a source-supported hypothesis for the **measured down-stage difference**,
not yet a causal Stream-K ablation. This capture does not separately prove quantized
activation bytes or down-weight bytes identical. The controlled next discriminator
would capture one selected down matrix, its quantized input and group bounds, then
replay both dispatch geometries through production MMQ. No reason yet to force a
slower deterministic production kernel or change numerical thresholds.

The old historical graph-on failure remains unproven: it did not save prefix state.
These prefill findings do not establish that graph replay caused or fixed it. Existing
controlled exact graph comparisons retain their original scope. No throughput run,
default change or new speed recommendation was made in this round.
