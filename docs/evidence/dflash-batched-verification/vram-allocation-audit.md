# Verification VRAM allocation audit

2026-10-15. Source, saved logs and header metadata only; no build, GPU calls or implementation. Configuration: one sequence, 131,072 logical positions, FP16 target latents, DFlash, scratch64, requested expert pool22GiB. Capacity is distinct from occupied context.

**Finding:** there is no established disposable scratch allocation of at least1GiB. The material code opportunity is lending unused *future target KV capacity* to expert residency. This can expose approximately1.45GiB at short occupancy without reducing logical128K support, but requires an explicit allocator/ownership change that returns those bytes to KV as context grows. Merely delaying `cudaMalloc` after sizing a fixed expert pool is incorrect. At fully occupied128K, this opportunity disappears; no unconditional1GiB saving is established.

## What is actually allocated

`component-clean/Q4_K_M/engine.log:2-9` reports scratch864MiB; pre-pool usage10.52/31.84GiB; dense weights5.01GiB; state/KV1.78GiB; actual expert pool20.63GiB despite the22GiB request. The prompt's7,567MiB is explicitly *borrowed* from that expert pool. Its pinned1,296MiB landing ring and the verification's pinned864MiB staging are host RAM, not reclaimable VRAM.

| Device allocation | Source-derived size | Ownership / conclusion |
|---|---:|---|
| Dense weights |5.01GiB logged|Already native quant/BF16; embeddings are omitted from device upload. No duplicate full-precision arena to remove (`glm_model.cu:2780-2849`).|
| Target state arena |1,818.375MiB|Breakdown below. One arena atc=1; extra sequence arenas require `STRATA_GLM_SEQS>1` (`glm_multi.cu:118`).|
| Expert scratch64 |864MiB|64×13.5MiB maximum expert stride; allocated once, shared across layers. Per-token destinations remain live across asynchronous fetch/compute. Even eliminating all of it is below1GiB and would remove required functionality (`glm_fast_path.cu:954-970`; `glm_window.cu:676-705`).|
| Window arena, maximumT16 |75.298MiB|Includes51.664MiB of saved KDA inputs for34 layers. These support partial-commit replay; they are not34 independent temporary workspaces that can simply alias (`glm_window.cu:209-302,1880`).|
| KDA verification backup |145.563MiB|One copy of34 recurrent states/history, not one per token. Required to restore/replay accepted prefixes (`glm_window.cu:381-398`; `glm_fast_path.cu:5888`).|
| Optional request snapshot |145.563MiB plus33KiB open-pool data|Separate lifetime from verification backup; lazily allocated. Removing it cannot provide1GiB (`glm_fast_path.cu:6355-6395`).|
| DFlash total |717MiB logged|Header audit:655.731MiB weight payload; K/V rings40.625MiB, plus bounded block/injection buffers. The declared1M context does not allocate a1M ring: actual window2048+32. Capture staging is pinned host RAM (`glm_dflash.cu:831-835,937-1005`).|
| Prompt buffers/prestage |7,567MiB logged loan|Already share the expert pool tail. `prefill_bind` carves the supplied region; `prefill_return` returns lent slots after stream joins. Freeing another7.4GiB would double-count (`glm_prefill.cu:834-847,1188-1207`).|

The window calculation transcribes the aligned `win_alloc` formula withT16, E4096, DI8192,34 KDA layers and HC batching. The target geometry comes from `glm_model.hpp:65-87` and agrees with the model's projection shapes. MTP is disabled in the saved configuration. The normal activation arena is small; the log's0.11GiB activation figure already includes eight13.5MiB scratch slots.

The3.63GiB “other” line is not evidence of3.63GiB abandoned buffers: its accounting subtracts **only eight scratch slots**, placing the remaining56×13.5=756MiB there, plus window/backup allocations and runtime/context costs (`glm_fast_path.cu:981-987`). The remaining runtime portion has no source-backed claim of being safely reclaimable.

## The material future-KV opportunity

`glm_model.cu:3050-3110` eagerly allocates and resets the complete state arena. At128K the11 DSA layers contain:

| State component | Formula | MiB |
|---|---|---:|
| FP16 latents |11×131072×512×2|1408.000|
| Completed index pools |11×(131072/4)×128×4|176.000|
| Key/gate rings |11×2×8256×128×4|88.688|
| Live KDA state/history |34×8192×(128+3×3)×4|145.563|
| Residual pair |2×4×4096×4|0.125|

Only the first two components grow with causal occupancy. Their combined reserved capacity is1,584MiB, or12,672 bytes per future token. Keeping a committed high-water mark of8,320 positions leaves1,483.453MiB (1.449GiB) unused; at32,768 occupied positions the unused capacity is1,188MiB (1.160GiB); at65,536 it is792MiB. Guards, allocation granularity and partial pools reduce these figures. The unrounded1GiB threshold is approximately46,339 committed positions. At the short saved workload, the22GiB requested-pool cap would limit useful extra residency to roughly22−20.63=1.37GiB, even if more unused KV bytes were available.

This preserves the same FP16 bytes and logical positions. Existing causal bounds select only completed pools and the current tail (`fast_dsa_window`, `glm_window.cu:526-545`); future capacity need not hold physical pages until a prompt/window writes it. The existing fixed `--kv-resident` mechanism also preserves logical positions, but switching to that setting is a separate parameter/backend experiment and is not the proposed code gain here.

## Required ownership design and decision

A concrete approach is a fixed virtual state address range with demand-committed latent/pool pages, paired with a reclaimable expert-page arena. Another is extending the existing explicit tail-lending mechanism to own KV regions, which requires replacing the current state-relative cache addressing. Neither is implemented today: state and main expert pools use monolithic `cudaMalloc` allocations (`glm_model.cu:3110`; `glm_fast_path.cu:1219-1251`). Existing prefill lending makes slots unavailable but does not free physical pages.

The minimum safe feature must commit through `p0+T` **before** prompt/window kernels, retire compute/fetch/background readers before repurposing expert pages, preserve published pointer tables and RAM/demotion ownership, and return capacity to KV before growing context. Prompt scratch and persistent KV require distinct simultaneous loans. Whole-arena reset currently touches every future page (`glm_model.cu:784-790`), so reset must become allocation-aware while preserving initialization. Graph capture/replay needs stable virtual addresses or explicit invalidation; cancellation, partial commit, snapshots and full128K prefill must retain their semantics. VMM availability/granularity must be checked on the actual driver before selecting that design.

**Recommendation:** reject a small lazy/shared-scratch patch as a source-backed route to1GiB. Retain occupancy-backed KV/expert capacity as the one material allocator project: its byte opportunity is real, its current implementation gap is explicit, and it requires no weaker arithmetic or smaller supported context. Do not implement it as an isolated lazy-KV allocation. Its first qualification must demonstrate that expert capacity is reclaimed before long-context growth succeeds; only subsequent miss-byte/transfer and whole-round measurements can establish a speed benefit.
