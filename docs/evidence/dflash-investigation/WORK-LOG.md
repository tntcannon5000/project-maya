# DFlash verification investigation

2026-10-14. Local work only. Parent checkpoint: `5e33803`; branch `research/dflash-verification`.

## Questions and bounded comparisons

The 21-cell screen found high acceptance but poor TPS and scratch exhaustion. Test actual other implementations before changing Maya:

- Stock Maya v1.1.7 and v1.1.9, identical target pack, Q4 DFlash2, c=1, 128K capacity, 192-token coding task, warmup/control and widths 2/4.
- Installed neurall llama.cpp `9d35489`, same target GGUF/token IDs, 128K capacity: no drafting, DFlash2 width 2, MTP width 1. Its q8 KV, mmap loading, CPU execution and cache policy differ; use within-engine controls, not an arithmetic-parity claim.
- Retained Maya executable saved as `build/strata-before-verify-fix.exe` for a same-session control.

## Established findings

- Stock v1.1.7: control 28.46 TPS, widths 2/4 19.44/14.15 TPS. Scratch fallback observed.
- Stock v1.1.9: control 28.77 TPS, widths 2/4 24.58/16.71 TPS. Scratch fallback observed. Relevant GGUF verification code unchanged between these releases; do not attribute the timing difference to a verified code improvement.
- Independent no-draft: warm repeat 26.36 TPS. MTP: 21.22 TPS, 47/49 draft tokens accepted. MTP is functional but slower in this configuration; the engine's adaptive drafting does not propose on every step.
- Independent DFlash2: loaded, but first request fails `t_layer_inp[il] != nullptr`. GLM graph lacks the requested layer-input outputs. No throughput result. This is an installed-engine compatibility limitation, not evidence against DFlash2.
- Original Maya verification scratch has eight slots; stagger splits them into groups of four. Overflow triggers a full verification-window rerun after recurrent-state restoration. Earlier logs only warned once per process.
- Scratch-resident copies were not looked up for reuse by later tokens; grouping uses pointers, so repeated copies also prevent those entries from sharing a GPU expert group.

## Candidate mechanisms

1. Opt-in `MAYA_WIN_SCRATCH`, 8..128 in multiples of 8: reserve scratch before expert-pool sizing. For a sufficiently provisioned window, each token owns eight slots. Both staggered groups remain disjoint even while operating on different layers. If capacity is insufficient, use the existing claimed-slot/fallback path.
2. Opt-in `MAYA_WIN_REUSE=1`: search earlier plans within the same layer/group for a completed GPU expert copy. Routing/fetches are sequential on the compute stream, so reused data is ready. No global ownership or permanent residency is changed. CPU answers are not reused as weights.
3. Use existing `STRATA_GLM_WIN_STAGGER=0` to test a full-layer group. This trades CPU overlap for larger reuse groups. Also test GPU-only assignment to expose transfer reuse without repeated CPU expert calculations.
4. Add window/retry/reuse counts. Specialize the routing kernel so the ordinary token path does not carry the reuse search.

## Interim experiments

- Expanded scratch + reuse, staggered: all retries removed (69 and 47 windows), 1/160 reused rows at widths 2/4. TPS 32.08/30.49, loaded control 34.67. Background OBS present; control shifted materially.
- Saved retained binary in the same session: control 42.48, widths 2/4 38.81/21.65 TPS.
- Expanded scratch + reuse, full-layer group: control 41.82, widths 2/4 42.39/35.52 TPS; no retries, 58/489 reused rows. Width 4 improved substantially but remained below non-speculative decode. Token histories differ; do not treat acceptance changes as isolated model-quality changes.

## Qualification

- GPU-only full-group throughput was 31.17 / 28.24 TPS at widths 2/4, control 28.66. Reject as a speed improvement over hybrid inference.
- Single-node CPU batching reuses upstream's merged-row implementation in two unpinned half-sized pools; normal decode keeps its original pool. Opt-in `MAYA_WIN_SINGLE_NODE_BATCH=1`. Hybrid plan: 389/11929 and 904/15664 repeated CPU expert entries eliminated (widths 2/4). TPS 39.04/31.83 with a 33.70 control; large between-run control variation prevents an absolute speedup claim versus earlier 42 TPS runs.
- Expanded scratch with reuse on/off: 28,188,160 GPU-only full logits bit-exact, including partial commits and repeat. Legacy overflow-retry versus candidate differs (max 6.5748, 181/182 greedy rows matching). This is unresolved pending a retry-free token-path control; do not call parity qualified yet.

Keep the changes opt-in pending validation. Fixed-input GPU-only verification tests compare full logits with the same arithmetic backend, partial window commits, reset and repeated requests; they exercise mixed expert formats and eviction pressure. Benchmark variants are not a numerical-parity qualification by themselves. No default launcher change or publication.

## Final qualification and decision

Retry-free token-path control resolved the candidate parity question: candidate/reuse-off/token-path all match bit-exactly across 28,188,160 logits. The differing result belongs to legacy overflow/retry (maximum difference 6.574832); its exact internal cause remains open. Controlled CPU merged/serial placement also matches all 28,188,160 logits exactly. Native CPU row test passed.

Final cancellation/recovery smoke passed and engine exited cleanly: control 35.14 TPS, width 2 34.14 TPS, recovery 33.03 TPS. This does not establish CPU batching as a reliable throughput win. Candidate flags remain opt-in and daily settings unchanged. See RESULTS.md for all engine comparisons, memory costs, engagement counts and limitations. No broad prefill or long occupied-context qualification claimed.
