# C=4: upstream Maya v1.1.7 versus retained fork

c=4, 32768-token capacity per slot; same GSQ-RCO 3.0bit Q4-attention pack and expert profile; no MTP, token speculation, K4V or KV streaming. Unprofiled server runs with identical four coding prompts, greedy sampling, up to 512 tokens each, two cohorts per build. Upstream runs first; retained second. No inference source changes. Allocated capacity is not occupied context: these are short prompts with 512 generated tokens.

Aggregate decode TPS counts streamed tokens strictly after the last lane first emits through the earliest lane last emission: an interval when all four lanes are generating. Per-request TPS uses (tokens-1)/(last-first arrival). End-to-end TPS includes prompt processing from batch submission to final DONE. These are wall-clock server measures, not the previous c=1 fixed-input forward-call timer; do not infer concurrency scaling from those different fixtures. Outputs can differ between builds, and this is not a parity or quality test.

| Build | Cohort | Aggregate decode TPS | Per-request TPS range | End-to-end TPS | Four-lane interval seconds |
|---|---|---:|---:|---:|---:|
| upstream | 1 | 37.10 | 8.99-9.29 | 35.58 | 55.02 |
| upstream | 2 | 37.10 | 9.27-9.29 | 37.11 | 55.02 |
| retained | 1 | 39.52 | 9.56-9.89 | 37.81 | 51.64 |
| retained | 2 | 40.78 | 10.20-10.21 | 40.84 | 50.05 |

## Capacity and qualification

- upstream: minimum system RAM available 18.66 GiB.
- glm multi: CUDA0 4 sequence states: the first for 32768 tokens (0.66 GB), 3 more for 32768 (2.59 GB in all, with every sequence's snapshot)
- glm fast: CUDA0 layers [0,45) expert pool 21.02 GB, 49-84 slots/layer (2573 total)
- glm fast: CUDA0 RAM tier 88.66 GB pinned, 10705 slots
- glm KDA graphs: 272 captures, 0 replays, 238 invalidations
- retained: minimum system RAM available 9.60 GiB.
- glm multi: CUDA0 4 sequence states: the first for 32768 tokens (0.66 GB), 3 more for 32768 (2.59 GB in all, with every sequence's snapshot)
- glm fast: CUDA0 layers [0,45) expert pool 21.03 GB, 49-84 slots/layer (2574 total)
- glm fast: CUDA0 RAM tier 97.82 GB pinned, 11819 slots
- glm KDA graphs: 272 captures, 0 replays, 238 invalidations
- maya MoE suffix graphs: 11 captures, 77 replays, 0 invalidations

Both returned READY multi=4, completed eight 512-token requests (4096 tokens per build), and exited normally. Prompts and environments are preserved in manifests, responses and timing in results.json, and wire/hardware logs remain local. Small differences need confirmation; this is one two-cohort comparison, not a broad concurrency qualification. Retained flags are configured but their presence alone is not proof each mechanism engages in the multi-request path.
