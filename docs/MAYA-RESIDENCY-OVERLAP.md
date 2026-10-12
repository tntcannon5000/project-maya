# Focused residency and overlap investigation

Start: `3a4f906`, retained graph configuration at `799c34f`. Branch
`research/maya-residency-overlap`. Previous prototypes remain at `c3cb3a4`.
No publication, arithmetic changes, speculative tokens or new quantization.
Keep RAM backing and demotion scheduling. Select one or two targeted runs
per iteration; do not launch the full previous suite.

## Mechanisms and decision gates

1. Residency: Maya already admits GPU-fetched experts into spare slots and
   evicts by aged frequency. Simple recency is not a missing magic feature.
   A recent-32 replay helped writing 1.2% but worsened CSV 3.4%; reject.
   Next-route transition reuse learns causal expert co-occurrence across
   consecutive routes of the SAME layer. Smooth sparse rows with frequency;
   score eviction candidates, without extra prefetch or arithmetic changes.
   Causal replay: writing 392.82 -> 388.26 MB/token; CSV 413.92 -> 395.82.
   Replay still simplifies spare admission and CPU identities: test actual
   demand bytes, changed-victim counters and time. Plausible saving ~0.5ms,
   not the oracle's 185MB. New history costs ~16MiB at this geometry and some
   service-thread computation; reject if overhead or altered placement loses.

2. Overlap: reuse the validated GU-prefix/down-suffix event fork/join, but
   limit competing down-copy CTAs to 32, rather than a full-SM-sized grid.
   This limits copy blocks, not guaranteed SM affinity. Prefix retains full
   bandwidth. Same quantized bytes, single original gate/up, down waits for
   suffix and graph completion protects slot reuse. Native aligned formats
   only; expert prediction cannot share these events. Plausible saving below
   1ms if contention falls without starving PCIe. Inspect an actual trace
   before claiming improved overlap efficiency.

## Chronological checkpoints

- Prior work is committed, with only unrelated `maya-bench.txt` untracked.
- Recent-frequency and causal transition feasibility results saved under
  `docs/evidence/maya-residency-overlap`. No model runs used for those screens.
- First overlap32 run: 40.10/43.34 TPS first/repeated versus previous
  graph-only 39.31/42.85. Difference is within prior variation; NOT a confirmed
  gain. Evidence initially landed at `../maya-demand/overlap32` because the
  runner overrode an explicit evidence root; runner now respects it.
- Experimental copy kernel compiled after fixing placement of launch-error
  checking. No arithmetic operations were changed.
- Checkpoint `85385a4`: both opt-in mechanisms, causal replay, preserved
  previous engine behavior when flags are unset.
- Fresh same-binary graph baseline 39.86/43.30 TPS. Transition policy
  39.97/42.72 TPS, repeated demand 413.79 -> 383.00 MB/token, GPU hits
  228.92 -> 235.62 and CPU experts 63.60 -> 60.46. 7,815 different victim
  choices across 13,272 observed routes. Mechanism engages, throughput not
  improved. Low-grid overlap43.34 and fresh baseline43.30 are effectively tied.
- Follow-up dependency: reuse scoring ran before CPU answers were published.
  c=1 now scores AFTER publishing the answer, before releasing the service
  boundary. Merged windows retain their original service ordering. This is
  not an arithmetic change; the small count-aging-boundary difference is
  recorded (scoring now sees aged counts at those boundaries).
- Two diagnostic captures selected: transition and bounded overlap, to locate
  costs. No full workload suite. Nsight event export is substantial overhead;
  these captures remain separate from unprofiled throughput evidence.
- Transition after CPU publication: 40.07/43.54 TPS. Transition plus all-layer
  bounded overlap: 40.29/44.70 TPS. Mechanisms may complement one another;
  isolated overlap was indistinguishable from fresh baseline.
- Overlap trace: arithmetic 9.08ms (no prior >11ms contention), overlap0.65ms,
  idle1.97ms. Gap attribution: uncaptured work grows0.877->1.653ms; internal
  graph gaps0.097->0.090ms. This specifically motivates restricting component
  overlap to captured recurrent layers, avoiding event costs on MLA layers.
- Strata source comparison: `Strata-v042-fast/src/program/generate.cpp` adapt
  logic ranks incoming/victim frequency differences, requires a gain margin
  (default1.5), and budgets swaps. Its `ExpertCache::admit` does not itself
  evict; adaptive replacement is managed by the serve loop. Maya promotes
  fetched misses opportunistically. This is a real admission distinction,
  not evidence that Qwen's cache hit rate transfers to GLM. A dynamic margin
  is a possible next policy; it is not claimed implemented or measured here.
- Captured-only overlap 40.37/43.65 TPS; combined transition + captured-only
  overlap 40.70/45.20 TPS. Baseline39.86/43.30: repeat +4.4%, first +2.1%.
  Same demand383.00MB and same changed-victim count as the transition prototype.
- All overlap-only compared logits remain exact. All transition variants
  share the SAME numerical difference from the hybrid baseline (max5.314,
  greedy top1 98.83% across256 forced inputs). Moving scoring/adding overlap
  introduces no additional discrepancy. Controlled GPU-only qualification
  follows; these hybrid differences are not declared numerical parity.
- Teardown now joins the component side stream before releasing pinned
  expert sources, including an error before the normal consumer join.
- Held-out web continuation plus STOP/recovery selected as one workload
  pair; no complete writing/coding suite. Long-context GPU-only comparison
  will combine ownership, mixed-format, prefill-pressure and parity checks.

Correctness qualification: transport-only outputs must be exact. Residency
can change CPU/GPU assignment; compare controlled GPU-only backends if it
proves useful, and report hybrid differences without relaxing thresholds.
Checkpoint prototypes even when rejected, then keep PR-facing changes small.
