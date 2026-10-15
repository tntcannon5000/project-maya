# Native MoE graph serving gate

2026-10-15. One 40-token coding prompt, 128 emitted-token cap, 128K allocated context, Q4_K_M drafter. Both arms enable paired expert dots, HC batching, component fetch, exact columns, scratch64, token-owned staging, nonstaggered windows and the promotion-ownership fix. Only MAYA_WIN_MOE_GRAPH differs. Both allocate 2,523 expert slots (reported 20.63 GB pool), requested pool22/RAM94 GiB. No profiler/owner scanner; identical per-round drafter statistics enabled.

| Request | Graph off TPS | Graph on TPS | Difference | Verify ms/round off -> on |
|---|---:|---:|---:|---:|
| Ordinary control |45.27|45.06|-0.48%|n/a|
| Two drafts, first request |49.69|52.00|+4.64%|54.816 -> 52.240|
| Four drafts, first request |49.16|51.29|+4.33%|81.217 -> 77.736|
| Two drafts, repeated |50.82|52.42|+3.13%|53.624 -> 51.867|
| Four drafts, repeated |51.22|52.91|+3.31%|80.511 -> 77.809|

Every corresponding output-token history, offered count and accepted count matches. Acceptance two-draft83/90; four-draft96/124 first and97/120 repeated. These are repeated requests within one process per arm, not independent statistical trials. Different-width histories are not an identical-work comparison.

Mechanism engagement: 84 captures and 6,258 replays across four measured speculative requests, zero bypasses or capture/fallback/launch errors. Cancellation and32-token recovery add630replays and both pass; clean process exit0. Graphs are built per width on first use and reused; host capture cost is included in first-request timing.

Cold prompt preparation731.1 ->718.9ms; warm prompt times29.6 ->29.4ms. This small prompt is only a prefill smoke check. Active-clock samples overlap: off2550-2632MHz/on2542-2632MHz, memory16701MHz, temperatures43-48C. Thus no gross clock disparity is observed; this does not exclude fine-grained variance.

Decision: modest positive gate, retain opt-in and check the existing longer coding task (6,793occupied tokens) with reversed launch order. Do not credit this experiment with all50+TPS: the graph contribution is the paired3-4.6%, not the whole combined implementation. No60TPS claim, default change or production sign-off. Historical numerical failure remains unattributed.

The earlier serving-off attempt lacked the fixture scheduling/scratch settings and is excluded. The summary tool now rejects non-engagement, mismatched manifests/environments and graph errors. Raw logits remain local.

## Longer occupied-context gate

Same saved6,793-token coding prompt,128output-token cap. Graph-on launched first to reverse the short test order. Both arms have identical settings except the graph flag; all paired histories and acceptance counts match.

| Measurement | Graph off | Graph on |
|---|---:|---:|
| Ordinary TPS |40.87|40.96|
| Two-draft TPS |24.02|25.05|
| Four-draft TPS |23.57|23.78|
| Two-draft verify ms/round |68.253|65.319|
| Four-draft verify ms/round |91.485|90.596|
| Cold prefill ms |3232.9|3219.8|
| Two-draft accepted/offered |53/148|53/148|
| Four-draft accepted/offered |70/228|70/228|

Two-draft gain4.29%; four-draft0.91% is within plausible variation. Prefill shows no regression in this bounded pair, not statistical assurance. Cache/graph engagement and complete manifests are preserved in long-serving-summary.json. This task's forced drafting remains slower than ordinary. Post-hoc confidence counts in acceptance-budget.json motivate a bounded existing-policy check, not a claimed speedup or an acceptance guarantee. Retain graphs opt-in; no additional broad tests.
