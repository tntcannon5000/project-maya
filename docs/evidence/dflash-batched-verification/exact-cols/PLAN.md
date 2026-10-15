# Exact-column DP4A qualification

Mechanism: remove unused fourth/eighth column arithmetic from T3/T5 verification. Preserve native quant dot products, reduction order, per-warp row counts and mixed-job dispatch. Opt-in `MAYA_MV_EXACT_COLS`; ordinary single-token dispatch is unchanged.

Completed: 587 strict native-quant and mixed-column cases pass with the flag both off and on; the independent Q4 harness also passes 240 cases in both configurations. Engagement messages identify the selected type, width and row geometry. Larger T5 microbenchmarks improve roughly 14–16%; T3 is neutral or noisy. These are isolated kernel results, not model TPS.

Current gate: corrected GPU-only T3/T5 fixed-input windows, six windows in each of two repeated requests, including partial commits. Compare all logits exactly. If they pass, run one short coding comparison (128 output tokens, four drafts), with the flag off and on and an ordinary-decode control. Keep component streaming off to isolate arithmetic. Broaden testing only if whole-round savings justify it.
