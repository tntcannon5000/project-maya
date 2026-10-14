# Outside-expert verification arithmetic: source assessment

2026-10-14. Source and existing evidence only; no implementation, build or inference in this assessment. Most large projection matrices already reuse weights across verification tokens. The next substantial performance gate remains useful route/fetch overlap.

Costs below are CUDA intervals per verification window from `baseline-summary.json:96-147`, covering layers `[0,45)` in the same short coding screen. Two/four drafts correspond to target windows T=3/5. Intervals can include host submission gaps and are ceilings for each entire stage, not predicted savings.

| Stage | T=3 ms | T=5 ms | Existing implementation |
|---|---:|---:|---|
| KDA projections + output | 4.24 | 5.40 | `Glm5Model::fast_window_layers`, `glm_window.cu:1061-1083`: all six inputs and output use `MvJob.ncol=n`. |
| DSA projections + output | 1.98 | 2.39 | `Glm5Model::fast_dsa_window`, `glm_window.cu:383-407,438-450,493-496`: `ncol=T`. |
| Router/shared gate/up | 1.64 | 1.83 | `fast_moe_window`, `glm_window.cu:507-531`: `ncol=T`. Shared down is also multicolumn at `glm_window.cu:687-697`. |
| Dense FFN | 0.30 | 0.38 | `fast_window_layers`, `glm_window.cu:1091-1107`: gate/up/down use multiple columns; SwiGLU remains a small per-token launch. |
| Hyper-connections | 3.39 | 5.43 | `fast_window_layers::hc_rows`, `glm_window.cu:1017-1042`: complete HC invoked per token with shared reusable scratch/counter. |
| KDA token body | 1.57 | 3.68 | `kda_token`, `glm_window.cu:90-113,1075-1078`: prep and recurrence per token. |
| DSA attention interval | 3.42 | 5.35 | `fast_dsa_window`, `glm_window.cu:451-492`: absorbed q/out projections already batch; score, select and latent attention remain per token. |

`glm_fast::mv` dispatches common Q4_K/Q6_K/Q8_0 jobs to `mvc_kernel_t` (`glm_fast.cu:1676-1720,4404-4435`). Its K loop loads decoded weight calls once before the column loop. `mvc_kernel_bf16` similarly loads each BF16 row fragment once and stages activations (`glm_fast.cu:1773-1823`). The absorbed MLA uses `headwise_gemv_cols_kernel` with weights outside the column loop (`glm_fast.cu:2778-2818`). These are existing reuse, not unimplemented GEMM opportunities.

A header-only check of the target GGUF identified Q4_K (type 12) for the head, all 169 quantized attention-projection tensors, all 126 shared-expert tensors and nine dense-FFN tensors. The other attention tensors are BF16/F32; the pack keeps its large float matrices as BF16 (`glm_model.cu:2787-2789`). `index.txt`'s native rows carry shape placeholders, not their actual GGUF quant types (`tools/iq_pack.py:12-14`). Inspection stopped at the tensor-header boundary, byte 9,516,456; no tensor payload was read.

The final head already sets `ncol=T` (`glm_window.cu:1254-1273`). Its measured interval was only 0.30/0.47 ms in the deliberately perturbed group diagnostic (`group-summary.json:119,156`). Head prep/pick were each below 0.1 ms there. Target jobs retain `MvJob.approx=false`; enabling rounding-tolerant tensor-core paths would require new numerical qualification (`glm_fast.hpp:40-42`, `glm_fast.cu:4378-4401`). No head or projection rewrite is justified by this evidence.

Concrete next candidates, in order:

1. **Batch the existing three exact HC3 phases across a window group.** Use a grid `(32,T)` and private per-token partial rows; adapt the existing `hc3_kernel<1/2/3>` sequence (`glm_fast.cu:2258-2392,4608-4613`) with token strides. Keep `gridDim.x=32`, the same 32-partial reductions, Sinkhorn operations and q8 stores. Kernel boundaries replace global grid spins, so this avoids extending the resident-grid requirement in `hc()` (`glm_fast.cu:4593-4605`). About 3.25 KiB partial scratch per token is sufficient for the current 25+1 rows. At T=3 this has the same launch count as three singleton calls but can process independent tokens concurrently; at T=5 it replaces five calls with three phases. This targets the 3.39/5.43 ms HC interval. It is the smallest complete next arithmetic prototype, conditional on weak async-overlap results. Validate all logits, each partial commit/reset path, and resource/phase timing; do not assume weight-traffic or whole-round savings.

2. **Persistent exact-order KDA recurrence plus window prep.** `glmf::kda_rec_kernel` reloads and writes the 128x128 state every token (`glm_fast.cu:2487-2538`). A window version can retain its existing 64 register values per lane across T, preserve the two-half reductions and q8 output, and write state once. Window prep must preserve the 128-thread q/k normalization and convolution-history order while batching the BF16 f_b/g_b gate products (`glm_fast.cu:2436-2483`). The prefill persistent kernel supplies a structural example (`glm_batch.cu:266-336`) but writes FP16 output and normalizes differently, so direct reuse is not qualified. Preserve raw `W.kin` projections for partial-commit replay (`glm_window.cu:1861-1883`). The entire eligible interval is only 1.57/3.68 ms; recurrence/prep split timing is needed before implementation.

3. **Batch DSA score/select/latent attention only after finer timing.** Give each token its own score, selected-cell and split-attention partial rows while retaining its causal completed-pool count (`glm_window.cu:463-477`). Extend existing per-token score/select/split/merge kernels, preserving their reductions; q_abs and MLA output are already batched. Initially restrict to KV streaming off, because streaming resolves each token's cells before its attention (`glm_window.cu:473-475`). The 3.42/5.35 ms interval also includes already-batched absorbed projections; measure its internal components and cross-token selected-cell overlap before attributing a reusable-cache opportunity.

Even the complete 17.11 ms non-route/non-expert remainder at T=3 is smaller than the approximately 19 ms whole-round reduction sought for 60 TPS (`WORK-LOG.md`, first profile assessment). These candidates are bounded improvements, not a source-backed route to that headline alone.
