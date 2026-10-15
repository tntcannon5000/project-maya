# Exact-column compiled resource audit

2026-10-15. CPU-only inspection of the existing `build/CMakeFiles/strata_kernels.dir/src/kernels/cuda/glm_fast.cu.obj` with CUDA 13.2 `cuobjdump --dump-resource-usage` and `--dump-sass`. No build, GPU execution or inference was performed for this audit. The object contains sm_120 code.

| Quant | Window columns | Legacy NC / rows per warp | Exact NC / rows per warp | Registers/thread, legacy → exact |
|---|---:|---:|---:|---:|
| Q4_K | 3 | 4 / 4 | 3 / 4 | 103 → 80 |
| Q4_K | 5 | 8 / 2 | 5 / 2 | 80 → 75 |
| Q8_0 | 3 | 4 / 4 | 3 / 4 | 80 → 64 |
| Q8_0 | 5 | 8 / 2 | 5 / 2 | 106 → 48 |
| Q6_K | 3 | 4 / 4 | 3 / 4 | 80 → 80 |
| Q6_K | 5 | 8 / 4 | 5 / 4 | 174 → 80 |

All compared kernels report `STACK:0`, `LOCAL:0`, `SHARED:1024`, and `CONSTANT[0]:1712`. SASS inspection finds zero `LDL`, `STL`, `LDS` or `STS` instructions in every compared kernel. Thus neither variant shows explicit local-memory spills; the reported 1,024 shared bytes are unchanged and do not correspond to a new source-level shared staging array. Raw selected resource records are in `resources-mvc.txt`; instruction counts are in `resources-mvc-memory-instructions.txt`.

Each kernel still uses 256 threads. Block/row geometry, weight decoding, actual-column arithmetic and reduction order remain unchanged. Removing one of four or three of eight compiled column slots reduces redundant activation/dot/reduction work, but register allocation does not scale proportionally. The large Q8/Q6 register reductions could affect residency; these static counts do not measure achieved occupancy, stalls or critical-path savings. The current target's large dense projections are Q4_K, so Q8/Q6 resource improvements cannot be credited to its dense runtime.

The parent's initial `micro-off.txt` / `micro-on.txt` screen reports approximately 14.4% and 16.2% lower T5 DP4A engine intervals for the synthetic N8192,K8192/K12288 matrices, using averages of the two printed controls. T3 is neutral or variable. These isolated synthetic results do not establish the whole-model gain; full-model exact logits and clean coding timings remain separate gates.

Object SHA256: `a23176ec1c677a7d2b733c920030cc9aa7a4add1471b010a1536b66d72d173db`. Source `src/kernels/cuda/glm_fast.cu` SHA256 at inspection: `c3358f53a11471eac0ca0302973ef645921bfe2b7621d79e61bb7f4b6b107043`.
