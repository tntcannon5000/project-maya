#pragma once
#include "strata/kernels/glm_fast.hpp"

// Isolated feasibility experiment; never called by the engine dispatcher.
// Exactly three/five columns, Q4_K, n_in divisible by 256. No g_skip support.
bool imma_q4_exact_trial(const strata::kernels::glmf::MvJob& job, cudaStream_t stream);
