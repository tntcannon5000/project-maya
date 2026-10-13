// Experimental exact Q2_K pairs: independent ggml AVX2 chains in zmm halves.
#pragma once

#include "strata/kernels/cpu/native_expert.hpp"

namespace strata::kernels::cpu {

// Implemented in the ordinary-ISA dispatch TU, so feature checks never enter
// an AVX512-compiled function on an unsupported CPU/OS.
bool q2_pair_avx512_available() noexcept;

// Preconditions: available(), n is a positive multiple of 256, packed Q2_K
// rows / Q8_K activation and non-aliasing output scalars. No feature check in
// this inner dot. Its two outputs use ggml's separate original reduction trees.
void q2_k_dot_pair_avx512(int n, const void* row0, const void* row1,
                         const void* q8, float* out0, float* out1);

// Return false without writing output when type/geometry/CPU is unsupported.
// Single-token split-base helpers; caller retains its unchanged fallback.
bool q2_pair_gu_rows(const NativeFmt& f, const uint8_t* gate_base,
                     const uint8_t* up_base, const void* act, float* ff,
                     int r0, int r1, float swiglu_limit);
bool q2_pair_down_rows(const NativeFmt& f, const uint8_t* down_base,
                       const void* hq, float* out, int r0, int r1);

}  // namespace strata::kernels::cpu
