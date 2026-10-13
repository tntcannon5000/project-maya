// Prototype derived from ggml's Q2_K AVX2 dot (MIT, pinned llama.cpp).
// Each low/high 256-bit half preserves one original row's integer add tree,
// eight FP accumulator lanes, min/data FMA ordering and horizontal reduction.
#include "strata/kernels/cpu/q2_pair_avx512.hpp"

#include <immintrin.h>
#include <cassert>
#include <cstdint>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml-impl.h"
#include "simd-mappings.h"

namespace strata::kernels::cpu {
namespace {

inline __m512i pair256(__m256i lo, __m256i hi) {
    return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}

inline __m256i repeat128(__m128i v) {
    return _mm256_inserti128_si256(_mm256_castsi128_si256(v), v, 1);
}

inline __m512 broadcast_pair(float lo, float hi) {
    return _mm512_castsi512_ps(pair256(_mm256_castps_si256(_mm256_set1_ps(lo)),
                                      _mm256_castps_si256(_mm256_set1_ps(hi))));
}

// Same 128-bit-lane shuffle masks as get_scale_shuffle_q3k(i), duplicated
// for the second original ymm chain. No cross-half integer regrouping.
inline __m512i scale_shuffle(int i) {
    const uint64_t a = static_cast<uint64_t>((4 * i) | ((4 * i + 1) << 8)) *
                       UINT64_C(0x0001000100010001);
    const uint64_t b = static_cast<uint64_t>((4 * i + 2) | ((4 * i + 3) << 8)) *
                       UINT64_C(0x0001000100010001);
    return _mm512_setr_epi64(a, a, b, b, a, a, b, b);
}

inline float hsum_original(__m256 v) {
    // Copy of selected ggml x86 hsum_float_8; preserve these additions.
    __m128 res = _mm256_extractf128_ps(v, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(v));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

}  // namespace

void q2_k_dot_pair_avx512(int n, const void* row0, const void* row1,
                         const void* q8, float* out0, float* out1) {
    static_assert(QK_K == 256, "prototype matches the pinned 256-value Q2_K dot");
    assert(n > 0 && n % QK_K == 0);
    const auto* x0 = static_cast<const block_q2_K*>(row0);
    const auto* x1 = static_cast<const block_q2_K*>(row1);
    const auto* y = static_cast<const block_q8_K*>(q8);
    const __m512i m3 = _mm512_set1_epi8(3);
    const __m128i m4 = _mm_set1_epi8(0x0f);
    const __m512i sh0 = scale_shuffle(0), sh1 = scale_shuffle(1);
    const __m512i sh2 = scale_shuffle(2), sh3 = scale_shuffle(3);
    __m512 acc = _mm512_setzero_ps();

    for (int i = 0; i < n / QK_K; ++i) {
        // Use exactly the selected ggml CPU half-conversion macro, including
        // its lookup table path. Caller initializes ggml as for native dots.
        const float d0 = y[i].d * GGML_CPU_FP16_TO_FP32(x0[i].d);
        const float d1 = y[i].d * GGML_CPU_FP16_TO_FP32(x1[i].d);
        const float dm0 = -y[i].d * GGML_CPU_FP16_TO_FP32(x0[i].dmin);
        const float dm1 = -y[i].d * GGML_CPU_FP16_TO_FP32(x1[i].dmin);
        const __m128i ms0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x0[i].scales));
        const __m128i ms1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x1[i].scales));
        const __m256i as0 = _mm256_cvtepi8_epi16(_mm_and_si128(ms0, m4));
        const __m256i as1 = _mm256_cvtepi8_epi16(_mm_and_si128(ms1, m4));
        const __m512i mins = pair256(
            _mm256_cvtepi8_epi16(_mm_and_si128(_mm_srli_epi16(ms0, 4), m4)),
            _mm256_cvtepi8_epi16(_mm_and_si128(_mm_srli_epi16(ms1, 4), m4)));
        const __m512i sums = _mm512_broadcast_i64x4(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[i].bsums)));
        const __m512i prod = _mm512_madd_epi16(mins, sums);
        acc = _mm512_fmadd_ps(broadcast_pair(dm0, dm1), _mm512_cvtepi32_ps(prod), acc);

        const __m512i scales[2] = {
            pair256(repeat128(_mm256_castsi256_si128(as0)),
                    repeat128(_mm256_castsi256_si128(as1))),
            pair256(repeat128(_mm256_extracti128_si256(as0, 1)),
                    repeat128(_mm256_extracti128_si256(as1, 1))),
        };
        __m512i sumi = _mm512_setzero_si512();
        for (int j = 0; j < QK_K / 128; ++j) {
            const __m512i bits = pair256(
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x0[i].qs + 32 * j)),
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x1[i].qs + 32 * j)));
            const int8_t* q = y[i].qs + 128 * j;
            const __m512i a0 = _mm512_broadcast_i64x4(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(q)));
            const __m512i a1 = _mm512_broadcast_i64x4(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 32)));
            const __m512i a2 = _mm512_broadcast_i64x4(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 64)));
            const __m512i a3 = _mm512_broadcast_i64x4(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 96)));
            const __m512i q0 = _mm512_and_si512(bits, m3);
            const __m512i q1 = _mm512_and_si512(_mm512_srli_epi16(bits, 2), m3);
            const __m512i q2 = _mm512_and_si512(_mm512_srli_epi16(bits, 4), m3);
            const __m512i q3 = _mm512_and_si512(_mm512_srli_epi16(bits, 6), m3);
            __m512i p0 = _mm512_maddubs_epi16(q0, a0);
            __m512i p1 = _mm512_maddubs_epi16(q1, a1);
            __m512i p2 = _mm512_maddubs_epi16(q2, a2);
            __m512i p3 = _mm512_maddubs_epi16(q3, a3);
            p0 = _mm512_madd_epi16(_mm512_shuffle_epi8(scales[j], sh0), p0);
            p1 = _mm512_madd_epi16(_mm512_shuffle_epi8(scales[j], sh1), p1);
            p2 = _mm512_madd_epi16(_mm512_shuffle_epi8(scales[j], sh2), p2);
            p3 = _mm512_madd_epi16(_mm512_shuffle_epi8(scales[j], sh3), p3);
            p0 = _mm512_add_epi32(p0, p1);
            p2 = _mm512_add_epi32(p2, p3);
            sumi = _mm512_add_epi32(sumi, _mm512_add_epi32(p0, p2));
        }
        acc = _mm512_fmadd_ps(broadcast_pair(d0, d1), _mm512_cvtepi32_ps(sumi), acc);
    }
    *out0 = hsum_original(_mm512_castps512_ps256(acc));
    *out1 = hsum_original(_mm512_extractf32x8_ps(acc, 1));
}

}  // namespace strata::kernels::cpu
