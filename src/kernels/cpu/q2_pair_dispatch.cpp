// Keep this TU at the normal target ISA, matching native_expert.cpp's scalar
// clamp/SwiGLU. Only q2_pair_avx512.cpp receives the AVX512 source flags.
#include "strata/kernels/cpu/q2_pair_avx512.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

namespace strata::kernels::cpu {
namespace {
std::atomic<uint64_t> counted_gu_rows{0}, counted_down_paired_rows{0}, counted_down_tail_rows{0};

void report_pair_counts() {
    std::fprintf(stderr, "maya Q2 pair: GU rows %llu, down paired rows %llu, down fallback tail rows %llu\n",
                 static_cast<unsigned long long>(counted_gu_rows.load(std::memory_order_relaxed)),
                 static_cast<unsigned long long>(counted_down_paired_rows.load(std::memory_order_relaxed)),
                 static_cast<unsigned long long>(counted_down_tail_rows.load(std::memory_order_relaxed)));
}

bool pair_count_on() {
    static const bool enabled = [] {
        const char* value = std::getenv("MAYA_Q2_PAIR_COUNT");
        const bool on = value && std::atoi(value) > 0;
        if (on) std::atexit(report_pair_counts);
        return on;
    }();
    return enabled;
}
}  // namespace

bool q2_pair_avx512_available() noexcept {
    static const bool ok = [] {
        // Existing probes include CPUID, OSXSAVE/XCR0, AVX2/FMA and a
        // conservative AVX512 feature set. Insert/extract of a 256-bit half
        // additionally needs DQ, which cpu_avx512_ok does not explicitly test.
        if (!cpu_avx2_ok() || !cpu_avx512_ok()) return false;
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
        int regs[4];
        __cpuidex(regs, 7, 0);
        return ((static_cast<unsigned>(regs[1]) >> 17) & 1u) != 0;
#elif defined(__x86_64__) || defined(__i386__)
        unsigned a, b, c, d;
        __cpuid_count(7, 0, a, b, c, d);
        return ((b >> 17) & 1u) != 0;
#else
        return false;
#endif
    }();
    return ok;
}

bool q2_pair_gu_rows(const NativeFmt& f, const uint8_t* gate_base,
                     const uint8_t* up_base, const void* act, float* ff,
                     int r0, int r1, float swiglu_limit) {
    if (f.gu_type != GGML_TYPE_Q2_K || f.gu_act != GGML_TYPE_Q8_K ||
        f.n_embd <= 0 || f.n_embd % 256 || r0 < 0 || r1 < r0 ||
        r1 > f.n_ff || !q2_pair_avx512_available()) return false;
    if (pair_count_on()) counted_gu_rows.fetch_add(static_cast<uint64_t>(r1 - r0), std::memory_order_relaxed);
    const bool clamp = swiglu_limit != std::numeric_limits<float>::infinity();
    for (int r = r0; r < r1; ++r) {
        float g = 0.0f, u = 0.0f;
        q2_k_dot_pair_avx512(static_cast<int>(f.n_embd),
                            gate_base + static_cast<size_t>(r) * f.gu_row,
                            up_base + static_cast<size_t>(r) * f.gu_row,
                            act, &g, &u);
        if (clamp) {
            g = std::fmin(g, swiglu_limit);
            u = std::fmin(std::fmax(u, -swiglu_limit), swiglu_limit);
        }
        ff[r] = (g / (1.0f + std::exp(-g))) * u;
    }
    return true;
}

bool q2_pair_down_rows(const NativeFmt& f, const uint8_t* down_base,
                       const void* hq, float* out, int r0, int r1) {
    if (f.d_type != GGML_TYPE_Q2_K || f.d_act != GGML_TYPE_Q8_K ||
        f.n_ff <= 0 || f.n_ff % 256 || r0 < 0 || r1 < r0 ||
        r1 > f.n_embd || !q2_pair_avx512_available()) return false;
    if (pair_count_on()) {
        const int rows = r1 - r0;
        counted_down_paired_rows.fetch_add(static_cast<uint64_t>(rows - rows % 2), std::memory_order_relaxed);
        counted_down_tail_rows.fetch_add(static_cast<uint64_t>(rows % 2), std::memory_order_relaxed);
    }
    int r = r0;
    for (; r + 1 < r1; r += 2) {
        q2_k_dot_pair_avx512(static_cast<int>(f.n_ff),
                            down_base + static_cast<size_t>(r) * f.d_row,
                            down_base + static_cast<size_t>(r + 1) * f.d_row,
                            hq, out + r, out + r + 1);
    }
    if (r < r1) {
        // An arbitrary task boundary can leave one output row. Keep its
        // exact original ggml dot rather than reading a row outside the task.
        const auto dot = ggml_get_type_traits_cpu(GGML_TYPE_Q2_K)->vec_dot;
        dot(static_cast<int>(f.n_ff), out + r, 0,
            down_base + static_cast<size_t>(r) * f.d_row, 0, hq, 0, 1);
    }
    return true;
}

}  // namespace strata::kernels::cpu
