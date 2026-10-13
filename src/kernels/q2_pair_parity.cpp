// CPU-only, deterministic exact parity qualification. No timing/benchmarking.
// Link strata_kernels_cpu + ggml-cpu/ggml-base, compile this TU at normal ISA.
#include "strata/kernels/cpu/q2_pair_avx512.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {
uint32_t state = 0x51a7e3d9u;

float sample() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return static_cast<float>(static_cast<int32_t>(state >> 8) - 8388608) /
           8388608.0f;
}

uint32_t bits(float x) {
    uint32_t v;
    std::memcpy(&v, &x, sizeof(v));
    return v;
}

bool same(float got, float expected, const char* what, int n, int row) {
    if (bits(got) == bits(expected)) return true;
    std::fprintf(stderr, "%s n=%d row=%d got=%08x expected=%08x (%g / %g)\n",
                 what, n, row, bits(got), bits(expected), got, expected);
    return false;
}

bool pair_dots() {
    const auto* wt = ggml_get_type_traits_cpu(GGML_TYPE_Q2_K);
    const auto* at = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    for (int n : {256, 512, 2048, 4096}) {
        const size_t wb = ggml_row_size(GGML_TYPE_Q2_K, n);
        const size_t ab = ggml_row_size(GGML_TYPE_Q8_K, n);
        // Valid type alignments, deliberately varied cache-line offsets.
        std::vector<uint8_t> w0(wb + 64), w1(wb + 64), aq(ab + 64);
        std::vector<float> a(n), b(n), x(n);
        for (int trial = 0; trial < 24; ++trial) {
            const int wo0 = 2 * (trial % 16), wo1 = 2 * ((trial + 3) % 16);
            const int ao = 4 * ((trial + 1) % 8);
            for (int k = 0; k < n; ++k) {
                a[k] = 0.04f * (sample() + sample() + sample());
                b[k] = 0.03f * (sample() + sample() + sample());
                x[k] = sample() + sample() + sample();
                switch (trial % 6) {
                    case 0: a[k] = b[k] = x[k] = 0.0f; break;
                    case 1: x[k] = (k & 1) ? -1.0f : 1.0f; break;
                    case 2: a[k] *= 0.0001f; b[k] *= 0.001f; break;
                    case 3: x[k] *= 64.0f; break;
                    case 4: x[k] = (k % 127 == 0) ? x[k] : 0.0f; break;
                    default: break;
                }
            }
            wt->from_float(a.data(), w0.data() + wo0, n);
            wt->from_float(b.data(), w1.data() + wo1, n);
            at->from_float(x.data(), aq.data() + ao, n);
            float e0 = 0.0f, e1 = 0.0f, g0 = 0.0f, g1 = 0.0f;
            wt->vec_dot(n, &e0, 0, w0.data() + wo0, 0, aq.data() + ao, 0, 1);
            wt->vec_dot(n, &e1, 0, w1.data() + wo1, 0, aq.data() + ao, 0, 1);
            cpu::q2_k_dot_pair_avx512(n, w0.data() + wo0, w1.data() + wo1,
                                     aq.data() + ao, &g0, &g1);
            if (!same(g0, e0, "pair low", n, trial) ||
                !same(g1, e1, "pair high", n, trial)) return false;
        }
    }
    return true;
}

void make_matrix(std::vector<uint8_t>& dst, int n, int rows, size_t row_bytes) {
    const auto quant = ggml_get_type_traits_cpu(GGML_TYPE_Q2_K)->from_float;
    std::vector<float> row(n);
    for (int r = 0; r < rows; ++r) {
        for (int k = 0; k < n; ++k) {
            row[k] = 0.03f * (sample() + sample() + sample());
            if (r % 19 == 0) row[k] = 0.0f;
            if (r % 23 == 0) row[k] = (k & 1) ? -0.0625f : 0.0625f;
        }
        quant(row.data(), dst.data() + static_cast<size_t>(r) * row_bytes, n);
    }
}

bool split_rows(int e, int ff) {
    cpu::NativeFmt f;
    std::string error;
    if (!cpu::native_fmt(GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, e, ff, f, error)) {
        std::fprintf(stderr, "native format: %s\n", error.c_str());
        return false;
    }
    std::vector<uint8_t> gate(f.gu_row * ff), up(f.gu_row * ff), down(f.d_row * e);
    make_matrix(gate, e, ff, f.gu_row);
    make_matrix(up, e, ff, f.gu_row);
    make_matrix(down, ff, e, f.d_row);
    std::vector<float> x(e), ref_ff(ff), got_ff(ff), ref_dn(e), got_dn(e);
    std::vector<uint8_t> aq(f.act_bytes), ref_h(f.h_bytes), got_h(f.h_bytes);
    const void* acts[1] = {aq.data()};
    float* ref_ffs[1] = {ref_ff.data()};
    const std::array<std::array<int, 2>, 5> gu_ranges = {{{0, 1}, {1, 13},
        {13, ff - 3}, {ff - 3, ff}, {3, 3}}};
    const std::array<std::array<int, 2>, 5> dn_ranges = {{{0, 1}, {1, 5},
        {5, e - 3}, {e - 3, e}, {3, 3}}};
    const float limits[] = {std::numeric_limits<float>::infinity(), 7.0f, 0.25f, 0.0f};

    for (float limit : limits) {
        for (int k = 0; k < e; ++k) x[k] = sample() + sample() + sample();
        cpu::native_quant_act(f, x.data(), aq.data());
        cpu::native_gu_rows_split(f, gate.data(), up.data(), acts, 1, ref_ffs,
                                  0, ff, limit);
        std::fill(got_ff.begin(), got_ff.end(), -123456.0f);
        for (const auto& range : gu_ranges) {
            const auto before = got_ff;
            if (!cpu::q2_pair_gu_rows(f, gate.data(), up.data(), aq.data(),
                                      got_ff.data(), range[0], range[1], limit)) return false;
            for (int r = 0; r < ff; ++r) {
                const float expected = r >= range[0] && r < range[1] ? ref_ff[r] : before[r];
                if (!same(got_ff[r], expected, "GU boundary", e, r)) return false;
            }
        }
        for (int r = 0; r < ff; ++r)
            if (!same(got_ff[r], ref_ff[r], "GU final", e, r)) return false;
        cpu::native_quant_h(f, ref_ff.data(), ref_h.data());
        cpu::native_quant_h(f, got_ff.data(), got_h.data());
        if (std::memcmp(ref_h.data(), got_h.data(), f.h_bytes)) {
            std::fprintf(stderr, "hidden quantized bytes differ e=%d ff=%d\n", e, ff);
            return false;
        }
        const void* hs[1] = {ref_h.data()};
        float* ds[1] = {ref_dn.data()};
        cpu::native_down_rows_split(f, down.data(), hs, 1, ds, 0, e);
        std::fill(got_dn.begin(), got_dn.end(), -123456.0f);
        for (const auto& range : dn_ranges) {
            const auto before = got_dn;
            if (!cpu::q2_pair_down_rows(f, down.data(), got_h.data(), got_dn.data(),
                                        range[0], range[1])) return false;
            for (int r = 0; r < e; ++r) {
                const float expected = r >= range[0] && r < range[1] ? ref_dn[r] : before[r];
                if (!same(got_dn[r], expected, "down boundary/tail", ff, r)) return false;
            }
        }
        for (int r = 0; r < e; ++r)
            if (!same(got_dn[r], ref_dn[r], "down final", ff, r)) return false;

        auto unsupported = f;
        unsupported.gu_type = GGML_TYPE_Q3_K;
        const auto before_ff = got_ff;
        if (cpu::q2_pair_gu_rows(unsupported, gate.data(), up.data(), aq.data(),
                                 got_ff.data(), 0, ff, limit) ||
            std::memcmp(before_ff.data(), got_ff.data(), ff * sizeof(float))) return false;
        unsupported = f;
        unsupported.d_type = GGML_TYPE_Q4_K;
        const auto before_dn = got_dn;
        if (cpu::q2_pair_down_rows(unsupported, down.data(), got_h.data(),
                                   got_dn.data(), 0, e) ||
            std::memcmp(before_dn.data(), got_dn.data(), e * sizeof(float))) return false;
    }
    return true;
}
}  // namespace

int main() {
    // The parent integrates optional dispatch in native_*_split. Keep those
    // calls as the unchanged oracle even if the shell inherited the flag.
#if defined(_WIN32)
    if (_putenv_s("MAYA_CPU_Q2_PAIR", "0") != 0) return 2;
#else
    if (setenv("MAYA_CPU_Q2_PAIR", "0", 1) != 0) return 2;
#endif
    if (!cpu::q2_pair_avx512_available()) {
        std::fprintf(stderr, "SKIP: Q2 pair AVX512 CPU/OS gate unavailable\n");
        return 77;
    }
    ggml_cpu_init();
    if (!pair_dots() || !split_rows(512, 256) || !split_rows(4096, 2048)) return 1;
    std::printf("PASS: exact paired Q2 dots, split GU/down, hidden bytes, odd tails and guards\n");
    return 0;
}
