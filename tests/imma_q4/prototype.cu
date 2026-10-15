// Research prototype: preserve DecW<12>'s *eight-value* integer dots, its
// two-subblock FP expression, its lane-strided K order and XOR32 reduction.
// INT8 MMA alone does not imply the same floating-point answer.
#include "prototype.cuh"
#include <cuda_fp16.h>
#define GGML_COMMON_DECL_CUDA
#include "ggml-common.h"

namespace {
using Job = strata::kernels::glmf::MvJob;

__device__ __forceinline__ void scale_min(const block_q4_K& w, int j, int& s, int& m) {
    if (j < 4) {
        s = w.scales[j] & 63;
        m = w.scales[j + 4] & 63;
    } else {
        s = (w.scales[j + 4] & 15) | ((w.scales[j - 4] >> 6) << 4);
        m = (w.scales[j + 4] >> 4) | ((w.scales[j] >> 6) << 4);
    }
}

template<int NC>
__global__ __launch_bounds__(256) void exact_trial_kernel(Job J) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int NT = (4 * NC + 7) / 8;
    // Logical lane = block_parity*16 + subblock_pair*4 + quarter.
    __shared__ float partial[16][NC][32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, tig = lane & 3;
    const int parity = warp >> 2, pair = warp & 3;
    const int row0 = blockIdx.x * 16;
    const int nb = J.n_in / 256;
    const auto* wa = (const block_q4_K*) J.w + (size_t) min(row0 + g, J.n_out - 1) * nb;
    const auto* wb = (const block_q4_K*) J.w + (size_t) min(row0 + g + 8, J.n_out - 1) * nb;
    float acc[NT][4] = {};

    // Same superblocks as original lane: kb = parity, parity+2, ... .
    for (int kb = parity; kb < nb; kb += 2) {
        const int2 qa = *(const int2*) (wa[kb].qs + 32 * pair + 8 * tig);
        const int2 qb = *(const int2*) (wb[kb].qs + 32 * pair + 8 * tig);
        const float2 dma = __half22float2(wa[kb].dm), dmb = __half22float2(wb[kb].dm);
        float sd[NT][4] = {}, sm[NT][4] = {};
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int j = 2 * pair + i, shift = 4 * i;
            const int a0 = (qa.x >> shift) & 0x0f0f0f0f;
            const int a2 = (qa.y >> shift) & 0x0f0f0f0f;
            const int a1 = (qb.x >> shift) & 0x0f0f0f0f;
            const int a3 = (qb.y >> shift) & 0x0f0f0f0f;
            int sa, ma, sb, mb;
            scale_min(wa[kb], j, sa, ma);
            scale_min(wb[kb], j, sb, mb);
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                // A virtual column is (real token, original eight-value quarter).
                // Original quarter q selects packed q8 words q and q+4.
                const int v = 8 * t + g;
                const int token = min(v / 4, NC - 1), quarter = v & 3;
                const auto* xg = (const block_q8_1*) ((const char*) J.xq + token * J.xq_cstride);
                const int* xu = (const int*) xg[kb * 8 + j].qs;
                const int b0 = ((2 * tig) % 4 == quarter) ? xu[2 * tig] : 0;
                const int b1 = ((2 * tig + 1) % 4 == quarter) ? xu[2 * tig + 1] : 0;
                int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                // Integer sum of exactly the same eight q8 values, never ds.y.
                int su = __dp4a(b1, 0x01010101, __dp4a(b0, 0x01010101, 0));
                su += __shfl_xor_sync(0xffffffffu, su, 1);
                su += __shfl_xor_sync(0xffffffffu, su, 2);
                const int s0 = __shfl_sync(0xffffffffu, su, 8 * tig);
                const int s1 = __shfl_sync(0xffffffffu, su, 8 * tig + 4);
                const int token0 = min((8 * t + 2 * tig) / 4, NC - 1);
                const int token1 = min((8 * t + 2 * tig + 1) / 4, NC - 1);
                const auto* x0 = (const block_q8_1*) ((const char*) J.xq + token0 * J.xq_cstride);
                const auto* x1 = (const block_q8_1*) ((const char*) J.xq + token1 * J.xq_cstride);
                const float d0 = __low2float(x0[kb * 8 + j].ds);
                const float d1 = __low2float(x1[kb * 8 + j].ds);
                // Keep integer scale products and the original FP associations.
                sd[t][0] += d0 * (c0 * sa); sm[t][0] += d0 * (s0 * ma);
                sd[t][1] += d1 * (c1 * sa); sm[t][1] += d1 * (s1 * ma);
                sd[t][2] += d0 * (c2 * sb); sm[t][2] += d0 * (s0 * mb);
                sd[t][3] += d1 * (c3 * sb); sm[t][3] += d1 * (s1 * mb);
            }
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            acc[t][0] += dma.x * sd[t][0] - dma.y * sm[t][0];
            acc[t][1] += dma.x * sd[t][1] - dma.y * sm[t][1];
            acc[t][2] += dmb.x * sd[t][2] - dmb.y * sm[t][2];
            acc[t][3] += dmb.x * sd[t][3] - dmb.y * sm[t][3];
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int v = 8 * t + 2 * tig + c;
            if (v < 4 * NC) {
                const int logical = parity * 16 + pair * 4 + (v & 3);
                partial[g][v / 4][logical] = acc[t][c];
                partial[g + 8][v / 4][logical] = acc[t][c + 2];
            }
        }
    }
    __syncthreads();
    // Restore the original 32 logical lanes before the original XOR tree.
    for (int output = warp; output < 16 * NC; output += 8) {
        const int r = output / NC, c = output % NC;
        float s = partial[r][c][lane];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            s += __shfl_xor_sync(0xffffffffu, s, offset);
        if (lane == 0 && row0 + r < J.n_out) {
            const float bias = J.bias ? J.bias[row0 + r] : 0.0f;
            J.y[c * J.y_cstride + row0 + r] = __fadd_rn(__fmul_rn(J.alpha, s), bias);
        }
    }
#endif
}
} // namespace

bool imma_q4_exact_trial(const Job& job, cudaStream_t stream) {
    if (job.type != 12 || (job.ncol != 3 && job.ncol != 5) || job.n_in <= 0 ||
        job.n_in % 256 || job.n_out <= 0 || !job.w || !job.xq || !job.y ||
        job.xq_cstride < (int64_t) job.n_in / 32 * 36 || job.y_cstride < job.n_out)
        return false;
    const int blocks = (job.n_out + 15) / 16;
    if (job.ncol == 3) exact_trial_kernel<3><<<blocks, 256, 0, stream>>>(job);
    else exact_trial_kernel<5><<<blocks, 256, 0, stream>>>(job);
    return cudaPeekAtLastError() == cudaSuccess;
}
