// Compare the trial against the *linked engine's* mv(), with no numerical tolerance.
#include "prototype.cuh"
#include <cuda_fp16.h>
#define GGML_COMMON_DECL_CUDA
#include "ggml-common.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace gf = strata::kernels::glmf;
namespace {
void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct Buffer {
    void* p = nullptr;
    explicit Buffer(size_t n) { check(cudaMalloc(&p, n)); }
    ~Buffer() { if (p) cudaFree(p); }
    Buffer(const Buffer&) = delete;
};
uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
std::vector<block_q4_K> weights(int ni, int no, int seed) {
    std::vector<block_q4_K> w((size_t) no * (ni / 256));
    const float scales[] = {0.0f, 0.00006103515625f, 0.00390625f, 0.125f, 1.0f, 8.0f, -0.03125f};
    for (size_t b = 0; b < w.size(); ++b) {
        auto& v = w[b];
        v.dm = __floats2half2_rn(scales[(b + seed) % 7], scales[(b * 3 + seed + 2) % 7]);
        for (int j = 0; j < 12; ++j) {
            // Exercise every high-bit scale/min decoder, including extrema.
            v.scales[j] = seed == 0 ? 0 : seed == 1 ? 255 : (uint8_t) mix(b * 17 + j + 37 * seed);
        }
        for (int j = 0; j < 128; ++j)
            v.qs[j] = seed == 1 ? (j & 1 ? 0xf0 : 0x0f) : (uint8_t) mix(91 + b * 131 + j + seed);
    }
    return w;
}
std::vector<block_q8_1> activations(int ni, int nc, int seed, int stride_blocks) {
    std::vector<block_q8_1> x((size_t) nc * stride_blocks);
    for (auto& v : x) {
        v.ds = __floats2half2_rn(64.0f, 123.0f);
        for (auto& q : v.qs) q = 127; // nonzero poison in activation-stride padding
    }
    const float ds[] = {0.0f, 0.00006103515625f, 0.001953125f, 0.125f, 1.0f, 16.0f};
    for (int c = 0; c < nc; ++c) for (int b = 0; b < ni / 32; ++b) {
        auto& v = x[(size_t) c * stride_blocks + b];
        // ds.y is deliberately unrelated. Neither exact implementation may use it.
        v.ds = __floats2half2_rn(ds[(c * 3 + b + seed) % 6], -37.0f);
        for (int j = 0; j < 32; ++j) {
            const int r = (int) (mix(c * 7919 + b * 37 + j + seed * 131) % 255) - 127;
            v.qs[j] = seed == 1 ? (j & 1 ? 127 : -127) : (int8_t) r;
        }
    }
    return x;
}
void engine(const gf::MvJob& j) {
    if (!gf::mv(&j, 1, nullptr)) throw std::runtime_error("engine mv refused job");
    check(cudaGetLastError());
}
void trial(const gf::MvJob& j) {
    if (!imma_q4_exact_trial(j, nullptr)) throw std::runtime_error("trial refused job or launch failed");
}

bool parity(int ni, int no, int nc, int seed) {
    const int xb = ni / 32 + 3, ys = no + 11;
    auto wh = weights(ni, no, seed);
    auto xh = activations(ni, nc, seed, xb);
    std::vector<float> bias(no);
    for (int r = 0; r < no; ++r) bias[r] = (r % 13 - 6) * 0.03125f;
    Buffer w(wh.size() * sizeof(wh[0])), x(xh.size() * sizeof(xh[0]));
    Buffer b(no * sizeof(float)), ref((size_t) nc * ys * 4), out((size_t) nc * ys * 4);
    Buffer one((size_t) nc * ys * 4);
    check(cudaMemcpy(w.p, wh.data(), wh.size() * sizeof(wh[0]), cudaMemcpyHostToDevice));
    check(cudaMemcpy(x.p, xh.data(), xh.size() * sizeof(xh[0]), cudaMemcpyHostToDevice));
    check(cudaMemcpy(b.p, bias.data(), no * 4, cudaMemcpyHostToDevice));
    // Identical canaries verify output padding as well as the valid matrix.
    check(cudaMemset(ref.p, 0x5a, (size_t) nc * ys * 4));
    check(cudaMemset(out.p, 0x5a, (size_t) nc * ys * 4));
    check(cudaMemset(one.p, 0x5a, (size_t) nc * ys * 4));
    gf::MvJob j;
    j.w = w.p; j.xq = x.p; j.y = (float*) ref.p;
    j.bias = seed & 1 ? (float*) b.p : nullptr;
    j.alpha = seed & 1 ? -0.75f : 1.0f;
    j.type = 12; j.n_in = ni; j.n_out = no; j.ncol = nc;
    j.xq_cstride = (int64_t) xb * sizeof(block_q8_1); j.y_cstride = ys;
    engine(j);
    gf::MvJob tj = j; tj.y = (float*) out.p; trial(tj);
    for (int c = 0; c < nc; ++c) {
        gf::MvJob sj = j;
        sj.ncol = 1; sj.xq = (const char*) x.p + c * j.xq_cstride;
        sj.y = (float*) one.p + c * ys;
        engine(sj);
    }
    std::vector<float> rh((size_t) nc * ys), oh(rh.size()), sh(rh.size());
    check(cudaMemcpy(rh.data(), ref.p, rh.size() * 4, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(oh.data(), out.p, oh.size() * 4, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(sh.data(), one.p, sh.size() * 4, cudaMemcpyDeviceToHost));
    for (int c = 0; c < nc; ++c) for (int r = 0; r < ys; ++r) {
        const size_t at = (size_t) c * ys + r;
        if (!std::isfinite(rh[at]) || std::memcmp(&rh[at], &oh[at], 4) ||
            std::memcmp(&rh[at], &sh[at], 4)) {
            std::printf("FAIL ni=%d no=%d nc=%d seed=%d c=%d r=%d engine=%a trial=%a one=%a\n",
                        ni, no, nc, seed, c, r, rh[at], oh[at], sh[at]);
            return false;
        }
    }
    return true;
}

void timing(int ni, int no, int nc, const cudaDeviceProp& prop) {
    auto wh = weights(ni, no, 2);
    auto xh = activations(ni, nc, 3, ni / 32);
    const size_t bytes = wh.size() * sizeof(wh[0]);
    // Distinct address ring >2x reported L2; do not time one L2-resident matrix
    // fifty times and call it the model's streaming bandwidth.
    const int slots = std::max(2, (int) ((2ull * prop.l2CacheSize) / bytes) + 1);
    Buffer w(bytes * slots), x(xh.size() * sizeof(xh[0])), out((size_t) nc * no * 4);
    for (int i = 0; i < slots; ++i)
        check(cudaMemcpy((char*) w.p + i * bytes, wh.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(x.p, xh.data(), xh.size() * sizeof(xh[0]), cudaMemcpyHostToDevice));
    gf::MvJob j;
    j.xq = x.p; j.y = (float*) out.p; j.type = 12;
    j.n_in = ni; j.n_out = no; j.ncol = nc;
    j.xq_cstride = (int64_t) ni / 32 * sizeof(block_q8_1); j.y_cstride = no;
    cudaEvent_t begin, end;
    check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
    const auto run = [&](bool use_trial) {
        // Two full ring traversals before recording, with identical addresses.
        for (int it = 0; it < 2 * slots; ++it) {
            j.w = (char*) w.p + (it % slots) * bytes;
            if (use_trial) trial(j); else engine(j);
        }
        check(cudaEventRecord(begin));
        const int reps = slots * 8;
        for (int it = 0; it < reps; ++it) {
            j.w = (char*) w.p + (it % slots) * bytes;
            if (use_trial) trial(j); else engine(j);
        }
        check(cudaEventRecord(end)); check(cudaEventSynchronize(end));
        float ms = 0; check(cudaEventElapsedTime(&ms, begin, end));
        return 1000.0f * ms / reps;
    };
    const float b0 = run(false), t0 = run(true), t1 = run(true), b1 = run(false);
    std::printf("TIMING %dx%d nc=%d ring=%d %.1fMiB L2=%.1fMiB engine_us=%.3f/%.3f trial_us=%.3f/%.3f\n",
                no, ni, nc, slots, bytes * slots / 1048576.0, prop.l2CacheSize / 1048576.0, b0, b1, t0, t1);
    check(cudaEventDestroy(begin)); check(cudaEventDestroy(end));
}
} // namespace

int main(int argc, char** argv) try {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    const bool model_shapes = argc > 1 && std::string(argv[1]) == "--model-shapes";
    if (argc > 1 && !bench && !model_shapes)
        throw std::runtime_error("usage: imma_q4_feasibility [--bench|--model-shapes]");
#ifdef _WIN32
    _putenv_s("STRATA_GLM_MV_TILED", "1");
    _putenv_s("STRATA_GLM_MV_GENERIC", "");
    _putenv_s("STRATA_GLM_MV_APPROX", "0");
#else
    setenv("STRATA_GLM_MV_TILED", "1", 1); unsetenv("STRATA_GLM_MV_GENERIC");
    setenv("STRATA_GLM_MV_APPROX", "0", 1);
#endif
    cudaDeviceProp prop; check(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 8) throw std::runtime_error("trial requires NVIDIA sm_80 or newer");
    std::printf("Device %s sm_%d%d, strict bitwise comparison against linked engine archive\n",
                prop.name, prop.major, prop.minor);
    int count = 0;
    // Odd/even superblock counts, K tails across warp assignment, row tails,
    // nondefault strides, extrema/random scales and both activation signs.
    for (int ni : {256, 512, 768, 4096, 8192, 12288})
        for (int no : {1, 15, 16, 17, 67}) for (int nc : {3, 5}) for (int seed : {0, 1, 2, 3}) {
            if (!parity(ni, no, nc, seed)) return 1;
            ++count;
        }
    if (gf::launch_errors()) throw std::runtime_error("linked engine recorded launch errors");
    std::printf("PASS %d cases: all valid floats and padding bitwise equal to mvc and singleton mv\n", count);
    if (bench) for (int ni : {4096, 8192, 12288}) for (int nc : {3, 5}) timing(ni, 8192, nc, prop);
    if (model_shapes) {
        for (int ni : {8192, 12288, 16384}) for (int seed : {2, 3})
            if (!parity(ni, 4096, 5, seed)) return 1;
        std::printf("PASS six additional exact model-shape cases (T5, N4096)\n");
        for (int ni : {8192, 12288, 16384}) timing(ni, 4096, 5, prop);
    }
    check(cudaDeviceSynchronize());
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what()); return 2;
}
