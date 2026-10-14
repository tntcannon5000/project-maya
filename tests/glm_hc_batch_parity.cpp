// Isolate HC3 arithmetic from window scheduling: serial hc() versus hc_batch().
// No tolerance: all float bits, q8_1 bytes, and partial scratch must match.
#include "strata/kernels/glm_fast.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
namespace gf = strata::kernels::glmf;
constexpr size_t kE = 4096;
constexpr size_t kR = 4 * kE;
constexpr size_t kMix = 24;
constexpr size_t kQBytes = kE / 32 * 36; // block_q8_1: two half scalars + 32 int8 values
constexpr size_t kPart = gf::kHcBatchPartFloats;
constexpr size_t kGuard = 64;
constexpr uint8_t kGuardByte = 0xa5;
constexpr float kSentinel = -913.25f;

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

template<class T> uint64_t bits(T value) {
    uint64_t result = 0;
    static_assert(sizeof(T) <= sizeof(result));
    std::memcpy(&result, &value, sizeof(T));
    return result;
}

template<class T>
bool equal_bits(const char* label, const std::vector<T>& expected, const std::vector<T>& actual,
                size_t stride = 1, size_t first_row = 0) {
    if (expected.size() != actual.size()) throw std::runtime_error("comparison size mismatch");
    size_t count = 0, first = 0;
    double max_abs = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (bits(expected[i]) != bits(actual[i])) {
            if (count++ == 0) first = i;
            if constexpr (std::is_same_v<T, float>) {
                if (std::isfinite(expected[i]) && std::isfinite(actual[i]))
                    max_abs = std::max(max_abs, std::fabs(double(expected[i]) - double(actual[i])));
            }
        }
    }
    if (!count) return true;
    std::fprintf(stderr,
        "  %s: %zu differing elements; first row=%zu token=%lld element=%zu ref=0x%llx batch=0x%llx",
        label, count, first / stride, static_cast<long long>(first / stride) - static_cast<long long>(first_row),
        first % stride, static_cast<unsigned long long>(bits(expected[first])),
        static_cast<unsigned long long>(bits(actual[first])));
    if constexpr (std::is_same_v<T, float>) std::fprintf(stderr, " max_abs=%.9g", max_abs);
    std::fputc('\n', stderr);
    return false;
}

// Payloads start 64 bytes after cudaMalloc's aligned address. The complete
// allocation is downloaded once so every payload read also checks both guards.
template<class T> class Device {
    uint8_t* allocation_ = nullptr;
    std::string name_;
public:
    std::vector<T> initial;
    Device(std::string name, const std::vector<T>& values) : name_(std::move(name)), initial(values) {
        std::vector<uint8_t> bytes(kGuard + values.size() * sizeof(T) + kGuard, kGuardByte);
        std::memcpy(bytes.data() + kGuard, values.data(), values.size() * sizeof(T));
        check(cudaMalloc(reinterpret_cast<void**>(&allocation_), bytes.size()), "cudaMalloc");
        const auto err = cudaMemcpy(allocation_, bytes.data(), bytes.size(), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(allocation_);
            allocation_ = nullptr;
            check(err, "upload guarded allocation");
        }
    }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    ~Device() { if (allocation_) cudaFree(allocation_); }
    T* data() const { return reinterpret_cast<T*>(allocation_ + kGuard); }
    std::vector<T> read() const {
        const size_t payload = initial.size() * sizeof(T);
        std::vector<uint8_t> bytes(kGuard + payload + kGuard);
        check(cudaMemcpy(bytes.data(), allocation_, bytes.size(), cudaMemcpyDeviceToHost), "read guarded allocation");
        for (size_t i = 0; i < kGuard; ++i) {
            if (bytes[i] != kGuardByte || bytes[kGuard + payload + i] != kGuardByte)
                throw std::runtime_error(name_ + ": allocation guard changed at byte " + std::to_string(i));
        }
        std::vector<T> values(initial.size());
        std::memcpy(values.data(), bytes.data() + kGuard, payload);
        return values;
    }
    bool unchanged() const { return equal_bits(name_.c_str(), initial, read()); }
};

uint16_t bf16(float value) {
    uint32_t u = 0;
    std::memcpy(&u, &value, sizeof(u));
    return static_cast<uint16_t>((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

struct WeightHost {
    std::vector<uint16_t> fn;
    std::vector<float> scale, base, norm;
    explicit WeightHost(uint32_t seed) : fn(kMix * kR), scale(3), base(kMix), norm(kE) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> nd(0, 1);
        for (auto& v : fn) v = bf16(nd(rng) * .02f);
        for (auto& v : scale) v = 1.f + .1f * nd(rng);
        for (auto& v : base) v = .1f * nd(rng);
        for (auto& v : norm) v = 1.f + .05f * nd(rng);
    }
};

struct Weights {
    Device<uint16_t> fn;
    Device<float> scale, base, norm;
    explicit Weights(const WeightHost& w) : fn("weights.fn", w.fn), scale("weights.scale", w.scale),
        base("weights.base", w.base), norm("weights.norm", w.norm) {}
    bool unchanged() const {
        bool ok = fn.unchanged();
        ok &= scale.unchanged(); ok &= base.unchanged(); ok &= norm.unchanged();
        return ok;
    }
};

struct Case {
    uint32_t seed;
    int width;
    bool block, post_alias, comb_alias;
    size_t first, rows, qstride;
    std::vector<float> old, block_out, post_in, comb_in, new_r, x, pre, post, comb, part;
    std::vector<uint8_t> xq;
    Case(uint32_t seed_, int width_, bool block_, bool post_alias_, bool comb_alias_, bool padded)
        : seed(seed_), width(width_), block(block_), post_alias(post_alias_), comb_alias(comb_alias_),
          first(padded ? 2 : 0), rows(first + static_cast<size_t>(width) + 1),
          qstride(kQBytes + (padded ? 72 : 0)), old(rows * kR), block_out(rows * kE),
          post_in(rows * 8, kSentinel), comb_in(rows * 16), new_r(rows * kR, kSentinel),
          x(rows * kE, kSentinel), pre(rows * 8, kSentinel), post(rows * 8, kSentinel),
          comb(rows * 16, kSentinel), part(rows * kPart, kSentinel), xq(rows * qstride, 0x6d) {
        std::mt19937 rng(seed ^ (static_cast<uint32_t>(width) * 0x9e3779b9u));
        std::normal_distribution<float> nd(0, 1);
        for (auto& v : old) v = nd(rng);
        for (auto& v : block_out) v = .3f * nd(rng);
        for (size_t row = 0; row < rows; ++row) {
            for (size_t d = 0; d < 4; ++d) post_in[row * 8 + d] = 1.f + .1f * nd(rng);
            for (size_t d = 0; d < 4; ++d) {
                float sum = 0;
                for (size_t s = 0; s < 4; ++s) {
                    const float v = (d == s ? .8f : .05f) + .025f * std::fabs(nd(rng));
                    comb_in[row * 16 + d * 4 + s] = v;
                    sum += v;
                }
                for (size_t s = 0; s < 4; ++s) comb_in[row * 16 + d * 4 + s] /= sum;
            }
        }
        if (post_alias) post = post_in;
        if (comb_alias) comb = comb_in;
    }
    void describe() const {
        std::fprintf(stderr, "seed=%u T=%d block=%d post_alias=%d comb_alias=%d first=%zu qstride=%zu\n",
            seed, width, block, post_alias, comb_alias, first, qstride);
    }
};

struct Snapshot {
    std::vector<float> new_r, x, pre, post, comb, part;
    std::vector<uint8_t> xq;
};

struct State {
    Device<float> old, block, post_in, comb_in, new_r, x, pre, post, comb, scratch, part;
    Device<uint8_t> xq;
    Device<uint64_t> counter;
    State(const Case& c, const std::string& name)
        : old(name + ".R_old", c.old), block(name + ".block_out", c.block_out),
          post_in(name + ".post_in", c.post_in), comb_in(name + ".comb_in", c.comb_in),
          new_r(name + ".R_new", c.new_r), x(name + ".x", c.x), pre(name + ".pre", c.pre),
          post(name + ".post", c.post), comb(name + ".comb", c.comb),
          scratch(name + ".serial_scratch", std::vector<float>(kPart, kSentinel)),
          part(name + ".part_rows", c.part), xq(name + ".xq", c.xq),
          counter(name + ".counter", std::vector<uint64_t>(1, 0)) {}
    gf::HcArgs args(const Case& c, const Weights& w, size_t row, bool batch) const {
        gf::HcArgs a;
        a.block_out = c.block ? block.data() + row * kE : nullptr;
        a.R_old = old.data() + row * kR;
        a.R_new = new_r.data() + row * kR;
        a.post_in = c.post_alias ? post.data() + row * 8 : post_in.data() + row * 8;
        a.comb_in = c.comb_alias ? comb.data() + row * 16 : comb_in.data() + row * 16;
        a.pre = pre.data() + row * 8;
        a.post = post.data() + row * 8;
        a.comb = comb.data() + row * 16;
        a.w_fn = w.fn.data(); a.w_scale = w.scale.data(); a.w_base = w.base.data(); a.norm_w = w.norm.data();
        a.norm_eps = 1e-5f; a.hc_eps = 1e-6f; a.iters = 20; a.n_embd = static_cast<int>(kE);
        a.x = x.data() + row * kE;
        a.xq = xq.data() + row * c.qstride;
        a.part = batch ? part.data() + row * kPart : scratch.data();
        a.counter = batch ? nullptr : reinterpret_cast<unsigned int*>(counter.data());
        a.split_graph = false;
        return a;
    }
    Snapshot read() const {
        return {new_r.read(), x.read(), pre.read(), post.read(), comb.read(), part.read(), xq.read()};
    }
    bool read_only_unchanged() const {
        bool ok = old.unchanged();
        ok &= block.unchanged(); ok &= post_in.unchanged(); ok &= comb_in.unchanged();
        return ok;
    }
};

template<class T>
bool validate_writes(const char* label, const std::vector<T>& values, const std::vector<T>& initial,
                     const Case& c, size_t stride, size_t written) {
    for (size_t i = 0; i < values.size(); ++i) {
        const size_t row = i / stride, element = i % stride;
        const bool active = row >= c.first && row < c.first + static_cast<size_t>(c.width) && element < written;
        if (!active && bits(values[i]) != bits(initial[i])) {
            std::fprintf(stderr, "  %s: untouched row=%zu element=%zu changed\n", label, row, element);
            return false;
        }
        if constexpr (std::is_same_v<T, float>) {
            if (active && !std::isfinite(values[i])) {
                std::fprintf(stderr, "  %s: nonfinite output row=%zu element=%zu\n", label, row, element);
                return false;
            }
        }
    }
    return true;
}

bool validate(const Snapshot& s, const Case& c, const char* name) {
    bool ok = true;
    auto floats = [&](const char* field, const std::vector<float>& values, const std::vector<float>& initial,
                      size_t stride, size_t written) {
        const auto label = std::string(name) + "." + field;
        ok &= validate_writes(label.c_str(), values, initial, c, stride, written);
    };
    floats("R_new", s.new_r, c.new_r, kR, c.block ? kR : 0);
    floats("x", s.x, c.x, kE, kE);
    floats("pre", s.pre, c.pre, 8, 4);
    floats("post", s.post, c.post, 8, 4);
    floats("comb", s.comb, c.comb, 16, 16);
    floats("part", s.part, c.part, kPart, kPart);
    const auto label = std::string(name) + ".xq";
    ok &= validate_writes(label.c_str(), s.xq, c.xq, c, c.qstride, kQBytes);
    return ok;
}

// Returns 77 when ordinary hc() chooses HC1/HC2: comparing their different
// reduction tree would not establish HC3 batch parity.
int run_case(const Case& c, const Weights& w, cudaStream_t stream) {
    State serial(c, "serial"), batch(c, "batch");
    const int errors_before = gf::launch_errors();
    if (!gf::hc_batch(batch.args(c, w, c.first, true), c.width, c.qstride, stream)) {
        std::fprintf(stderr, "SKIP: hc_batch unavailable (ordinary hc() must select HC3; check GPU and STRATA_GLM_HC1/HC2).\n");
        return 77;
    }
    check(cudaStreamSynchronize(stream), "batch HC synchronize");
    for (int token = 0; token < c.width; ++token) {
        const size_t row = c.first + static_cast<size_t>(token);
        gf::hc(serial.args(c, w, row, false), stream);
        // The production serial path reuses one partial arena. Preserve each
        // row's result before the next hc() overwrites it.
        check(cudaMemcpyAsync(serial.part.data() + row * kPart, serial.scratch.data(),
            kPart * sizeof(float), cudaMemcpyDeviceToDevice, stream), "snapshot serial HC scratch");
    }
    check(cudaStreamSynchronize(stream), "serial HC synchronize");
    if (gf::launch_errors() != errors_before) throw std::runtime_error("HC launch error count increased");
    const auto ref = serial.read(), got = batch.read();
    bool ok = equal_bits("R_new", ref.new_r, got.new_r, kR, c.first);
    ok &= equal_bits("x", ref.x, got.x, kE, c.first);
    ok &= equal_bits("xq bytes", ref.xq, got.xq, c.qstride, c.first);
    ok &= equal_bits("pre", ref.pre, got.pre, 8, c.first);
    ok &= equal_bits("post", ref.post, got.post, 8, c.first);
    ok &= equal_bits("comb", ref.comb, got.comb, 16, c.first);
    ok &= equal_bits("part", ref.part, got.part, kPart, c.first);
    ok &= validate(ref, c, "serial"); ok &= validate(got, c, "batch");
    ok &= serial.read_only_unchanged(); ok &= batch.read_only_unchanged();
    ok &= batch.scratch.unchanged(); ok &= batch.counter.unchanged();
    const std::vector<uint64_t> expected_counter(1, 64ull * static_cast<uint64_t>(c.width));
    ok &= equal_bits("serial.counter", expected_counter, serial.counter.read());
    // Read the reused scratch to check its guards and final row snapshot too.
    const auto last_scratch = serial.scratch.read();
    const size_t last = (c.first + static_cast<size_t>(c.width) - 1) * kPart;
    const std::vector<float> last_snapshot(ref.part.begin() + last, ref.part.begin() + last + kPart);
    ok &= equal_bits("serial.final_scratch", last_snapshot, last_scratch);
    ok &= w.unchanged();
    if (!ok) c.describe();
    return ok ? 0 : 1;
}

struct Stream {
    cudaStream_t value = nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "cudaStreamCreateWithFlags"); }
    ~Stream() { if (value) cudaStreamDestroy(value); }
};
} // namespace

int main(int argc, char** argv) {
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--selftest") != 0)) {
        std::fprintf(stderr, "usage: glm_hc_batch_parity [--selftest]\n");
        return 2;
    }
    try {
        int count = 0;
        const auto err = cudaGetDeviceCount(&count);
        if (err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver || (err == cudaSuccess && count == 0)) {
            std::fprintf(stderr, "SKIP: no usable CUDA device.\n");
            return 77;
        }
        check(err, "cudaGetDeviceCount");
        Stream stream;
        size_t cases = 0, tokens = 0;
        for (uint32_t seed : {0x4d415941u, 0x48433342u}) {
            const WeightHost host(seed);
            const Weights weights(host);
            for (int width : {2, 3, 5, 8}) {
                size_t width_cases = 0;
                for (bool block : {false, true})
                    for (bool post_alias : {false, true})
                        for (bool comb_alias : {false, true})
                            for (bool padded : {false, true}) {
                                const Case c(seed, width, block, post_alias, comb_alias, padded);
                                int result;
                                try {
                                    result = run_case(c, weights, stream.value);
                                } catch (...) {
                                    c.describe();
                                    throw;
                                }
                                if (result != 0) return result;
                                ++width_cases; ++cases; tokens += static_cast<size_t>(width);
                            }
                std::printf("HC3 exact parity seed=%u T=%d: %zu cases passed\n", seed, width, width_cases);
                std::fflush(stdout);
            }
        }
        std::printf("PASS: %zu cases, %zu token rows; all output bits, partial scratch, guards, and untouched regions match.\n",
            cases, tokens);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
