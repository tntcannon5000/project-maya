// Fixed-input verification windows, partial commits and repeated-request state reuse.
#include "../src/core/glm_fast_state.hpp"
#include <cuda_profiler_api.h>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4) return 2; // PACK TOKEN_FILE OUTPUT
    std::ifstream input(argv[2]); std::vector<int32_t> ids;
    int32_t id;
    while (input >> id) ids.push_back(id);
    const char* prefix_env = std::getenv("MAYA_VERIFY_PREFIX_TOKENS");
    const int prefix = prefix_env ? std::atoi(prefix_env) : 40;
    const bool batch_prefill = std::getenv("MAYA_VERIFY_BATCH_PREFILL") != nullptr &&
                               std::atoi(std::getenv("MAYA_VERIFY_BATCH_PREFILL")) != 0;
    if (prefix < 1 || prefix > 131000 || ids.size() < (size_t) prefix + 72) return 2;
    strata::core::Glm5Model m; std::string err;
    m.set_expert_profile("data/expert-profile-glm.bin");
    if (!m.load_pack_env(argv[1], 131072, err)) {
        std::fprintf(stderr, "%s\n", err.c_str()); return 3;
    }
    FILE* f = std::fopen(argv[3], "wb");
    if (f == nullptr) return 4;
    std::vector<float> logits;
    const bool trace = std::getenv("MAYA_VERIFY_TRACE") != nullptr;
    const char* reps_env = std::getenv("MAYA_VERIFY_REPEATS");
    const char* steps_env = std::getenv("MAYA_VERIFY_STEPS");
    const char* wa_env = std::getenv("MAYA_VERIFY_WIDTH_A");
    const char* wb_env = std::getenv("MAYA_VERIFY_WIDTH_B");
    const int width_a = wa_env ? std::atoi(wa_env) : 5;
    const int width_b = wb_env ? std::atoi(wb_env) : 8;
    const int repeats = reps_env ? std::atoi(reps_env) : 2;
    const int steps = steps_env ? std::atoi(steps_env) : 14;
    if (repeats < 1 || repeats > 2 || steps < 1 || steps > 14) return 2;
    if (width_a < 2 || width_a > 8 || width_b < 2 || width_b > 8) return 2;
    std::fprintf(stderr, "verify fixture: repeats=%d, max windows/repeat=%d, widths=%d/%d, occupied prefix=%d\n", repeats, steps, width_a, width_b, prefix);
    for (int rep = 0; rep < repeats; ++rep) {
        m.reset();
        if (batch_prefill && prefix > 1) {
            if (!m.prefill(std::vector<int32_t>(ids.begin(), ids.begin()+prefix-1), err, ids[prefix-1])) {
                std::fprintf(stderr, "batched fixture prefill failed or unavailable: %s\n", err.c_str()); return 5;
            }
            if (!m.forward({ids[prefix-1]}, logits, err)) return 5;
            std::fprintf(stderr, "verify fixture: completed batched prefix %d, repeat %d\n", prefix, rep);
        } else if (!m.forward(std::vector<int32_t>(ids.begin(), ids.begin()+prefix), logits, err)) return 5;
        for (int pos=prefix, step=0; pos < prefix+64 && step < steps; ++step) {
            if (trace && rep == 1 && step == 0) cudaProfilerStart();
            const int T = step % 2 == 0 ? width_a : width_b;
            int32_t picks[8];
            if (!m.window(ids.data()+pos, T, picks, err)) {
                std::fprintf(stderr, "%s\n", err.c_str()); return 6;
            }
            logits.resize((size_t)T*m.geometry().n_vocab);
            if (cudaMemcpy(logits.data(), m.fast_->win.logits, logits.size()*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return 7;
            for (float v:logits) if (!std::isfinite(v)) return 8;
            if (std::fwrite(logits.data(),sizeof(float),logits.size(),f)!=logits.size()) return 9;
            const int keep = step % 3 == 0 ? 1 : T;
            if (!m.window_commit(keep,err)) return 10;
            if (trace && rep == 1 && step == 3) cudaProfilerStop();
            pos += keep;
        }
        m.win_prof_report();
    }
    std::fclose(f);
    std::printf("PASS finite logits and partial commits (%d repeat(s), max %d windows/repeat)\n", repeats, steps);
}
