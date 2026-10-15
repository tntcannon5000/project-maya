// Fixed-input verification windows, partial commits and repeated-request state reuse.
#include "../src/core/glm_fast_state.hpp"
#include <cuda_profiler_api.h>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <string>
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
    const char* mode_env = std::getenv("MAYA_VERIFY_REFERENCE_MODE");
    const std::string reference_mode = mode_env ? mode_env : "windows";
    if (reference_mode != "windows" && reference_mode != "forward" && reference_mode != "window_first") return 2;
    if (repeats < 1 || repeats > 2 || steps < 1 || steps > 14) return 2;
    if (width_a < 2 || width_a > 8 || width_b < 2 || width_b > 8) return 2;
    std::fprintf(stderr, "verify fixture: repeats=%d, max windows/repeat=%d, widths=%d/%d, occupied prefix=%d\n", repeats, steps, width_a, width_b, prefix);
    std::fprintf(stderr, "verify fixture: reference mode=%s (forward/window_first advance one fixed input per step)\n", reference_mode.c_str());
    for (int rep = 0; rep < repeats; ++rep) {
        m.reset();
        if (batch_prefill && prefix > 1) {
            if (!m.prefill(std::vector<int32_t>(ids.begin(), ids.begin()+prefix-1), err, ids[prefix-1])) {
                std::fprintf(stderr, "batched fixture prefill failed or unavailable: %s\n", err.c_str()); return 5;
            }
            if (std::getenv("MAYA_VERIFY_SAVE_PREFILL_SEAMS") != nullptr) {
                // Preserve the existing diagnostic seams before forward() can
                // overwrite their filenames with the final token's activations.
                const char* source = std::getenv("GLM_CB_DIR");
                if (!source || !*source) return 13;
                try {
                    const auto target = std::filesystem::path(std::string(argv[3]) + ".prefill-" + std::to_string(rep));
                    std::filesystem::create_directories(target);
                    int copied = 0;
                    for (const auto& entry : std::filesystem::directory_iterator(source)) {
                        if (entry.is_regular_file() && entry.path().extension() == ".f32") {
                            std::filesystem::copy_file(entry.path(), target / entry.path().filename(),
                                std::filesystem::copy_options::overwrite_existing);
                            ++copied;
                        }
                    }
                    if (!copied) return 13;
                    std::fprintf(stderr, "verify prefill seams: repeat=%d files=%d path=%s\n", rep, copied, target.string().c_str());
                } catch (const std::filesystem::filesystem_error& e) {
                    std::fprintf(stderr, "prefill seam preservation: %s\n", e.what()); return 13;
                }
            }
            if (!m.forward({ids[prefix-1]}, logits, err)) return 5;
            std::fprintf(stderr, "verify fixture: completed batched prefix %d, repeat %d\n", prefix, rep);
        } else if (!m.forward(std::vector<int32_t>(ids.begin(), ids.begin()+prefix), logits, err)) return 5;
        if (std::getenv("MAYA_VERIFY_PREFIX_LOGITS") != nullptr) {
            const std::string prefix_path = std::string(argv[3]) + ".prefix.f32";
            FILE* pf = std::fopen(prefix_path.c_str(), rep == 0 ? "wb" : "ab");
            if (!pf || logits.size() != (size_t)m.geometry().n_vocab) return 11;
            const bool written = std::fwrite(logits.data(), sizeof(float), logits.size(), pf) == logits.size();
            std::fclose(pf);
            if (!written) return 11;
        }
        if (std::getenv("MAYA_VERIFY_PREFIX_STATE") != nullptr) {
            // Diagnostic only: use the engine's conversation serializer to
            // retain recurrent state and occupied DSA caches before verification.
            // Snapshot allocation/copies perturb memory/timing; never use this
            // mode as throughput evidence. Prefix logits were recorded above.
            size_t free_before = 0, total = 0, free_after = 0;
            if (cudaMemGetInfo(&free_before, &total) != cudaSuccess) return 12;
            if (!m.snapshot_save()) { std::fprintf(stderr, "prefix snapshot failed\n"); return 12; }
            const std::string path = std::string(argv[3]) + ".prefix-" + std::to_string(rep) + ".slot";
            const uint64_t bytes = m.slot_save(path, err);
            if (!bytes) { std::fprintf(stderr, "prefix state save: %s\n", err.c_str()); return 12; }
            if (cudaMemGetInfo(&free_after, &total) != cudaSuccess) return 12;
            std::fprintf(stderr, "verify prefix state: repeat=%d bytes=%llu free_before=%llu free_after=%llu path=%s\n",
                rep, (unsigned long long)bytes, (unsigned long long)free_before,
                (unsigned long long)free_after, path.c_str());
        }
        if (reference_mode != "windows") {
            if (logits.size() != (size_t)m.geometry().n_vocab) return 7;
            for (float v : logits) if (!std::isfinite(v)) return 8;
            if (std::fwrite(logits.data(),sizeof(float),logits.size(),f)!=logits.size()) return 9;
        }
        for (int pos=prefix, step=0; pos < prefix+64 && step < steps; ++step) {
            if (trace && rep == 1 && step == 0) cudaProfilerStart();
            const int T = step % 2 == 0 ? width_a : width_b;
            int32_t picks[8];
            const auto call_start = std::chrono::steady_clock::now();
            if (reference_mode == "forward") {
                if (!m.forward({ids[(size_t)pos]}, logits, err)) {
                    std::fprintf(stderr, "%s\n", err.c_str()); return 6;
                }
            } else if (!m.window(ids.data()+pos, T, picks, err)) {
                std::fprintf(stderr, "%s\n", err.c_str()); return 6;
            }
            const double call_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - call_start).count();
            std::fprintf(stderr, "verify call: repeat=%d step=%d width=%d position=%d ms=%.6f\n",
                         rep, step, reference_mode == "forward" ? 1 : T, pos, call_ms);
            const int rows = reference_mode == "windows" ? T : 1;
            if (reference_mode != "forward") {
                logits.resize((size_t)rows*m.geometry().n_vocab);
                if (cudaMemcpy(logits.data(), m.fast_->win.logits, logits.size()*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return 7;
            }
            if (logits.size() != (size_t)rows*m.geometry().n_vocab) return 7;
            for (float v:logits) if (!std::isfinite(v)) return 8;
            if (std::fwrite(logits.data(),sizeof(float),logits.size(),f)!=logits.size()) return 9;
            const int keep = reference_mode == "windows" ? (step % 3 == 0 ? 1 : T) : 1;
            if (reference_mode != "forward" && !m.window_commit(keep,err)) return 10;
            if (trace && rep == 1 && step == 3) cudaProfilerStop();
            pos += keep;
        }
        m.win_prof_report();
    }
    std::fclose(f);
    std::printf("PASS finite logits (mode=%s, %d repeat(s), max %d steps/repeat)\n", reference_mode.c_str(), repeats, steps);
}
