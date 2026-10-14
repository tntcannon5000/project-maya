// Fixed-input verification windows, partial commits and repeated-request state reuse.
#include "../src/core/glm_fast_state.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4) return 2; // PACK TOKEN_FILE OUTPUT
    std::ifstream input(argv[2]); std::vector<int32_t> ids;
    int32_t id;
    while (input >> id) ids.push_back(id);
    if (ids.size() < 120) return 2;
    strata::core::Glm5Model m; std::string err;
    m.set_expert_profile("data/expert-profile-glm.bin");
    if (!m.load_pack_env(argv[1], 131072, err)) {
        std::fprintf(stderr, "%s\n", err.c_str()); return 3;
    }
    FILE* f = std::fopen(argv[3], "wb");
    if (f == nullptr) return 4;
    std::vector<float> logits;
    for (int rep = 0; rep < 2; ++rep) {
        m.reset();
        if (!m.forward(std::vector<int32_t>(ids.begin(), ids.begin()+40), logits, err)) return 5;
        for (int pos=40, step=0; pos < 104; ++step) {
            const int T = step % 2 == 0 ? 5 : 8;
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
            pos += keep;
        }
        m.win_prof_report();
    }
    std::fclose(f);
    std::puts("PASS finite logits, partial commits, reset and repeat");
}
