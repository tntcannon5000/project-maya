// Fixed-input decode and logits recorder. No sampler or speculative path.
#include "strata/core/glm_model.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#ifdef MAYA_BASELINE_DRIVER
#define LAB_METRIC(stats,field) 0ull
#else
#define LAB_METRIC(stats,field) ((stats).field)
#endif

int main(int argc, char** argv) {
    if(argc != 6) { std::fprintf(stderr,"PACK TOKEN_FILE PROMPT_COUNT OUTPUT_PREFIX REPEATS\n"); return 2; }
    std::ifstream input(argv[2]); std::vector<int32_t> ids;
    int32_t id; while(input >> id) ids.push_back(id);
    const int prompt=std::atoi(argv[3]), repeats=std::atoi(argv[5]);
    if(prompt<1 || prompt >= (int)ids.size() || repeats<1) return 2;
    strata::core::Glm5Model m; std::string err;
    m.set_expert_profile("data/expert-profile-glm.bin");
    if(!m.load_pack_env(argv[1],131072,err)) { std::fprintf(stderr,"%s\n",err.c_str());return 1; }
    if(m.has_mtp() || m.n_parts_!=1) return 3;
    const std::string prefix=argv[4];
    FILE* times=std::fopen((prefix+".csv").c_str(),"w");
    FILE* dump=std::fopen((prefix+".logits.f32").c_str(),"wb");
    if(!times || !dump) return 4;
    std::fprintf(times,"repeat,position,input,ms,hits,ram_fetches,cpu_experts,promotions,disk_reads,pool_slots,ram_slots,pool_gib,ram_gib,fetch_bytes,demotion_bytes,demotions,shadow_hits,shadow_avoided_bytes,predicted_bytes,predicted_used_bytes,dma_bytes\n");
    std::vector<float> logits;
    using Clock=std::chrono::steady_clock;
    for(int r=0;r<repeats;++r) {
        m.reset();
        auto t=Clock::now();
        if(prompt>=128) {
            if(!m.prefill(std::vector<int32_t>(ids.begin(),ids.begin()+prompt-1),err,ids[prompt-1]) ||
               !m.forward({ids[prompt-1]},logits,err)) { std::fprintf(stderr,"%s\n",err.c_str());return 5; }
        } else if(!m.forward(std::vector<int32_t>(ids.begin(),ids.begin()+prompt),logits,err)) return 5;
        std::fprintf(stderr,"prefill repeat=%d tokens=%d ms=%.6f\n",r,prompt,
            std::chrono::duration<double,std::milli>(Clock::now()-t).count());
        for(size_t i=prompt;i<ids.size();++i) {
            const auto before=m.fast_stats(); t=Clock::now();
            if(!m.forward({ids[i]},logits,err)) { std::fprintf(stderr,"%s\n",err.c_str());return 6; }
            const double ms=std::chrono::duration<double,std::milli>(Clock::now()-t).count();
            const auto after=m.fast_stats();
            for(float x:logits) if(!std::isfinite(x)) return 7;
            if(std::fwrite(logits.data(),sizeof(float),logits.size(),dump)!=logits.size()) return 8;
            std::fprintf(times,"%d,%zu,%d,%.6f,%llu,%llu,%llu,%llu,%llu,%lld,%lld,%.6f,%.6f,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",r,i,ids[i],ms,
                (unsigned long long)(after.hits-before.hits),(unsigned long long)(after.ram_hits-before.ram_hits),
                (unsigned long long)(after.cpu_experts-before.cpu_experts),(unsigned long long)(after.promotions-before.promotions),
                (unsigned long long)(after.disk_reads-before.disk_reads),(long long)after.pool_slots,(long long)after.ram_slots,
                after.pool_gb,after.ram_gb,(unsigned long long)(LAB_METRIC(after,fetch_bytes)-LAB_METRIC(before,fetch_bytes)),
                (unsigned long long)(LAB_METRIC(after,demotion_bytes)-LAB_METRIC(before,demotion_bytes)),(unsigned long long)(LAB_METRIC(after,demotions)-LAB_METRIC(before,demotions)),
                (unsigned long long)(LAB_METRIC(after,shadow_hits)-LAB_METRIC(before,shadow_hits)),(unsigned long long)(LAB_METRIC(after,shadow_avoided_bytes)-LAB_METRIC(before,shadow_avoided_bytes)),
                (unsigned long long)(LAB_METRIC(after,predicted_bytes)-LAB_METRIC(before,predicted_bytes)), (unsigned long long)(LAB_METRIC(after,predicted_used_bytes)-LAB_METRIC(before,predicted_used_bytes)),
                (unsigned long long)(LAB_METRIC(after,dma_bytes)-LAB_METRIC(before,dma_bytes)));
        }
        std::fflush(times); std::fflush(dump);
    }
    std::fclose(times); std::fclose(dump);
    if(std::getenv("MAYA_TRANSFER_PROBE") && !m.transport_probe(prefix+".transport.csv",err)) {
        std::fprintf(stderr,"transport replay: %s\n",err.c_str());return 9;
    }
    return 0;
}
