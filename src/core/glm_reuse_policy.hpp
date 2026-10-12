#pragma once
#include <array>
#include <algorithm>
#include <cstdint>
#include <vector>

namespace strata::core {
// Causal same-layer next-route reuse, used ONLY for choosing a completed
// resident to evict. Later residency can change CPU/GPU assignment; weights
// and the required expert set are preserved.
// Owned by FastState; accessed under its existing service/boundary mutex.
struct GlmReusePolicy {
    struct Layer {
        std::vector<float> transitions, seen, score;
        std::array<int,8> previous{};
        int previous_n=0;
    };
    int experts=0;
    uint64_t observations=0, changed_victims=0;
    std::vector<Layer> layers;
    void init(int nl,int ne) {
        experts=ne; layers.resize(nl);
        for(auto& l:layers) {
            l.transitions.assign((size_t)ne*ne,0);
            l.seen.assign(ne,0); l.score.assign(ne,0);
        }
    }
    void observe(int il,const int* ids,int n,const uint32_t* frequency) {
        auto& l=layers[(size_t)il];
        for(int a=0;a<l.previous_n;++a) {
            const int e=l.previous[a]; float* row=l.transitions.data()+(size_t)e*experts;
            if(l.seen[e]>=2048) { l.seen[e]*=0.5f; for(int j=0;j<experts;++j)row[j]*=0.5f; }
            ++l.seen[e];
            for(int b=0;b<n;++b)++row[ids[b]];
        }
        double total=0;
        for(int e=0;e<experts;++e)total+=frequency[e];
        const float rate=(float)(n/std::max(total,1.0));
        std::fill(l.score.begin(),l.score.end(),0.0f);
        for(int a=0;a<n;++a) {
            const int e=ids[a];const float weight=1.0f/(n*(l.seen[e]+16.0f));
            const float* row=l.transitions.data()+(size_t)e*experts;
            for(int j=0;j<experts;++j)l.score[j]+=(row[j]+16.0f*frequency[j]*rate)*weight;
            l.previous[a]=e;
        }
        l.previous_n=n; ++observations;
    }
};
}
