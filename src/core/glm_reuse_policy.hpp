#pragma once
#include <array>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace strata::core {
// Causal same-layer reuse (next route by default), used ONLY for choosing a completed
// resident to evict. Later residency can change CPU/GPU assignment; weights
// and the required expert set are preserved.
// Owned by FastState; accessed under its existing service/boundary mutex.
struct GlmReusePolicy {
    struct Layer {
        std::vector<float> transitions, seen, score;
        std::array<int,8> previous{};
        int previous_n=0;
        // Only the opt-in 16-route model uses this bounded ring. An anchor is
        // trained once all of its next 16 same-layer routes have completed.
        std::array<std::array<int,8>,17> history{};
        std::array<int,17> history_sizes{};
        int history_next=0, history_n=0;
        std::vector<uint8_t> future;
        std::vector<float> window_prior;
    };
    int experts=0, horizon=1;
    uint64_t observations=0, changed_victims=0, matured_anchors=0;
    std::vector<Layer> layers;
    void init(int nl,int ne) {
        const char* requested=std::getenv("MAYA_REUSE_HORIZON");
        horizon=requested && std::strcmp(requested,"16")==0 ? 16 : 1;
        experts=ne; layers.resize(nl);
        for(auto& l:layers) {
            l.transitions.assign((size_t)ne*ne,0);
            l.seen.assign(ne,0); l.score.assign(ne,0);
            if(horizon==16) {
                l.future.assign(ne,0); l.window_prior.assign(ne,0);
                l.history_next=0; l.history_n=0;
            }
        }
    }
    void observe(int il,const int* ids,int n,const uint32_t* frequency) {
        auto& l=layers[(size_t)il];
        if(horizon==16) {
            // Current demand is already observed. Only the oldest anchor gets
            // this label; no future of the current eviction decision is read.
            const int k=std::max(0,std::min(n,8));
            const int at=l.history_next;
            for(int a=0;a<k;++a)l.history[(size_t)at][(size_t)a]=ids[a];
            l.history_sizes[(size_t)at]=k;
            l.history_next=(at+1)%17;
            l.history_n=std::min(17,l.history_n+1);
            if(l.history_n==17) {
                std::fill(l.future.begin(),l.future.end(),0);
                std::array<int,128> future_ids{};
                int future_n=0;
                for(int d=1;d<=16;++d) {
                    const int p=(l.history_next+d)%17;
                    for(int a=0;a<l.history_sizes[(size_t)p];++a) {
                        const int e=l.history[(size_t)p][(size_t)a];
                        if(!l.future[(size_t)e]) {
                            l.future[(size_t)e]=1;
                            future_ids[(size_t)future_n++]=e;
                        }
                    }
                }
                const int anchor=l.history_next;
                for(int a=0;a<l.history_sizes[(size_t)anchor];++a) {
                    const int e=l.history[(size_t)anchor][(size_t)a];
                    float* row=l.transitions.data()+(size_t)e*experts;
                    if(l.seen[e]>=2048) {
                        l.seen[e]*=0.5f;
                        for(int j=0;j<experts;++j)row[j]*=0.5f;
                    }
                    ++l.seen[e];
                    for(int j=0;j<future_n;++j)++row[future_ids[(size_t)j]];
                }
                ++matured_anchors;
            }
            double total=0;
            for(int e=0;e<experts;++e)total+=frequency[e];
            // The marginal chance of an expert in 16 routes (8 selections
            // each) supplies the same 16-anchor smoothing as the old model.
            // 128 is a power of two: avoid a libm pow call per expert/route.
            for(int j=0;j<experts;++j) {
                double remain=1.0-frequency[j]/std::max(total,1.0);
                for(int power=0;power<7;++power)remain*=remain;
                l.window_prior[(size_t)j]=(float)(1.0-remain);
            }
            std::fill(l.score.begin(),l.score.end(),0.0f);
            for(int a=0;a<k;++a) {
                const int e=ids[a];const float weight=1.0f/(k*(l.seen[e]+16.0f));
                const float* row=l.transitions.data()+(size_t)e*experts;
                for(int j=0;j<experts;++j)
                    l.score[j]+=(row[j]+16.0f*l.window_prior[(size_t)j])*weight;
                l.previous[a]=e;
            }
            l.previous_n=k; ++observations;
            return;
        }
        // Preserve the default one-route model's arithmetic and update order.
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
