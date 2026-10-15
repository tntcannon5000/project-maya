#pragma once
// Opt-in, synchronizing diagnostic. Never use capture timings as performance data.
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
namespace strata::core {
class PrefillMoeCapture {
    FILE* file_ = nullptr;
    FILE* products_ = nullptr;
    std::string path_;
    int k_, e_;
    cudaStream_t stream_;
    std::vector<int> positions_;
    void write(const void* p, size_t n) {
        if (std::fwrite(p, 1, n, file_) != n) throw std::runtime_error("MoE capture write failed");
    }
    void copy(void* host, const void* device, size_t n) {
        if (cudaMemcpyAsync(host, device, n, cudaMemcpyDeviceToHost, stream_) != cudaSuccess ||
            cudaStreamSynchronize(stream_) != cudaSuccess) throw std::runtime_error("MoE capture copy failed");
    }
    void vector(const float* device) {
        std::vector<float> h(e_); copy(h.data(), device, e_*sizeof(float)); write(h.data(), e_*sizeof(float));
    }
public:
    PrefillMoeCapture(int layer, int t, int k, int e, int cpu_rows, const int* ids,
                      const int* pos, const float* weights, const float* initial, cudaStream_t stream)
        : k_(k), e_(e), stream_(stream) {
        static const char* root = std::getenv("MAYA_PREFILL_MOE_CAPTURE");
        if (!root || !*root || layer != 3) return;
        if (cpu_rows) throw std::runtime_error("MoE capture requires GPU-only prefill");
        static std::atomic<unsigned> serial{0};
        auto path = std::string(root) + "-" + std::to_string(serial++) + ".bin";
        path_ = path;
        file_ = std::fopen(path.c_str(), "wb");
        if (!file_) throw std::runtime_error("MoE capture open failed");
        const int header[4] = {0x4d4f4531, t, k, e}; write(header, sizeof header);
        std::vector<int> hids(k); positions_.resize(k); std::vector<float> hw(k);
        copy(hids.data(), ids+(t-1)*k, k*sizeof(int));
        copy(positions_.data(), pos+(t-1)*k, k*sizeof(int));
        copy(hw.data(), weights+(t-1)*k, k*sizeof(float));
        write(hids.data(), k*sizeof(int)); write(positions_.data(), k*sizeof(int)); write(hw.data(), k*sizeof(float));
        vector(initial+(t-1)*e);
        std::fprintf(stderr,"[moe-capture] %s T=%d K=%d E=%d\n",path.c_str(),t,k,e);
    }
    ~PrefillMoeCapture() { if (file_) std::fclose(file_); if (products_) std::fclose(products_); }
    // Capture selected post-SwiGLU vectors before down overwrites the output window.
    // Rows refer to the same global sorted space as positions_. Only MMQ calls engage.
    void product(const float* hidden, const float* input, int first, int rows, int max_rows,
                 int experts, int gu_type, int down_type, bool fused, int nff, int ld) {
        if (!file_) return;
        for(int j=0;j<k_;++j) {
            const int row=positions_[j]-first;
            if(row<0 || row>=rows) continue;
            if(!products_) {
                products_=std::fopen((path_+".products").c_str(),"wb");
                if(!products_) throw std::runtime_error("MoE products open failed");
            }
            const int h[10]={j,positions_[j],rows,max_rows,experts,gu_type,down_type,(int)fused,nff,e_};
            std::vector<float> x(e_), mid(nff);
            copy(x.data(),input,e_*sizeof(float));
            copy(mid.data(),hidden+(size_t)row*ld,nff*sizeof(float));
            if(std::fwrite(h,1,sizeof h,products_)!=sizeof h ||
               std::fwrite(x.data(),sizeof(float),e_,products_)!=(size_t)e_ ||
               std::fwrite(mid.data(),sizeof(float),nff,products_)!=(size_t)nff)
                throw std::runtime_error("MoE products write failed");
        }
    }
    void flush(const float* output, int base, int lo, int hi) {
        if (!file_) return;
        std::vector<int> selected;
        for(int j=0;j<k_;++j) if(positions_[j]>=lo && positions_[j]<hi) selected.push_back(j);
        if(selected.empty()) return;
        const int n = (int)selected.size(); write(&n,sizeof n);
        for(int j:selected) { write(&j,sizeof j); vector(output+(size_t)(positions_[j]-base)*e_); }
    }
    void finish(const float* final, int t) {
        if (!file_) return;
        const int end=-1; write(&end,sizeof end); vector(final+(t-1)*e_);
    }
};
}
