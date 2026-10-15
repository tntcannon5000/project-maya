// Replay captured prefill combines with the actual production CUDA kernel.
#include "strata/kernels/glm_batch.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <vector>
void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Capture {
    int t,k,e; std::vector<int> ids,pos; std::vector<float> weights,initial,outputs,final;
    std::vector<std::vector<int>> groups;
    explicit Capture(const char* path) {
        std::ifstream f(path,std::ios::binary);
        auto read=[&](void* p,size_t n) { if(!f.read((char*)p,n)) throw std::runtime_error("truncated capture"); };
        int h[4]; read(h,sizeof h); t=h[1];k=h[2];e=h[3];
        if(h[0]!=0x4d4f4531 || t<1 || k<1 || k>8 || e<1 || e>65536) throw std::runtime_error("bad header");
        ids.resize(k);pos.resize(k);weights.resize(k);initial.resize(e);final.resize(e);outputs.resize(k*e);
        read(ids.data(),k*4);read(pos.data(),k*4);read(weights.data(),k*4);read(initial.data(),e*4);
        std::vector<int> seen(k);
        for(;;) { int n;read(&n,4);if(n==-1)break;
            if(n<1 || n>k) throw std::runtime_error("bad group");
            groups.emplace_back();
            for(int a=0;a<n;++a) { int j;read(&j,4);if(j<0 || j>=k || seen[j]++) throw std::runtime_error("bad route");
                groups.back().push_back(j);read(outputs.data()+j*e,e*4); }
        }
        read(final.data(),e*4);if(f.peek()!=EOF || std::count(seen.begin(),seen.end(),1)!=k) throw std::runtime_error("incomplete capture");
    }
};
std::vector<float> replay(const Capture& values,const Capture& partition) {
    const int k=values.k,e=values.e;
    std::vector<int> order(k),pos(k);std::iota(order.begin(),order.end(),0);
    std::sort(order.begin(),order.end(),[&](int a,int b){return partition.pos[a]<partition.pos[b];});
    std::vector<float> out(k*e);
    for(int r=0;r<k;++r) { pos[order[r]]=r;std::copy_n(values.outputs.data()+order[r]*e,e,out.data()+r*e); }
    float *dout,*dw,*df;int* dp;
    check(cudaMalloc(&dout,out.size()*4));check(cudaMalloc(&dw,k*4));check(cudaMalloc(&df,e*4));check(cudaMalloc(&dp,k*4));
    check(cudaMemcpy(dout,out.data(),out.size()*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dw,values.weights.data(),k*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(df,values.initial.data(),e*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dp,pos.data(),k*4,cudaMemcpyHostToDevice));
    for(const auto& group:partition.groups) {
        int lo=k,hi=0;for(int j:group) {lo=std::min(lo,pos[j]);hi=std::max(hi,pos[j]+1);}
        if(hi-lo!=(int)group.size())throw std::runtime_error("noncontiguous group");
        strata::kernels::glmb::moe_combine_add(dout,0,lo,hi,dp,dw,1,k,e,df,nullptr);
    }
    std::vector<float> result(e);check(cudaMemcpy(result.data(),df,e*4,cudaMemcpyDeviceToHost));
    check(cudaFree(dout));check(cudaFree(dw));check(cudaFree(df));check(cudaFree(dp));return result;
}
void difference(const char* name,const std::vector<float>& a,const std::vector<float>& b,bool comma=true) {
    double maximum=0;size_t different=0;
    for(size_t i=0;i<a.size();++i) { if(!std::isfinite(a[i]) || !std::isfinite(b[i]))throw std::runtime_error("nonfinite");
        maximum=std::max(maximum,std::abs(double(a[i])-b[i]));different+=std::memcmp(&a[i],&b[i],4)!=0; }
    std::printf("  \"%s\": {\"max_abs\": %.12g, \"different_bits\": %zu}%s\n",name,maximum,different,comma?",":"");
}
int main(int argc,char** argv) { try {
    if(argc!=3)throw std::runtime_error("usage: maya_moe_combine_replay cold.bin warm.bin");
    Capture a(argv[1]),b(argv[2]);
    if(a.t!=b.t || a.k!=b.k || a.e!=b.e || a.ids!=b.ids) throw std::runtime_error("routes/shapes differ: cross replay invalid");
    auto aa=replay(a,a),ab=replay(a,b),ba=replay(b,a),bb=replay(b,b);
    std::printf("{\n  \"cold_groups\": %zu, \"warm_groups\": %zu,\n",a.groups.size(),b.groups.size());
    difference("weights",a.weights,b.weights);difference("initial",a.initial,b.initial);difference("expert_outputs",a.outputs,b.outputs);
    difference("captured_final",a.final,b.final);difference("cold_replay",aa,a.final);difference("warm_replay",bb,b.final);
    difference("cold_values_warm_partition_vs_cold",ab,aa);difference("cold_values_warm_partition_vs_warm",ab,bb);
    difference("warm_values_cold_partition_vs_cold",ba,aa,false);std::printf("}\n");
    return (std::memcmp(aa.data(),a.final.data(),a.e*4) || std::memcmp(bb.data(),b.final.data(),b.e*4))?2:0;
} catch(const std::exception& e) {std::fprintf(stderr,"%s\n",e.what());return 1;} }
