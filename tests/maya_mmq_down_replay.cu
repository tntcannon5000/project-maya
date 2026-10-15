#include "strata/prefill/moe_mmq.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <vector>
namespace mmq=strata::prefill::mmq;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Result {std::vector<uint8_t> weight,input;std::vector<float> expected,actual,isolated;};
Result replay(const char* path) {
    std::ifstream file(path,std::ios::binary);
    auto read=[&](void* p,size_t n){if(!file.read((char*)p,n))throw std::runtime_error("truncated down capture");};
    int64_t h[11];read(h,sizeof h);
    if(h[0]!=0x444f574e || h[2]<1 || h[2]>65536 || h[3]<1 || h[3]>65536 || h[4]<1 || h[4]>512 ||
       h[5]<1 || h[5]>131072 || h[6]<1 || h[6]>h[5] || h[7]<0 || h[7]>=h[5] ||
       !mmq::supported((int)h[1])) throw std::runtime_error("invalid down header");
    const int n=(int)h[4],rows=(int)h[5],target=(int)h[7],e=(int)h[2];
    const size_t matrix=mmq::matrix_bytes((int)h[1],e,h[3]),qb=mmq::q8_bytes(rows,h[3]);
    if(h[8]!=(int64_t)mmq::q8_block_bytes() || h[9]!=(int64_t)qb || h[10]!=(int64_t)matrix ||
       matrix*n>2ull*1024*1024*1024 || qb>1024ull*1024*1024)throw std::runtime_error("invalid capture sizes");
    std::vector<int> bounds(n+1);read(bounds.data(),bounds.size()*4);
    if(bounds.front()!=0 || bounds.back()!=rows || !std::is_sorted(bounds.begin(),bounds.end()))
        throw std::runtime_error("invalid bounds");
    int selected=0;while(selected<n && bounds[selected+1]<=target)++selected;
    uint8_t *w,*q;float* out;int *db,*ids;
    check(cudaMalloc(&w,matrix*n));check(cudaMalloc(&q,qb));check(cudaMalloc(&out,(size_t)rows*e*4));
    // MMQ reads a full output-column tile of identity IDs before masking tail
    // writes. Production iota has spare capacity; preserve it in this fixture.
    constexpr int id_padding=512;
    check(cudaMalloc(&db,bounds.size()*4));check(cudaMalloc(&ids,(rows+id_padding)*4));
    Result result;std::vector<uint8_t> host(std::max(matrix,qb));
    for(int i=0;i<n;++i){read(host.data(),matrix);check(cudaMemcpy(w+matrix*i,host.data(),matrix,cudaMemcpyHostToDevice));
        if(i==selected)result.weight.assign(host.begin(),host.begin()+matrix);}
    read(host.data(),qb);check(cudaMemcpy(q,host.data(),qb,cudaMemcpyHostToDevice));
    // Quantized input is column-block-major; ignore allocation tail padding.
    const size_t block=h[8],blocks=((h[3]+511)/512*512)/128;
    for(size_t b=0;b<blocks;++b)result.input.insert(result.input.end(),host.begin()+(b*rows+target)*block,host.begin()+(b*rows+target+1)*block);
    result.expected.resize(e);result.actual.resize(e);result.isolated.resize(e);read(result.expected.data(),e*4);
    if(file.peek()!=EOF)throw std::runtime_error("trailing data");
    std::vector<int> identity(rows+id_padding);std::iota(identity.begin(),identity.end(),0);
    check(cudaMemcpy(db,bounds.data(),bounds.size()*4,cudaMemcpyHostToDevice));check(cudaMemcpy(ids,identity.data(),identity.size()*4,cudaMemcpyHostToDevice));
    mmq::Context context;mmq::Product p;
    p.w=w;p.type=(int)h[1];p.w_rows=e;p.w_cols=h[3];p.expert_bytes=matrix;p.n=n;
    p.xq=q;p.bounds=db;p.ids=ids;p.total_rows=rows;p.max_rows=h[6];p.dst=out;p.ld_dst=e;
    context.run(p,nullptr);check(cudaMemcpy(result.actual.data(),out+(size_t)target*e,e*4,cudaMemcpyDeviceToHost));
    // Keep dispatch geometry, selected weights and selected quantized row fixed.
    // Remove unrelated data so a matching result also excludes cross-expert data
    // dependence as the explanation for a shape-dependent target-row difference.
    check(cudaMemset(w,0,matrix*n));check(cudaMemset(q,0,qb));check(cudaMemset(out,0,(size_t)rows*e*4));
    check(cudaMemcpy(w+matrix*selected,result.weight.data(),matrix,cudaMemcpyHostToDevice));
    for(size_t b=0;b<blocks;++b)
        check(cudaMemcpy(q+(b*rows+target)*block,result.input.data()+b*block,block,cudaMemcpyHostToDevice));
    context.run(p,nullptr);check(cudaMemcpy(result.isolated.data(),out+(size_t)target*e,e*4,cudaMemcpyDeviceToHost));
    check(cudaFree(w));check(cudaFree(q));check(cudaFree(out));check(cudaFree(db));check(cudaFree(ids));return result;
}
void diff(const char* label,const std::vector<float>& a,const std::vector<float>& b,bool comma=true) {
    if(a.size()!=b.size())throw std::runtime_error("shape mismatch");double mx=0;size_t bits=0;
    for(size_t i=0;i<a.size();++i){if(!std::isfinite(a[i])||!std::isfinite(b[i]))throw std::runtime_error("nonfinite");
        mx=std::max(mx,std::abs(double(a[i])-b[i]));bits+=std::memcmp(&a[i],&b[i],4)!=0;}
    std::printf("  \"%s\": {\"max_abs\": %.12g, \"different_bits\": %zu}%s\n",label,mx,bits,comma?",":"");
}
int main(int argc,char** argv){try{
    if(argc!=3)throw std::runtime_error("usage: maya_mmq_down_replay cold.down warm.down");
    auto a=replay(argv[1]),b=replay(argv[2]);
    std::printf("{\n  \"selected_weight_exact\": %s,\n  \"selected_quantized_input_exact\": %s,\n",a.weight==b.weight?"true":"false",a.input==b.input?"true":"false");
    diff("cold_replay",a.actual,a.expected);diff("warm_replay",b.actual,b.expected);
    diff("cold_unrelated_data_zeroed",a.actual,a.isolated);diff("warm_unrelated_data_zeroed",b.actual,b.isolated);
    diff("between_dispatches",a.actual,b.actual,false);std::printf("}\n");
    return (a.actual.size()!=a.expected.size() || b.actual.size()!=b.expected.size() ||
        std::memcmp(a.actual.data(),a.expected.data(),a.actual.size()*4) || std::memcmp(b.actual.data(),b.expected.data(),b.actual.size()*4) ||
        std::memcmp(a.actual.data(),a.isolated.data(),a.actual.size()*4) || std::memcmp(b.actual.data(),b.isolated.data(),b.actual.size()*4))?2:0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
