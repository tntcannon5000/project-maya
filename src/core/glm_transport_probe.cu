// Opt-in, post-inference diagnostic. No live model destinations are written.
#include "glm_fast_state.hpp"
#include "strata/kernels/glm_fast.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace strata::core {
namespace gf=strata::kernels::glmf;
bool Glm5Model::transport_probe(const std::string& output,std::string& err) {
    auto* F=fast_;
    if(!F || n_parts_!=1 || g_.n_embd!=4096 || F->max_pf) {err="probe requires native single-device c1 without prediction";return false;}
    if(!glmfast::cuda_ok(cudaDeviceSynchronize(),"probe idle",err))return false;
    while(F->processed.load(std::memory_order_acquire)<F->expected)std::this_thread::yield();
    std::lock_guard<std::mutex> lock(F->mu);
    std::vector<FastState::TransferProbe> records;
    for(auto& r:F->transfer_probe)if(r.seq)records.push_back(r);
    std::sort(records.begin(),records.end(),[](auto&a,auto&b){return a.seq<b.seq;});
    if(records.size()!=336) {err="need eight complete decode tokens";return false;}
    for(size_t i=1;i<records.size();++i)if(records[i].seq!=records[i-1].seq+1){err="non-contiguous routes";return false;}
    // RAII always joins streams before releasing any replay allocations.
    struct Resources {
        std::vector<void*> memory;cudaStream_t s=nullptr,side=nullptr;
        cudaEvent_t start=nullptr,prefix=nullptr,down0=nullptr,down1=nullptr,end=nullptr;
        cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
        ~Resources(){cudaDeviceSynchronize();if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);
            for(auto e:{start,prefix,down0,down1,end})if(e)cudaEventDestroy(e);
            if(s)cudaStreamDestroy(s);if(side)cudaStreamDestroy(side);for(auto p:memory)cudaFree(p);}
    } z;
#define PC(x) do{if(!glmfast::cuda_ok((x),#x,err))return false;}while(0)
    auto alloc=[&](size_t n)->void*{void*p=nullptr;if(cudaMalloc(&p,n)!=cudaSuccess)return nullptr;z.memory.push_back(p);return p;};
    size_t maxblob=0;for(auto&r:records)maxblob=std::max(maxblob,r.blob);
    auto* target=(unsigned char*)alloc(maxblob*8);
    auto* arrays=(unsigned long long*)alloc(24*sizeof(unsigned long long));
    auto* weights=(float*)alloc(8*sizeof(float));auto* flag=(int*)alloc(sizeof(int));
    auto* x=(float*)alloc(4096*sizeof(float));void* xq=alloc(4096/32*36);
    void* hq=alloc((size_t)8*g_.n_ff_exp/32*36);auto* out=(float*)alloc(4096*sizeof(float));
    if(!target||!arrays||!weights||!flag||!x||!xq||!hq||!out){err="probe allocation failed";return false;}
    PC(cudaMemset(flag,0,sizeof(int)));PC(cudaMemset(hq,0,(size_t)8*g_.n_ff_exp/32*36));
    PC(cudaStreamCreateWithFlags(&z.s,cudaStreamNonBlocking));PC(cudaStreamCreateWithFlags(&z.side,cudaStreamNonBlocking));
    for(auto*e:{&z.start,&z.prefix,&z.down0,&z.down1,&z.end})PC(cudaEventCreate(e));
    gf::MoeDev d{};d.fetch_src=arrays;d.plan_ptr=arrays+8;d.plan_w=weights;d.cpu_flag=flag;d.cpu_part=out;
    FILE* raw=std::fopen(output.c_str(),"w");if(!raw){err="probe output open";return false;}
    std::unique_ptr<FILE,decltype(&std::fclose)> file(raw,&std::fclose);
    std::fprintf(raw,"mode,pass,seq,layer,bytes,cpu_experts,fetch_ms,gpu_window_ms,cpu_ms,wall_ms,split,verified\n");
    // Each mode uses the identical sequence, pinned addresses and quant types.
    // GPU work reproduces routed gate/up and down, not whole-layer attention.
    for(int mode=0;mode<5;++mode) {
      const bool cpu=mode==1||mode==3,math=mode==2||mode==3,dma=mode==4;
      const char* name=mode==0?"pull":mode==1?"pull_cpu":mode==2?"pull_gpu":mode==3?"pull_cpu_gpu":"dma";
      for(const auto&r:records) {
        auto& L=F->L[(size_t)r.layer];if(L.exl3){err="native weights only";return false;}
        unsigned long long ptr[8]{};size_t bytes=0;int nc=0;const uint8_t* cb[8];float cw[8];
        for(int i=0;i<r.k;++i) {
            if(r.src[i]){ptr[i]=(unsigned long long)(target+i*maxblob);bytes+=r.blob;}
            else ptr[i]=r.resident[i];
            if((r.cpu_mask>>i)&1u){cb[nc]=(const uint8_t*)r.cpu_src[i];cw[nc++]=r.w[i];}
        }
        PC(cudaMemcpy(arrays,r.src,sizeof(r.src),cudaMemcpyHostToDevice));
        PC(cudaMemcpy(arrays+8,ptr,sizeof(ptr),cudaMemcpyHostToDevice));
        PC(cudaMemcpy(weights,r.w,sizeof(r.w),cudaMemcpyHostToDevice));
        PC(cudaMemcpy(x,r.x,sizeof(r.x),cudaMemcpyHostToDevice));
        gf::quantize_q8_1(x,xq,4096,z.s);PC(cudaStreamSynchronize(z.s));
        PC(cudaStreamBeginCapture(z.s,cudaStreamCaptureModeThreadLocal));
        PC(cudaEventRecord(z.start,z.s));
        if(dma) {
            for(int i=0;i<r.k;++i)if(r.src[i])PC(cudaMemcpyAsync((void*)ptr[i],(void*)r.src[i],r.blob,cudaMemcpyHostToDevice,z.s));
            PC(cudaEventRecord(z.prefix,z.s));
        } else if(r.split) {
            gf::moe_fetch_component(d,r.k,0,r.prefix,0,z.s);PC(cudaEventRecord(z.prefix,z.s));
            PC(cudaStreamWaitEvent(z.side,z.prefix,0));PC(cudaEventRecord(z.down0,z.side));
            gf::moe_fetch_component(d,r.k,r.prefix,r.blob-r.prefix,32,z.side);PC(cudaEventRecord(z.down1,z.side));
            if(math)gf::moe_gate_up(L.gu_type,d,r.k,4096,g_.n_ff_exp,g_.swiglu_exp,xq,hq,nullptr,0,nullptr,0,nullptr,z.s);
            PC(cudaStreamWaitEvent(z.s,z.down1,0));
        } else {gf::moe_fetch(d,r.k,r.blob,z.s);PC(cudaEventRecord(z.prefix,z.s));}
        if(math) {
            if(!r.split)gf::moe_gate_up(L.gu_type,d,r.k,4096,g_.n_ff_exp,g_.swiglu_exp,xq,hq,nullptr,0,nullptr,0,nullptr,z.s);
            gf::moe_down(L.d_type,d,r.k,4096,g_.n_ff_exp,r.prefix,hq,nullptr,out,z.s);
        }
        PC(cudaEventRecord(z.end,z.s));PC(cudaStreamEndCapture(z.s,&z.graph));
        PC(cudaGraphInstantiate(&z.exec,z.graph,nullptr,nullptr,0));
        float cpu_out[4096]{};
        for(int pass=-1;pass<2;++pass) {
            const auto begin=std::chrono::steady_clock::now();PC(cudaGraphLaunch(z.exec,z.s));
            const auto c0=std::chrono::steady_clock::now();
            if(cpu&&nc)fast_cpu_experts(r.layer,nc,cb,cw,r.x,cpu_out);
            const double cpu_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-c0).count();
            PC(cudaStreamSynchronize(z.s));
            const double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            float prefix=0,tail=0,window=0;PC(cudaEventElapsedTime(&prefix,z.start,z.prefix));
            if(r.split&&!dma)PC(cudaEventElapsedTime(&tail,z.down0,z.down1));
            PC(cudaEventElapsedTime(&window,z.start,z.end));
            if(pass>=0)std::fprintf(raw,"%s,%d,%u,%d,%zu,%d,%.6f,%.6f,%.6f,%.6f,%d,1\n",name,pass,r.seq,r.layer,bytes,nc,prefix+tail,window,cpu_ms,wall,r.split);
        }
        // Check every transferred byte after the measured passes. The source
        // allocation is still owned by the idle model; contents may have been
        // reused after original routing, so validate against CURRENT bytes.
        std::vector<unsigned char> back(r.blob);
        for(int i=0;i<r.k;++i)if(r.src[i]) {
            PC(cudaMemcpy(back.data(),(void*)ptr[i],r.blob,cudaMemcpyDeviceToHost));
            if(std::memcmp(back.data(),(void*)r.src[i],r.blob)){err="probe copy mismatch";return false;}
        }
        PC(cudaGraphExecDestroy(z.exec));z.exec=nullptr;PC(cudaGraphDestroy(z.graph));z.graph=nullptr;
      }
      std::fflush(raw);
    }
    return true;
#undef PC
}
}
