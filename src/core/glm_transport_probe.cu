// Opt-in, post-inference diagnostic. No live model destinations are written.
#include "glm_fast_state.hpp"
#include "strata/kernels/glm_fast.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace strata::core {
namespace gf=strata::kernels::glmf;
#if defined(STRATA_USE_HIP)
bool Glm5Model::transport_probe(const std::string&,std::string& err) {
    err="CUDA-only diagnostic; HIP transport replay is not implemented";return false;
}
#else
__global__ void probe_stamp(unsigned long long* values,int index) {
    unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));values[index]=t;
}
__global__ void probe_scrub(unsigned int* memory,size_t count) {
    for(size_t at=(size_t)blockIdx.x*blockDim.x+threadIdx.x;at<count;at+=(size_t)gridDim.x*blockDim.x)
        memory[at]+=1u;
}
bool Glm5Model::transport_probe(const std::string& output,std::string& err) {
    auto* F=fast_;
    if(!F || !F->transfer_probe || n_parts_!=1 || g_.n_embd!=4096 || F->max_pf) {err="probe requires recorded native single-device c1 without prediction";return false;}
    if(!glmfast::cuda_ok(cudaDeviceSynchronize(),"probe idle",err))return false;
    while(F->processed.load(std::memory_order_acquire)<F->expected)std::this_thread::yield();
    std::lock_guard<std::mutex> lock(F->mu);
    std::vector<FastState::TransferProbe> records;
    for(auto& r:*F->transfer_probe)if(r.seq)records.push_back(r);
    std::sort(records.begin(),records.end(),[](auto&a,auto&b){return a.seq<b.seq;});
    if(records.size()!=336) {err="need eight complete decode tokens";return false;}
    for(size_t i=1;i<records.size();++i)if(records[i].seq!=records[i-1].seq+1){err="non-contiguous routes";return false;}
    for(const auto&r:records)for(int i=0;i<r.k;++i) {
        for(auto ptr:{r.src[i],((r.cpu_mask>>i)&1u)?r.cpu_src[i]:0ull})if(ptr) {
            bool owned=false;
            for(const auto& rc:F->rc) {
                const auto base=(unsigned long long)rc.base;
                if(ptr>=base && ptr-base<(size_t)rc.n*rc.stride && (ptr-base)%rc.stride==0 && r.blob<=rc.stride)owned=true;
            }
            if(!owned){err="recorded source is not an owned RAM slot";return false;}
        }
    }
    // RAII always joins streams before releasing any replay allocations.
    struct Resources {
        std::vector<void*> memory;cudaStream_t s=nullptr,side=nullptr;
        cudaEvent_t start=nullptr,prefix=nullptr,down0=nullptr,down1=nullptr,end=nullptr;
        cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
        ~Resources(){
            if(s){cudaStreamCaptureStatus status=cudaStreamCaptureStatusNone;
                if(cudaStreamIsCapturing(s,&status)==cudaSuccess && status!=cudaStreamCaptureStatusNone){cudaGraph_t abandoned=nullptr;cudaStreamEndCapture(s,&abandoned);if(abandoned)cudaGraphDestroy(abandoned);}}
            cudaDeviceSynchronize();if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);
            for(auto e:{start,prefix,down0,down1,end})if(e)cudaEventDestroy(e);
            if(s)cudaStreamDestroy(s);if(side)cudaStreamDestroy(side);for(auto p:memory)cudaFree(p);}
    } z;
#define PC(x) do{if(!glmfast::cuda_ok((x),#x,err))return false;}while(0)
    auto alloc=[&](size_t n)->void*{void*p=nullptr;if(cudaMalloc(&p,n)!=cudaSuccess)return nullptr;z.memory.push_back(p);return p;};
    size_t maxblob=0;for(auto&r:records)maxblob=std::max(maxblob,r.blob);
    auto* target=(unsigned char*)alloc(maxblob*8);
    auto* arrays=(unsigned long long*)alloc(24*sizeof(unsigned long long));
    auto* weights=(float*)alloc(8*sizeof(float));auto* flag=(int*)alloc(sizeof(int));
    auto* stamps=(unsigned long long*)alloc(5*sizeof(unsigned long long));
    auto* x=(float*)alloc(4096*sizeof(float));void* xq=alloc(4096/32*36);
    void* hq=alloc((size_t)8*g_.n_ff_exp/32*36);auto* out=(float*)alloc(4096*sizeof(float));
    if(!target||!arrays||!weights||!flag||!x||!xq||!hq||!out||!stamps){err="probe allocation failed";return false;}
    PC(cudaMemset(flag,0,sizeof(int)));PC(cudaMemset(hq,0,(size_t)8*g_.n_ff_exp/32*36));
    PC(cudaStreamCreateWithFlags(&z.s,cudaStreamNonBlocking));PC(cudaStreamCreateWithFlags(&z.side,cudaStreamNonBlocking));
    for(auto*e:{&z.start,&z.prefix,&z.down0,&z.down1,&z.end})PC(cudaEventCreate(e));
    gf::MoeDev d{};d.fetch_src=arrays;d.plan_ptr=arrays+8;d.plan_w=weights;d.cpu_flag=flag;d.cpu_part=out;
    FILE* raw=std::fopen(output.c_str(),"w");if(!raw){err="probe output open";return false;}
    std::unique_ptr<FILE,decltype(&std::fclose)> file(raw,&std::fclose);
    const char* direct_env=std::getenv("MAYA_TRANSFER_DIRECT_ONLY");
    if(direct_env && std::atoi(direct_env)!=0) {
        // Isolated actual-address/current-quant-class replay. These records
        // are decode singleton groups, not a batched-window routing replay.
        // GPU output is the complete weighted GPU contribution; CPU experts
        // run concurrently as real contention but are not added to that output.
        int l2=0;PC(cudaDeviceGetAttribute(&l2,cudaDevAttrL2CacheSize,dev_));
        const size_t scrub_bytes=2*std::max<size_t>((size_t)std::max(l2,0),96ull<<20);
        auto* scrub=(unsigned int*)alloc(scrub_bytes);
        if(!scrub){err="direct probe L2 scrub allocation failed";return false;}
        PC(cudaMemset(scrub,0,scrub_bytes));
        const size_t hbytes=(size_t)8*g_.n_ff_exp/32*36;
        struct Reference {std::vector<float> output;std::vector<unsigned char> hq;};
        std::vector<Reference> reference(records.size());
        const auto setup=[&](const FastState::TransferProbe& r,bool direct)->bool {
            unsigned long long pointers[8]{};
            for(int i=0;i<r.k;++i)
                pointers[i]=r.src[i]?(direct?r.src[i]:(unsigned long long)(target+i*maxblob)):r.resident[i];
            PC(cudaMemcpyAsync(arrays,r.src,sizeof(r.src),cudaMemcpyHostToDevice,z.s));
            PC(cudaMemcpyAsync(arrays+8,pointers,sizeof(pointers),cudaMemcpyHostToDevice,z.s));
            PC(cudaMemcpyAsync(weights,r.w,sizeof(r.w),cudaMemcpyHostToDevice,z.s));
            PC(cudaMemcpyAsync(x,r.x,sizeof(r.x),cudaMemcpyHostToDevice,z.s));
            PC(cudaMemsetAsync(hq,0,hbytes,z.s));
            gf::quantize_q8_1(x,xq,4096,z.s);
            // The stack pointer array remains alive until its upload finishes.
            PC(cudaStreamSynchronize(z.s));
            return true;
        };
        const auto math=[&](const FastState::TransferProbe& r)->bool {
            const auto& L=F->L[(size_t)r.layer];
            if(L.exl3 || (L.gu_type!=10 && L.gu_type!=11 && L.gu_type!=12) ||
               (L.d_type!=10 && L.d_type!=11 && L.d_type!=12)) {
                err="direct probe requires native Q2_K/Q3_K/Q4_K";return false;
            }
            gf::moe_gate_up(L.gu_type,d,r.k,4096,g_.n_ff_exp,g_.swiglu_exp,xq,hq,nullptr,0,nullptr,0,nullptr,z.s);
            gf::moe_down(L.d_type,d,r.k,4096,g_.n_ff_exp,r.prefix,hq,nullptr,out,z.s);
            return true;
        };
        // Reference and all validation/readback are outside measured intervals.
        for(size_t index=0;index<records.size();++index) {
            const auto& r=records[index];
            if(!setup(r,false))return false;
            gf::moe_fetch(d,r.k,r.blob,z.s);
            if(!math(r))return false;
            PC(cudaStreamSynchronize(z.s));
            auto& ref=reference[index];ref.output.resize(4096);ref.hq.resize(hbytes);
            PC(cudaMemcpy(ref.output.data(),out,4096*sizeof(float),cudaMemcpyDeviceToHost));
            PC(cudaMemcpy(ref.hq.data(),hq,hbytes,cudaMemcpyDeviceToHost));
            for(float value:ref.output)if(!std::isfinite(value)) {
                err="direct probe staged reference has non-finite output at seq="+std::to_string(r.seq);return false;
            }
        }
        std::fprintf(raw,"mode,pass,seq,layer,bytes,cpu_experts,fetch_ms,gpu_window_ms,cpu_ms,wall_ms,split,verified,"
                         "gu_type,down_type,explicit_copy_bytes,cpu_enabled,exact_hq,l2_scrub_bytes\n");
        std::fprintf(stderr,"maya direct-RAM probe: current address/quant-class replay, %zu singleton route records, "
                            "%zu-byte L2 scrub outside timing; CPU output remains separate\n",records.size(),scrub_bytes);
        for(int mode=0;mode<4;++mode) {
            const bool direct=(mode&1)!=0,cpu=mode>=2;
            const char* name=mode==0?"stage_gpu":mode==1?"direct_gpu":mode==2?"stage_cpu_gpu":"direct_cpu_gpu";
            // Whole history between repetitions limits CPU-source hot-cache bias.
            for(int pass=-1;pass<2;++pass)for(size_t index=0;index<records.size();++index) {
                const auto& r=records[index];const auto& L=F->L[(size_t)r.layer];
                if(!setup(r,direct))return false;
                probe_scrub<<<256,256,0,z.s>>>(scrub,scrub_bytes/sizeof(*scrub));
                PC(cudaStreamSynchronize(z.s));
                PC(cudaStreamBeginCapture(z.s,cudaStreamCaptureModeThreadLocal));
                probe_stamp<<<1,1,0,z.s>>>(stamps,0);
                if(!direct)gf::moe_fetch(d,r.k,r.blob,z.s);
                probe_stamp<<<1,1,0,z.s>>>(stamps,1);
                if(!math(r))return false;
                probe_stamp<<<1,1,0,z.s>>>(stamps,4);
                PC(cudaStreamEndCapture(z.s,&z.graph));
                PC(cudaGraphInstantiate(&z.exec,z.graph,nullptr,nullptr,0));
                int nc=0;size_t bytes=0;const uint8_t* cb[8];float cw[8];
                for(int i=0;i<r.k;++i) {
                    if(r.src[i])bytes+=r.blob;
                    if((r.cpu_mask>>i)&1u){cb[nc]=(const uint8_t*)r.cpu_src[i];cw[nc++]=r.w[i];}
                }
                float cpu_out[4096]{};
                const auto begin=std::chrono::steady_clock::now();
                PC(cudaGraphLaunch(z.exec,z.s));
                const auto c0=std::chrono::steady_clock::now();
                if(cpu && nc)fast_cpu_experts(r.layer,nc,cb,cw,r.x,cpu_out);
                const double cpu_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-c0).count();
                PC(cudaStreamSynchronize(z.s));
                const double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
                unsigned long long ts[5]{};PC(cudaMemcpy(ts,stamps,sizeof(ts),cudaMemcpyDeviceToHost));
                std::vector<float> actual(4096);std::vector<unsigned char> actual_hq(hbytes);
                PC(cudaMemcpy(actual.data(),out,4096*sizeof(float),cudaMemcpyDeviceToHost));
                PC(cudaMemcpy(actual_hq.data(),hq,hbytes,cudaMemcpyDeviceToHost));
                for(float value:actual)if(!std::isfinite(value)){err="direct probe non-finite GPU output";return false;}
                if(cpu && nc)for(float value:cpu_out)if(!std::isfinite(value)){err="direct probe non-finite CPU output";return false;}
                if(std::memcmp(actual.data(),reference[index].output.data(),4096*sizeof(float)) ||
                   std::memcmp(actual_hq.data(),reference[index].hq.data(),hbytes)) {
                    err=std::string("direct probe exact output/hq mismatch mode=")+name+" seq="+std::to_string(r.seq);return false;
                }
                if(pass>=0)std::fprintf(raw,"%s,%d,%u,%d,%zu,%d,%.6f,%.6f,%.6f,%.6f,0,1,%d,%d,%zu,%d,1,%zu\n",
                    name,pass,r.seq,r.layer,bytes,nc,direct?0.0:(ts[1]-ts[0])/1e6,
                    (ts[4]-ts[0])/1e6,cpu_ms,wall,L.gu_type,L.d_type,direct?size_t(0):bytes,cpu?1:0,scrub_bytes);
                PC(cudaGraphExecDestroy(z.exec));z.exec=nullptr;
                PC(cudaGraphDestroy(z.graph));z.graph=nullptr;
            }
            if(std::fflush(raw)!=0 || std::ferror(raw)){err="direct probe write failed";return false;}
        }
        return true;
    }
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
        probe_stamp<<<1,1,0,z.s>>>(stamps,0);
        if(dma) {
            for(int i=0;i<r.k;++i)if(r.src[i])PC(cudaMemcpyAsync((void*)ptr[i],(void*)r.src[i],r.blob,cudaMemcpyHostToDevice,z.s));
            PC(cudaEventRecord(z.prefix,z.s));
            probe_stamp<<<1,1,0,z.s>>>(stamps,1);
        } else if(r.split) {
            gf::moe_fetch_component(d,r.k,0,r.prefix,0,z.s);probe_stamp<<<1,1,0,z.s>>>(stamps,1);PC(cudaEventRecord(z.prefix,z.s));
            PC(cudaStreamWaitEvent(z.side,z.prefix,0));probe_stamp<<<1,1,0,z.side>>>(stamps,2);
            gf::moe_fetch_component(d,r.k,r.prefix,r.blob-r.prefix,32,z.side);probe_stamp<<<1,1,0,z.side>>>(stamps,3);PC(cudaEventRecord(z.down1,z.side));
            if(math)gf::moe_gate_up(L.gu_type,d,r.k,4096,g_.n_ff_exp,g_.swiglu_exp,xq,hq,nullptr,0,nullptr,0,nullptr,z.s);
            PC(cudaStreamWaitEvent(z.s,z.down1,0));
        } else {gf::moe_fetch(d,r.k,r.blob,z.s);probe_stamp<<<1,1,0,z.s>>>(stamps,1);PC(cudaEventRecord(z.prefix,z.s));}
        if(math) {
            if(!r.split)gf::moe_gate_up(L.gu_type,d,r.k,4096,g_.n_ff_exp,g_.swiglu_exp,xq,hq,nullptr,0,nullptr,0,nullptr,z.s);
            gf::moe_down(L.d_type,d,r.k,4096,g_.n_ff_exp,r.prefix,hq,nullptr,out,z.s);
        }
        probe_stamp<<<1,1,0,z.s>>>(stamps,4);PC(cudaStreamEndCapture(z.s,&z.graph));
        PC(cudaGraphInstantiate(&z.exec,z.graph,nullptr,nullptr,0));
        float cpu_out[4096]{};
        for(int pass=-1;pass<2;++pass) {
            const auto begin=std::chrono::steady_clock::now();PC(cudaGraphLaunch(z.exec,z.s));
            const auto c0=std::chrono::steady_clock::now();
            if(cpu&&nc)fast_cpu_experts(r.layer,nc,cb,cw,r.x,cpu_out);
            const double cpu_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-c0).count();
            PC(cudaStreamSynchronize(z.s));
            const double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            unsigned long long ts[5]{};PC(cudaMemcpy(ts,stamps,sizeof(ts),cudaMemcpyDeviceToHost));
            double prefix=(ts[1]-ts[0])/1e6,tail=0,window=(ts[4]-ts[0])/1e6;
            if(r.split&&!dma)tail=(ts[3]-ts[2])/1e6;
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
#endif
}
