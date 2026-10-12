// H5 feasibility: same immutable quantized payload through mapped pull vs DMA.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#define CK(x) do { auto e=(x);if(e!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));return 1;} }while(0)
__global__ void pull(const uint4* src,uint4* dst,size_t n) {
    for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x) dst[i]=src[i];
}
__global__ void pull_sequential(const uint4* src,uint4* dst,size_t part,int count) {
    for(int e=0;e<count;++e)
        for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<part;i+=(size_t)gridDim.x*blockDim.x)
            dst[e*part+i]=src[e*part+i];
}
int main(int argc,char**argv){
    if(argc<2 || argc>4)return 2;
    std::ifstream f(argv[1],std::ios::binary|std::ios::ate);const size_t file_bytes=(size_t)f.tellg();
    const size_t bytes=argc>3 ? (size_t)std::strtoull(argv[3],nullptr,10) : file_bytes;f.seekg(0);
    if(bytes>file_bytes)return 3;
    if(!f || bytes==0 || bytes%16)return 3;
    const char* rotate_env=std::getenv("MAYA_TRANSPORT_ROTATE_MB");
    const size_t requested=rotate_env ? (size_t)std::strtoull(rotate_env,nullptr,10)*1024*1024 : 0;
    const size_t slots=std::max((size_t)1,(requested+bytes*4-1)/(bytes*4));
    void *host=nullptr,*dev=nullptr; CK(cudaHostAlloc(&host,bytes*4*slots,cudaHostAllocMapped));CK(cudaMalloc(&dev,bytes*4));
    f.read((char*)host,bytes);if(!f)return 4;
    for(size_t i=1;i<4*slots;++i)std::memcpy((char*)host+i*bytes,host,bytes);
    cudaDeviceProp prop{};CK(cudaGetDeviceProperties(&prop,0));
    const int blocks=argc>2 ? std::atoi(argv[2]) : prop.multiProcessorCount;
    if(blocks<1 || blocks>prop.multiProcessorCount)return 3;
    cudaStream_t s;cudaEvent_t a,b;CK(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
    CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));
    std::vector<char> back(bytes*4);
    std::puts("copies,mechanism,bytes,mean_ms,GBps,verified,blocks,host_bytes");
    for(int count:{1,4})for(int mode:{0,1,2}){
        const size_t n=bytes*count;
        size_t iteration=0;
        auto issue=[&](){void* source=(char*)host+(iteration++%slots)*bytes*4;
            if(mode==1)return cudaMemcpyAsync(dev,source,n,cudaMemcpyHostToDevice,s);
            if(mode==2)pull_sequential<<<blocks,256,0,s>>>((uint4*)source,(uint4*)dev,bytes/16,count);
            else pull<<<blocks,256,0,s>>>((uint4*)source,(uint4*)dev,n/16);
            return cudaGetLastError();};
        for(int i=0;i<8;++i){CK(issue());}
        CK(cudaEventRecord(a,s));
        for(int i=0;i<100;++i){CK(issue());}
        CK(cudaEventRecord(b,s));CK(cudaEventSynchronize(b));float ms;CK(cudaEventElapsedTime(&ms,a,b));
        CK(cudaMemcpy(back.data(),dev,n,cudaMemcpyDeviceToHost));if(std::memcmp(back.data(),host,n))return 5;
        std::printf("%d,%s,%zu,%.6f,%.6f,1,%d,%zu\n",count,mode==2?"pull-sequential":mode==1?"DMA":"pull",n,ms/100,n/(ms/100)/1e6,blocks,bytes*4*slots);
    }
    CK(cudaFree(dev));CK(cudaFreeHost(host));CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));CK(cudaStreamDestroy(s));return 0;
}
