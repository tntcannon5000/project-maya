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
int main(int argc,char**argv){
    if(argc!=2)return 2;
    std::ifstream f(argv[1],std::ios::binary|std::ios::ate);const size_t bytes=(size_t)f.tellg();f.seekg(0);
    if(!f || bytes==0 || bytes%16)return 3;
    void *host=nullptr,*dev=nullptr; CK(cudaHostAlloc(&host,bytes*4,cudaHostAllocMapped));CK(cudaMalloc(&dev,bytes*4));
    f.read((char*)host,bytes);if(!f)return 4;
    for(int i=1;i<4;++i)std::memcpy((char*)host+i*bytes,host,bytes);
    cudaDeviceProp prop{};CK(cudaGetDeviceProperties(&prop,0));
    cudaStream_t s;cudaEvent_t a,b;CK(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
    CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));
    std::vector<char> back(bytes*4);
    std::puts("copies,mechanism,bytes,mean_ms,GBps,verified");
    for(int count:{1,4})for(int mode:{0,1}){
        const size_t n=bytes*count;
        for(int i=0;i<8;++i){if(mode)CK(cudaMemcpyAsync(dev,host,n,cudaMemcpyHostToDevice,s));else pull<<<prop.multiProcessorCount,256,0,s>>>((uint4*)host,(uint4*)dev,n/16);}
        CK(cudaEventRecord(a,s));
        for(int i=0;i<100;++i){if(mode)CK(cudaMemcpyAsync(dev,host,n,cudaMemcpyHostToDevice,s));else pull<<<prop.multiProcessorCount,256,0,s>>>((uint4*)host,(uint4*)dev,n/16);}
        CK(cudaEventRecord(b,s));CK(cudaEventSynchronize(b));float ms;CK(cudaEventElapsedTime(&ms,a,b));
        CK(cudaMemcpy(back.data(),dev,n,cudaMemcpyDeviceToHost));if(std::memcmp(back.data(),host,n))return 5;
        std::printf("%d,%s,%zu,%.6f,%.6f,1\n",count,mode?"DMA":"pull",n,ms/100,n/(ms/100)/1e6);
    }
    CK(cudaFree(dev));CK(cudaFreeHost(host));CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));CK(cudaStreamDestroy(s));return 0;
}
