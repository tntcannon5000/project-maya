// Actual CUDA graph replay, dynamic pointers, side-stream join and failed capture.
#include "../src/core/glm_window_graphs.hpp"
#include <cstdio>
#include <cstdlib>
using Cache = strata::core::glmfast::WindowMoeGraphs;
static void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}
static void check(cudaError_t e) { require(e == cudaSuccess, cudaGetErrorString(e)); }
__global__ void add_selected(const int* const* table, const int* index, int* output, int width) {
    *output += *table[*index] * width;
}
__global__ void add_joined(int* output) { ++*output; }
int main() {
    cudaStream_t stream, side; cudaEvent_t ready, done;
    check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check(cudaStreamCreateWithFlags(&side,cudaStreamNonBlocking));
    check(cudaEventCreateWithFlags(&ready,cudaEventDisableTiming));
    check(cudaEventCreateWithFlags(&done,cudaEventDisableTiming));
    int *values, *index, *output; const int** table;
    check(cudaMalloc(&values,2*sizeof(int)));check(cudaMalloc(&index,sizeof(int)));
    check(cudaMalloc(&output,sizeof(int)));check(cudaMalloc(&table,2*sizeof(int*)));
    check(cudaMemset(output,0,sizeof(int)));
    Cache cache; Cache::Accounts accounts{}; std::string err;
    int wanted=0; uint64_t credits=0;
    for(int step=0;step<12;++step) {
        const int width=step%2 ? 5 : 3, selected=step%2;
        int vals[2]={step+2,step+7};const int* pointers[2]={values+(step%2),values+(1-step%2)};
        check(cudaMemcpyAsync(values,vals,sizeof vals,cudaMemcpyHostToDevice,stream));
        check(cudaMemcpyAsync(index,&selected,sizeof selected,cudaMemcpyHostToDevice,stream));
        check(cudaMemcpyAsync(table,pointers,sizeof pointers,cudaMemcpyHostToDevice,stream));
        Cache::Key key{}; key[0]=step<6 ? 1 : 2;
        const auto body=[&]() {
            check(cudaEventRecord(ready,stream));check(cudaStreamWaitEvent(side,ready,0));
            add_selected<<<1,1,0,side>>>(table,index,output,width);
            check(cudaEventRecord(done,side));check(cudaStreamWaitEvent(stream,done,0));
            add_joined<<<1,1,0,stream>>>(output);
            accounts[0]+=width; ++accounts[1]; ++accounts[3]; ++accounts[4];return true;
        };
        require(cache.enqueue(3,width,key,stream,side,accounts,body,err)==Cache::Result::Enqueued,err.c_str());
        check(cudaStreamSynchronize(stream));
        wanted+=vals[0]*width+1;credits+=width;
        int actual;check(cudaMemcpy(&actual,output,sizeof actual,cudaMemcpyDeviceToHost));
        require(actual==wanted,"dynamic pointer or side-stream replay mismatch");
        require(accounts[0]==credits && accounts[1]==(uint64_t)step+1 && accounts[3]==accounts[1] && accounts[4]==accounts[1],"request/component credits mismatch");
    }
    const auto before=accounts; Cache::Key key{};
    auto fail=[&]() { add_joined<<<1,1,0,stream>>>(output);++accounts[1];return false; };
    require(cache.enqueue(99,3,key,stream,side,accounts,fail,err)==Cache::Result::Fallback,"failed capture should fall back before launch");
    require(accounts==before,"failed capture must restore all host credits");
    require(cache.enqueue(99,3,key,stream,side,accounts,fail,err)==Cache::Result::Fallback,"disabled key must remain bypassed");
    check(cudaStreamSynchronize(stream));int actual;check(cudaMemcpy(&actual,output,sizeof actual,cudaMemcpyDeviceToHost));
    require(actual==wanted,"abandoned capture executed work");
    require(cache.stats.captures==4 && cache.stats.replays==8 && cache.stats.invalidations==2 && cache.stats.capture_failures==1 && cache.stats.disabled_fallbacks==1,"engagement counters mismatch");
    std::printf("PASS window graphs: dynamic pointers, T3/T5, side join, invalidation, exact credits, abandoned capture; %llu captures %llu replays\n",cache.stats.captures,cache.stats.replays);
    cache.clear();check(cudaFree(table));check(cudaFree(values));check(cudaFree(index));check(cudaFree(output));
    check(cudaEventDestroy(ready));check(cudaEventDestroy(done));check(cudaStreamDestroy(side));check(cudaStreamDestroy(stream));
}
