#include <cuda_runtime.h>
#include <cstdio>

template<int N> __global__ void probe() {
    __shared__ volatile unsigned char bytes[N];
    bytes[threadIdx.x % N] = static_cast<unsigned char>(threadIdx.x);
    __syncthreads();
}

template<int N> void check() {
    int blocks = 0;
    cudaFuncAttributes a{};
    cudaFuncGetAttributes(&a, probe<N>);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, probe<N>, 48, 0);
    std::printf("N=%d shared=%zu regs=%d resident=%d\n", N, a.sharedSizeBytes, a.numRegs, blocks);
}
int main() {
    check<6988>(); check<6800>(); check<6656>(); check<6400>(); check<6144>();
    check<6000>(); check<5632>(); check<5120>(); check<4096>();
}
