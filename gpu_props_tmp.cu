#include <cuda_runtime.h>
#include <cstdio>
int main() {
  cudaDeviceProp p{};
  int d = 0;
  cudaGetDevice(&d);
  cudaGetDeviceProperties(&p, d);
  std::printf("GPU: %s\nSMs: %d\nsharedMemPerMultiprocessor: %zu bytes\nsharedMemPerBlock: %zu bytes\nsharedMemPerBlockOptin: %zu bytes\nmaxThreadsPerMultiProcessor: %d\nmaxBlocksPerMultiProcessor: %d\nregsPerMultiprocessor: %d\n", p.name, p.multiProcessorCount, p.sharedMemPerMultiprocessor, p.sharedMemPerBlock, p.sharedMemPerBlockOptin, p.maxThreadsPerMultiProcessor, p.maxBlocksPerMultiProcessor, p.regsPerMultiprocessor);
  return 0;
}
