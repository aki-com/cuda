#include "packed_benchmark.cuh"
#include "packed_mask.cuh"

#include <stdexcept>
#include <string>

namespace tetris {
namespace {

constexpr int kWorkItems = 4096;
constexpr int kSteps = 256;

struct Input {
    packed::Mask field;
};

__global__ void coordinate_kernel(const Input* input, unsigned long long* output) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= kWorkItems) return;

    // T-shaped four-cell mask; positions are represented as (x,y), as in the
    // old coordinate collision test. The field has four wall columns per side
    // and a three-row floor.
    constexpr int cellX[4] = {0, 1, 2, 1};
    constexpr int cellY[4] = {1, 1, 1, 0};
    int x = 6;
    int y = 12;
    unsigned long long accepted = 0;
    for (int step = 0; step < kSteps; ++step) {
        const int move = (step + tid) & 3;
        const int dx = move == 0 ? -1 : (move == 1 ? 1 : 0);
        const int dy = move == 2 ? 1 : 0;
        bool valid = true;
        #pragma unroll
        for (int cell = 0; cell < 4; ++cell) {
            const int cx = cellX[cell] + x + dx;
            const int cy = cellY[cell] + y + dy;
            if (cx < 0 || cx >= packed::kColumns || cy < 0 || cy >= packed::kRows ||
                (input->field.row[cy] & (1u << cx)) != 0) valid = false;
        }
        if (valid) {
            x += dx;
            y += dy;
            ++accepted;
        }
    }
    output[tid] = accepted;
}

__global__ void row_mask_kernel(const Input* input, unsigned long long* output) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= kWorkItems) return;

    packed::ShapeRows shape{};
    shape.row[0] = 0x7u;
    shape.row[1] = 0x2u;
    shape.rowCount = 2;
    int x = 6;
    int y = 12;
    unsigned long long accepted = 0;
    for (int step = 0; step < kSteps; ++step) {
        const int move = (step + tid) & 3;
        const int dx = move == 0 ? -1 : (move == 1 ? 1 : 0);
        const int dy = move == 2 ? 1 : 0;
        if (packed::can_place(shape, input->field, x + dx, y + dy, 0, 12, 0, 0)) {
            x += dx;
            y += dy;
            ++accepted;
        }
    }
    output[tid] = accepted;
}

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

template<auto Kernel>
float time_kernel(Input* input, unsigned long long* output, int repetitions) {
    constexpr int threads = 128;
    constexpr int blocks = (kWorkItems + threads - 1) / threads;
    for (int i = 0; i < 3; ++i) Kernel<<<blocks, threads>>>(input, output);
    check(cudaGetLastError(), "benchmark warmup launch");
    check(cudaDeviceSynchronize(), "benchmark warmup synchronize");
    cudaEvent_t begin{}, end{};
    check(cudaEventCreate(&begin), "cudaEventCreate begin");
    check(cudaEventCreate(&end), "cudaEventCreate end");
    check(cudaEventRecord(begin), "cudaEventRecord begin");
    for (int i = 0; i < repetitions; ++i) Kernel<<<blocks, threads>>>(input, output);
    check(cudaGetLastError(), "benchmark launch");
    check(cudaEventRecord(end), "cudaEventRecord end");
    check(cudaEventSynchronize(end), "benchmark synchronize");
    float milliseconds = 0.0f;
    check(cudaEventElapsedTime(&milliseconds, begin, end), "cudaEventElapsedTime");
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return milliseconds / repetitions;
}

} // namespace

PackedBenchmarkResult benchmark_packed_movement(int repetitions) {
    if (repetitions < 1) throw std::invalid_argument("benchmark repetitions must be positive");
    Input host{};
    for (int row = 0; row < packed::kRows; ++row) {
        // Four wall cells on each side; the bottom three rows are floor.
        host.field.row[row] = row >= packed::kRows - 3 ? 0xffffu : 0xf00fu;
    }

    Input* deviceInput = nullptr;
    unsigned long long* deviceOutput = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&deviceInput), sizeof(Input)), "cudaMalloc input");
    check(cudaMalloc(reinterpret_cast<void**>(&deviceOutput), kWorkItems * sizeof(unsigned long long)), "cudaMalloc output");
    check(cudaMemcpy(deviceInput, &host, sizeof(Input), cudaMemcpyHostToDevice), "copy benchmark input");

    PackedBenchmarkResult result{};
    result.coordinateMilliseconds = time_kernel<coordinate_kernel>(deviceInput, deviceOutput, repetitions);
    unsigned long long accepted[kWorkItems];
    check(cudaMemcpy(accepted, deviceOutput, sizeof(accepted), cudaMemcpyDeviceToHost), "read coordinate benchmark");
    for (auto value : accepted) result.coordinateAccepted += value;

    result.rowMaskMilliseconds = time_kernel<row_mask_kernel>(deviceInput, deviceOutput, repetitions);
    check(cudaMemcpy(accepted, deviceOutput, sizeof(accepted), cudaMemcpyDeviceToHost), "read packed benchmark");
    for (auto value : accepted) result.rowMaskAccepted += value;
    check(cudaFree(deviceOutput), "cudaFree output");
    check(cudaFree(deviceInput), "cudaFree input");
    return result;
}

} // namespace tetris
