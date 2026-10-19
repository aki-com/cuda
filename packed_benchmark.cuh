#pragma once

namespace tetris {
struct PackedBenchmarkResult {
    float coordinateMilliseconds;
    float rowMaskMilliseconds;
    unsigned long long coordinateAccepted;
    unsigned long long rowMaskAccepted;
};

PackedBenchmarkResult benchmark_packed_movement(int repetitions = 20);
}
