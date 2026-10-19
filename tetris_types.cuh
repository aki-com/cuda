#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstddef>
#include <vector>

namespace tetris {

constexpr int kWidth = 10;
constexpr int kHeight = 20;
constexpr int kMinX = -2;
constexpr int kXCount = kWidth - kMinX;
constexpr int kMinY = -2;
constexpr int kYCount = kHeight - kMinY;
constexpr int kRotationCount = 4;
constexpr int kStateCount = kXCount * kYCount * kRotationCount;
// Per-depth frontier capacity. Overflow is detected and reported by the kernel.
constexpr int kFrontierCapacity = kXCount * kRotationCount;
constexpr int kVisitedWordCount = (kStateCount + 31) / 32;
constexpr int kActionsPerState = 5;
constexpr int kMaxMoveCount = 32; // Includes the final hard-drop action.

struct Board {
    // One 10-bit occupancy mask per row.
    std::uint16_t rows[kHeight]{};
};

struct Piece {
    // Each rotation is a 4x4 bit mask (needed by the I tetromino).
    std::uint16_t rotations[kRotationCount]{};
};

struct SceneResult {
    Board board;
    int x = 0;
    int y = 0;
    int rotation = 0;
    std::vector<char> moves;
    // Demonstration path: reach the grounded state, then replace trailing D steps with H.
    std::vector<char> trailingDownAsHardDrop;
    int compressedDownSteps = 0;
};

struct LayerMetrics {
    std::uint64_t inputScenes = 0;
    std::uint64_t candidateScenes = 0;
    std::uint64_t uniqueScenes = 0;
    double elapsedMilliseconds = 0.0;
    double searchMilliseconds = 0.0;
    double dedupeMilliseconds = 0.0;
    std::size_t peakTemporaryBytes = 0;
};

struct CrossPieceDedupeMetrics {
    std::uint64_t candidateScenes = 0;
    std::uint64_t withinPieceUniqueScenes = 0;
    std::uint64_t uniqueScenes = 0;
    double searchOnlyMilliseconds = 0.0;
    double dedupeMilliseconds = 0.0;
};

struct BatchSearchTiming {
    std::uint64_t sceneCount = 0;
    std::uint64_t candidateCount = 0;
    double medianMilliseconds = 0.0;
};

struct BatchSearchBenchmark {
    int multiprocessorCount = 0;
    int maxThreadsPerMultiprocessor = 0;
    int residentBlocksPerMultiprocessor = 0;
    int registersPerThread = 0;
    std::size_t staticSharedBytesPerBlock = 0;
    std::vector<BatchSearchTiming> timings;
};

struct State {
    int x;
    int y;
    int rotation;
};

__host__ __device__ constexpr int state_id(int x, int y, int rotation) {
    return (((y - kMinY) * kXCount + (x - kMinX)) * kRotationCount) + rotation;
}

__host__ __device__ constexpr State decode_state(int id) {
    const int rotation = id % kRotationCount;
    const int cell = id / kRotationCount;
    return {cell % kXCount + kMinX, cell / kXCount + kMinY, rotation};
}

__host__ __device__ inline bool can_place(const Board& board, const Piece& piece,
                                          int x, int y, int rotation) {
    const unsigned int shape = piece.rotations[rotation];
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            if ((shape & (1u << (row * 4 + col))) == 0) continue;
            const int boardX = x + col;
            const int boardY = y + row;
            if (boardX < 0 || boardX >= kWidth || boardY >= kHeight) return false;
            if (boardY >= 0 && (board.rows[boardY] & (1u << boardX))) return false;
        }
    }
    return true;
}

Piece make_t_piece();
Piece make_piece(int pieceIndex);
const char* piece_name(int pieceIndex);
void search_one_scene(const Board& board, const Piece& piece,
                      int spawnX, int spawnY, int spawnRotation,
                      std::vector<SceneResult>& results,
                      double& elapsedMilliseconds,
                      int threadsPerBlock = 48,
                      int* peakQueueSize = nullptr);
std::vector<LayerMetrics> benchmark_scene_tree(const std::vector<SceneResult>& initialScenes,
                                              int maxLayers, std::uint64_t maxCandidates,
                                              int searchThreads = 48,
                                              const Piece* piece = nullptr);
CrossPieceDedupeMetrics benchmark_cross_piece_dedupe(
    const std::vector<SceneResult>& parentScenes, int searchThreads = 48, int repetitions = 21);
BatchSearchBenchmark benchmark_scene_batch_sizes(
    const std::vector<SceneResult>& parentScenes, int searchThreads = 48, int repetitions = 15);

} // namespace tetris
