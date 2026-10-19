#include "tetris_types.cuh"
#include "packed_mask.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tetris {
namespace {

constexpr int kResultMapCapacity = 128;
constexpr int kMaxResultScenes = kXCount * kRotationCount;
constexpr int kPackedRowOffset = 2;
constexpr int kPackedColumnOffset = 1;
constexpr int kPackedFloorRow = kHeight + kPackedRowOffset;

__device__ __forceinline__ unsigned long long global_timer_ns() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

__device__ std::uint32_t hash_rows(const std::uint16_t* rows) {
    std::uint32_t hash = 2166136261u;
    for (int row = 0; row < kHeight; ++row) {
        hash ^= rows[row];
        hash *= 16777619u;
    }
    return hash;
}

__device__ void make_hard_drop_board(const Board& board, const Piece& piece,
                                     const State& state, int dropY, std::uint16_t* rows) {
    #pragma unroll
    for (int row = 0; row < kHeight; ++row) rows[row] = board.rows[row];

    // The piece is already known to be at a legal hard-drop landing. Build
    // directly in the candidate's shared-memory rows instead of materializing
    // a 32-row packed field and a second 20-row local array per thread.
    const unsigned int shape = piece.rotations[state.rotation];
    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        const unsigned int shapeRow = (shape >> (row * 4)) & 0xfu;
        if (shapeRow == 0) continue;
        const int boardRow = dropY + row;
        if (boardRow < 0 || boardRow >= kHeight) continue;
        const unsigned int positioned = state.x < 0
            ? shapeRow >> -state.x
            : shapeRow << state.x;
        rows[boardRow] |= static_cast<std::uint16_t>(positioned & ((1u << kWidth) - 1));
    }

    constexpr std::uint16_t fullRow = (1u << kWidth) - 1;
    int writeRow = kHeight - 1;
    for (int readRow = kHeight - 1; readRow >= 0; --readRow) {
        if (rows[readRow] == fullRow) continue;
        rows[writeRow--] = rows[readRow];
    }
    while (writeRow >= 0) rows[writeRow--] = 0;
}

__global__ void search_scene_kernel(const Board* inputBoards, Piece piece,
                                    int spawnX, int spawnY, int spawnRotation,
                                    int inputSceneCount,
                                    std::uint16_t* outputParent,
                                    std::uint16_t* outputBoards, int* outputStateIds,
                                    int* outputResultYs, int* outputMoveCounts, int* outputResultCount,
                                    int* outputError, unsigned long long* outputPhaseCycles,
                                    unsigned int* outputLayerCandidateCount,
                                    const unsigned int* stopFlag,
                                    unsigned int* processedInputCount,
                                    unsigned int* outputQueueOverflow) {
    // One CTA owns one input scene. Its threads process states at the same BFS depth.
    const int sceneIndex = blockIdx.x;
    if (sceneIndex >= inputSceneCount) return;
    if (stopFlag && *reinterpret_cast<const volatile unsigned int*>(stopFlag) != 0) return;
    if (outputQueueOverflow &&
        *reinterpret_cast<const volatile unsigned int*>(outputQueueOverflow) != 0) return;
    if (threadIdx.x == 0 && processedInputCount) atomicAdd(processedInputCount, 1u);
    const Board board = inputBoards[sceneIndex];
    std::uint16_t* sceneParent = outputParent ? outputParent + sceneIndex * kStateCount : nullptr;
    std::uint16_t* sceneBoards = outputBoards + sceneIndex * kMaxResultScenes * kHeight;
    int* sceneStateIds = outputStateIds ? outputStateIds + sceneIndex * kMaxResultScenes : nullptr;
    int* sceneYs = outputResultYs ? outputResultYs + sceneIndex * kMaxResultScenes : nullptr;
    int* sceneMoveCounts = outputMoveCounts ? outputMoveCounts + sceneIndex * kMaxResultScenes : nullptr;
    int* sceneResultCount = outputResultCount ? outputResultCount + sceneIndex : nullptr;
    int* sceneError = outputError ? outputError + sceneIndex : nullptr;
    unsigned long long* scenePhaseCycles = outputPhaseCycles ? outputPhaseCycles + sceneIndex * 4 : nullptr;
    // Every encoded state is in [0, 1055], so 16 bits are sufficient here.
    // The BFS frontier is the only queue payload; keep its two buffers compact.
    __shared__ std::uint16_t frontier[2][kFrontierCapacity];
    __shared__ unsigned int visitedBits[kVisitedWordCount];
    __shared__ std::uint16_t parent[kStateCount];
    __shared__ std::uint8_t depth[kStateCount];
    __shared__ std::uint8_t resultMap[kResultMapCapacity];
    __shared__ std::uint32_t resultHashes[kResultMapCapacity];
    __shared__ int frontierCount;
    __shared__ int nextCount;
    __shared__ int resultCount;
    __shared__ int currentFrontier;
    __shared__ int maxFrontierCount;
    __shared__ int queueOverflowDetected;
    __shared__ packed::Mask packedField;
    __shared__ packed::ShapeRows packedPieces[kRotationCount];
    __shared__ unsigned long long phaseStart;
    __shared__ std::uint16_t candidateBoards[kMaxResultScenes][kHeight];
    __shared__ std::uint32_t candidateHashes[kMaxResultScenes];
    __shared__ std::uint16_t candidateStateIds[kMaxResultScenes];
    __shared__ std::uint8_t candidateResultYs[kMaxResultScenes];
    __shared__ std::uint8_t candidateMoveCounts[kMaxResultScenes];
    __shared__ int bestStateForDrop[kMaxResultScenes];
    __shared__ int landingYForDrop[kMaxResultScenes];

    const int tid = threadIdx.x;
    for (int i = tid; i < kVisitedWordCount; i += blockDim.x) visitedBits[i] = 0;
    for (int i = tid; i < kStateCount; i += blockDim.x) {
        parent[i] = 0xffffu;
        depth[i] = 0;
    }
    for (int i = tid; i < kResultMapCapacity; i += blockDim.x) resultMap[i] = 0;
    for (int i = tid; i < kMaxResultScenes; i += blockDim.x) {
        bestStateForDrop[i] = -1;
        landingYForDrop[i] = -1;
    }
    for (int row = tid; row < packed::kRows; row += blockDim.x) {
        std::uint16_t packedRow = 0;
        if (row < kPackedFloorRow) {
            // Board cells occupy bits 1..10; padding bits form side walls.
            packedRow = static_cast<std::uint16_t>(1u | (0x1fu << (kWidth + 1)));
            if (row >= kPackedRowOffset && row < kPackedFloorRow) {
                packedRow |= static_cast<std::uint16_t>(board.rows[row - kPackedRowOffset] << kPackedColumnOffset);
            }
        } else {
            packedRow = 0xffffu;
        }
        packedField.row[row] = packedRow;
    }
    __shared__ int initialId;
    if (tid == 0) {
        int topOccupiedRow = kHeight;
        for (int row = 0; row < kHeight; ++row) {
            if (board.rows[row] != 0) { topOccupiedRow = row; break; }
        }
    int lowestShapeRow = 0;
        for (int rotation = 0; rotation < kRotationCount; ++rotation)
            for (int row = 0; row < 4; ++row)
                if (((piece.rotations[rotation] >> (row * 4)) & 0xfu) != 0)
                    lowestShapeRow = max(lowestShapeRow, row);
        // Keep a full board-width of falling distance so the skipped empty-air
        // prefix cannot remove a legal horizontal traversal before landing.
        const int startY = max(spawnY, topOccupiedRow - lowestShapeRow - 2 - kXCount);
        initialId = state_id(spawnX, startY, spawnRotation);
        for (int rotation = 0; rotation < kRotationCount; ++rotation)
            packedPieces[rotation] = packed::make_shape_rows(piece.rotations[rotation]);
        atomicOr(&visitedBits[initialId >> 5], 1u << (initialId & 31));
        frontier[0][0] = initialId;
        const State initialState = decode_state(initialId);
        const int initialKey = (initialState.x - kMinX) * kRotationCount + initialState.rotation;
        bestStateForDrop[initialKey] = initialId;
        frontierCount = 1;
        nextCount = 0;
        maxFrontierCount = 1;
        queueOverflowDetected = 0;
        resultCount = 0;
        currentFrontier = 0;
        if (sceneResultCount) *sceneResultCount = 0;
        if (sceneError) *sceneError = 0;
        phaseStart = global_timer_ns();
    }
    __syncthreads();

    while (frontierCount > 0) {
        const int readBuffer = currentFrontier;
        const int writeBuffer = readBuffer ^ 1;
        const int levelCount = frontierCount;

        for (int index = tid; index < levelCount; index += blockDim.x) {
            const int id = static_cast<int>(frontier[readBuffer][index]);
            const State state = decode_state(id);

            const packed::ShapeRows& currentShape = packedPieces[state.rotation];
            if (!packed::can_place(currentShape, packedField, state.x, state.y,
                                   kMinX, kWidth - 1, kPackedRowOffset,
                                   kPackedColumnOffset)) continue;

            // Record the hard-drop landing while BFS visits the falling states.
            // This keeps H as one action in the path and removes a second drop scan.
            if (!packed::can_place(currentShape, packedField, state.x, state.y + 1,
                                   kMinX, kWidth - 1, kPackedRowOffset,
                                   kPackedColumnOffset)) {
                const int key = (state.x - kMinX) * kRotationCount + state.rotation;
                atomicCAS(&landingYForDrop[key], -1, state.y);
                continue;
            }

            // A result path includes the final hard drop, so only states at
            // depth <= kMaxMoveCount - 1 can produce an allowed result.
            if (depth[id] >= kMaxMoveCount - 1) continue;

            #pragma unroll
            for (int move = 0; move < kActionsPerState; ++move) {
                int x = state.x;
                int y = state.y;
                int rotation = state.rotation;
                if (move == 0) {
                    --x;
                } else if (move == 1) {
                    ++x;
                } else if (move == 2) {
                    rotation = (rotation + 1) & 3;
                } else if (move == 3) {
                    rotation = (rotation + 3) & 3;
                } else {
                    ++y;
                }
                const bool valid = packed::can_place(packedPieces[rotation], packedField,
                    x, y, kMinX, kWidth - 1, kPackedRowOffset, kPackedColumnOffset);
                if (!valid || x < kMinX || x >= kWidth || y < kMinY || y >= kHeight) continue;

                const int candidate = state_id(x, y, rotation);
                const unsigned int visitedMask = 1u << (candidate & 31);
                if ((atomicOr(&visitedBits[candidate >> 5], visitedMask) & visitedMask) == 0) {
                    parent[candidate] = static_cast<std::uint16_t>(id);
                    depth[candidate] = depth[id] + 1;
                    const int dropKey = (x - kMinX) * kRotationCount + rotation;
                    atomicCAS(&bestStateForDrop[dropKey], -1, candidate);
                    const int output = atomicAdd(&nextCount, 1);
                    if (output < kFrontierCapacity) {
                        frontier[writeBuffer][output] = static_cast<std::uint16_t>(candidate);
                    } else {
                        atomicExch(&queueOverflowDetected, 1);
                    }
                }
            }
        }

        __syncthreads();
        if (tid == 0) {
            if (nextCount > maxFrontierCount) maxFrontierCount = nextCount;
            frontierCount = queueOverflowDetected ? 0 : nextCount;
            nextCount = 0;
            currentFrontier ^= 1;
        }
        __syncthreads();
    }
    if (queueOverflowDetected) {
        if (tid == 0) {
            if (sceneError) *sceneError = 2;
            if (sceneResultCount) *sceneResultCount = 0;
            if (outputQueueOverflow) atomicExch(outputQueueOverflow, 1u);
        }
        return;
    }
    if (tid == 0) {
        if (scenePhaseCycles) scenePhaseCycles[0] = global_timer_ns() - phaseStart;
        phaseStart = global_timer_ns();
    }
    __syncthreads();

    // Construct each candidate from data recorded during BFS: the first state
    // gives the shortest path, and the grounded state gives the landing y.
    for (int key = tid; key < kMaxResultScenes; key += blockDim.x) {
        const int bestId = bestStateForDrop[key];
        candidateStateIds[key] = 0xffffu;
        if (bestId >= 0 && landingYForDrop[key] >= 0) {
            candidateStateIds[key] = static_cast<std::uint16_t>(bestId);
            const State state = decode_state(bestId);
            const int dropY = landingYForDrop[key];
            std::uint16_t* rows = candidateBoards[key];
            make_hard_drop_board(board, piece, state, dropY, rows);
            candidateHashes[key] = hash_rows(rows);
            candidateStateIds[key] = bestId;
            candidateResultYs[key] = static_cast<std::uint8_t>(dropY);
            candidateMoveCounts[key] = static_cast<std::uint8_t>(depth[bestId] + 1);
        }
    }
    __syncthreads();

    if (tid == 0) {
        for (int key = 0; key < kMaxResultScenes; ++key) {
            const int id = candidateStateIds[key];
            if (id == 0xffffu) continue;
            const std::uint32_t hash = candidateHashes[key];
            int slot = static_cast<int>(hash) & (kResultMapCapacity - 1);
            for (int probe = 0; probe < kResultMapCapacity; ++probe) {
                const int entry = resultMap[slot];
                if (entry == 0) {
                    if (resultCount >= kMaxResultScenes) {
                        if (sceneError) *sceneError = 1;
                        break;
                    }
                    const int resultIndex = resultCount++;
                    resultMap[slot] = static_cast<std::uint8_t>(resultIndex + 1);
                    resultHashes[slot] = hash;
                    for (int row = 0; row < kHeight; ++row)
                        sceneBoards[resultIndex * kHeight + row] = candidateBoards[key][row];
                    if (sceneStateIds) sceneStateIds[resultIndex] = id;
                    if (sceneYs) sceneYs[resultIndex] = candidateResultYs[key];
                    if (sceneMoveCounts) sceneMoveCounts[resultIndex] = candidateMoveCounts[key];
                    break;
                }
                const int resultIndex = entry - 1;
                bool equal = resultHashes[slot] == hash;
                if (equal) {
                    for (int row = 0; row < kHeight; ++row) {
                        if (sceneBoards[resultIndex * kHeight + row] != candidateBoards[key][row]) {
                            equal = false;
                            break;
                        }
                    }
                }
                if (equal) {
                    if (!sceneMoveCounts || candidateMoveCounts[key] < sceneMoveCounts[resultIndex]) {
                        if (sceneStateIds) sceneStateIds[resultIndex] = id;
                        if (sceneYs) sceneYs[resultIndex] = candidateResultYs[key];
                        if (sceneMoveCounts) sceneMoveCounts[resultIndex] = candidateMoveCounts[key];
                    }
                    break;
                }
                slot = (slot + 1) & (kResultMapCapacity - 1);
                if (probe == kResultMapCapacity - 1 && sceneError) *sceneError = 1;
            }
        }
        if (scenePhaseCycles) scenePhaseCycles[1] = global_timer_ns() - phaseStart;
    }
    __syncthreads();

    if (tid == 0 && scenePhaseCycles) phaseStart = global_timer_ns();
    __syncthreads();
    if (scenePhaseCycles && sceneParent) {
        for (int i = tid; i < kStateCount; i += blockDim.x) {
            const std::uint16_t previousId = parent[i];
            if (sceneParent) sceneParent[i] = previousId;
        }
    }
    __syncthreads();
    if (tid == 0 && scenePhaseCycles) scenePhaseCycles[2] = global_timer_ns() - phaseStart;
    if (tid == 0 && scenePhaseCycles) scenePhaseCycles[3] = static_cast<unsigned long long>(maxFrontierCount);
    if (tid == 0 && sceneResultCount) *sceneResultCount = resultCount;
    if (tid == 0 && outputLayerCandidateCount) atomicAdd(outputLayerCandidateCount, static_cast<unsigned int>(resultCount));
}

__device__ void insert_unique_board(const std::uint16_t* rows,
                                    unsigned long long* hashTable, int hashCapacity,
                                    std::uint16_t* uniqueBoards, unsigned int uniqueCapacity,
                                    unsigned int* uniqueCount, unsigned int* overflow) {
    std::uint64_t hash = 1469598103934665603ull;
    #pragma unroll
    for (int row = 0; row < kHeight; ++row) {
        hash ^= rows[row];
        hash *= 1099511628211ull;
    }
    std::uint32_t fingerprint = static_cast<std::uint32_t>(hash >> 32);
    if (fingerprint == 0xffffffffu) --fingerprint;
    int slot = static_cast<int>(hash) & (hashCapacity - 1);
    for (int probe = 0; probe < hashCapacity; ++probe) {
        unsigned long long existing = atomicAdd(&hashTable[slot], 0ull);
        if (existing == 0) {
            existing = atomicCAS(&hashTable[slot], 0ull, 0xffffffffffffffffull);
            if (existing == 0) {
                const unsigned int outputIndex = atomicAdd(uniqueCount, 1u);
                const bool fits = outputIndex < uniqueCapacity;
                if (fits) {
                    for (int row = 0; row < kHeight; ++row)
                        uniqueBoards[static_cast<std::uint64_t>(outputIndex) * kHeight + row] = rows[row];
                } else {
                    atomicExch(overflow, 1u);
                }
                __threadfence();
                const unsigned int encodedIndex = fits ? outputIndex + 1 : 0xffffffffu;
                const unsigned long long entry = (static_cast<unsigned long long>(fingerprint) << 32) | encodedIndex;
                atomicExch(&hashTable[slot], entry);
                return;
            }
        }
        if (existing == 0xffffffffffffffffull) {
            do { existing = atomicAdd(&hashTable[slot], 0ull); }
            while (existing == 0xffffffffffffffffull);
        }
        if (static_cast<std::uint32_t>(existing >> 32) == fingerprint) {
            const unsigned int encodedIndex = static_cast<std::uint32_t>(existing);
            if (encodedIndex == 0xffffffffu) return;
            const std::uint16_t* other = uniqueBoards + static_cast<std::uint64_t>(encodedIndex - 1) * kHeight;
            bool equal = true;
            for (int row = 0; row < kHeight; ++row) {
                if (other[row] != rows[row]) { equal = false; break; }
            }
            if (equal) return;
        }
        slot = (slot + 1) & (hashCapacity - 1);
    }
}

__global__ void dedupe_layer_kernel(const std::uint16_t* candidates,
                                    const int* perSceneCounts, int sceneCount,
                                    unsigned long long* hashTable,
                                    int hashCapacity, std::uint16_t* uniqueBoards,
                                    unsigned int uniqueCapacity, unsigned int* uniqueCount,
                                    unsigned int* overflow,
                                    const unsigned int* queueOverflow) {
    const std::uint64_t candidateIndex = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t candidateCapacity = static_cast<std::uint64_t>(sceneCount) * kMaxResultScenes;
    if (candidateIndex >= candidateCapacity) return;
    if (*reinterpret_cast<volatile unsigned int*>(overflow) != 0) return;
    if (*reinterpret_cast<const volatile unsigned int*>(queueOverflow) != 0) return;
    const int sceneIndex = static_cast<int>(candidateIndex / kMaxResultScenes);
    const int localIndex = static_cast<int>(candidateIndex % kMaxResultScenes);
    if (localIndex >= perSceneCounts[sceneIndex]) return;

    const std::uint16_t* rows = candidates + candidateIndex * kHeight;
    insert_unique_board(rows, hashTable, hashCapacity, uniqueBoards,
                        uniqueCapacity, uniqueCount, overflow);
}

__global__ void dedupe_flat_boards_kernel(const std::uint16_t* candidates, unsigned int candidateCount,
                                         unsigned long long* hashTable, int hashCapacity,
                                         std::uint16_t* uniqueBoards, unsigned int uniqueCapacity,
                                         unsigned int* uniqueCount, unsigned int* overflow) {
    const unsigned int candidateIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidateIndex >= candidateCount || *reinterpret_cast<volatile unsigned int*>(overflow)) return;
    const std::uint16_t* rows = candidates + static_cast<std::uint64_t>(candidateIndex) * kHeight;
    insert_unique_board(rows, hashTable, hashCapacity, uniqueBoards,
                        uniqueCapacity, uniqueCount, overflow);
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)), "cudaMalloc");
    }
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : data_(other.data_) { other.data_ = nullptr; }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (data_) cudaFree(data_);
            data_ = other.data_;
            other.data_ = nullptr;
        }
        return *this;
    }
    T* get() const { return data_; }
private:
    T* data_ = nullptr;
};

std::uint16_t rotate_mask(std::uint16_t mask, int turns) {
    for (int turn = 0; turn < turns; ++turn) {
        std::uint16_t rotated = 0;
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                if (mask & (1u << (row * 4 + col))) {
                    rotated |= static_cast<std::uint16_t>(1u << (col * 4 + (3 - row)));
                }
            }
        }
        mask = rotated;
    }
    return mask;
}

char action_between(int previousId, int currentId) {
    const State previous = decode_state(previousId);
    const State current = decode_state(currentId);
    if (current.x < previous.x) return 'L';
    if (current.x > previous.x) return 'R';
    if (current.y > previous.y) return 'D';
    if ((current.rotation - previous.rotation + kRotationCount) % kRotationCount == 1) return 'C';
    if ((previous.rotation - current.rotation + kRotationCount) % kRotationCount == 1) return 'A';
    throw std::runtime_error("BFS parent and child are not connected by one legal action.");
}

std::vector<char> reconstruct_path(int id, const std::vector<std::uint16_t>& parent,
                                   bool appendHardDrop = true) {
    std::vector<char> path;
    for (int steps = 0; ; ++steps) {
        if (id < 0 || id >= kStateCount || steps > kStateCount) {
            throw std::runtime_error("Invalid or cyclic BFS parent chain.");
        }
        const int predecessor = parent[id] == 0xffffu ? -1 : parent[id];
        if (predecessor < 0) break;
        path.push_back(action_between(predecessor, id));
        id = predecessor;
    }
    std::reverse(path.begin(), path.end());
    if (appendHardDrop) path.push_back('H');
    return path;
}

} // namespace

Piece make_t_piece() {
    const std::uint16_t base = (1u << (1 * 4 + 1)) |
        (1u << (2 * 4 + 0)) | (1u << (2 * 4 + 1)) | (1u << (2 * 4 + 2));
    Piece piece{};
    for (int rotation = 0; rotation < kRotationCount; ++rotation) {
        piece.rotations[rotation] = rotate_mask(base, rotation);
    }
    return piece;
}

Piece make_piece(int pieceIndex) {
    static constexpr std::uint16_t bases[7] = {
        // I, O, T, S, Z, J, L; rows are stored top-to-bottom in a 4x4 box.
        0x00f0u,
        0x0660u,
        static_cast<std::uint16_t>((1u << 5) | (1u << 8) | (1u << 9) | (1u << 10)),
        static_cast<std::uint16_t>((1u << 5) | (1u << 6) | (1u << 8) | (1u << 9)),
        static_cast<std::uint16_t>((1u << 4) | (1u << 5) | (1u << 9) | (1u << 10)),
        static_cast<std::uint16_t>((1u << 4) | (1u << 8) | (1u << 9) | (1u << 10)),
        static_cast<std::uint16_t>((1u << 6) | (1u << 8) | (1u << 9) | (1u << 10))
    };
    if (pieceIndex < 0 || pieceIndex >= 7) throw std::invalid_argument("pieceIndex must be in [0, 6].");
    Piece piece{};
    for (int rotation = 0; rotation < kRotationCount; ++rotation)
        piece.rotations[rotation] = rotate_mask(bases[pieceIndex], rotation);
    return piece;
}

const char* piece_name(int pieceIndex) {
    static constexpr const char* names[] = {"I", "O", "T", "S", "Z", "J", "L"};
    if (pieceIndex < 0 || pieceIndex >= 7) return "?";
    return names[pieceIndex];
}

void search_one_scene(const Board& board, const Piece& piece,
                      int spawnX, int spawnY, int spawnRotation,
                      std::vector<SceneResult>& results,
                      double& elapsedMilliseconds,
                      int threadsPerBlock,
                      int* peakQueueSize) {
    if (threadsPerBlock <= 0 || threadsPerBlock > 1024) {
        throw std::invalid_argument("threadsPerBlock must be in [1, 1024].");
    }
    if (spawnY < kMinY || spawnY >= kHeight || spawnX < kMinX || spawnX >= kWidth ||
        spawnRotation < 0 || spawnRotation >= kRotationCount ||
        !can_place(board, piece, spawnX, spawnY, spawnRotation)) {
        throw std::invalid_argument("Initial piece cannot spawn on this board.");
    }

    int topOccupiedRow = kHeight;
    for (int row = 0; row < kHeight; ++row) {
        if (board.rows[row] != 0) {
            topOccupiedRow = row;
            break;
        }
    }
    int lowestOccupiedShapeRow = 0;
    for (int rotation = 0; rotation < kRotationCount; ++rotation) {
        for (int row = 0; row < 4; ++row) {
            if (((piece.rotations[rotation] >> (row * 4)) & 0xfu) != 0)
                lowestOccupiedShapeRow = std::max(lowestOccupiedShapeRow, row);
        }
    }
    // Keep one extra empty row below the piece so every rotation can still make
    // a D transition before contact disables further airborne controls.
    const int searchStartY = std::max(spawnY,
        topOccupiedRow - lowestOccupiedShapeRow - 2 - kXCount);
    const int skippedDropSteps = searchStartY - spawnY;
    if (!can_place(board, piece, spawnX, searchStartY, spawnRotation)) {
        throw std::logic_error("Computed empty-air search start is not collision-free.");
    }
    std::cerr << "Empty-air fast-forward: " << skippedDropSteps
              << " D steps; start y=" << searchStartY
              << ", top occupied row=" << topOccupiedRow << '\n';

    DeviceBuffer<Board> dInputBoards(1);
    check_cuda(cudaMemcpy(dInputBoards.get(), &board, sizeof(Board), cudaMemcpyHostToDevice), "copy input board");
    DeviceBuffer<std::uint16_t> dParent(kStateCount);
    DeviceBuffer<std::uint16_t> dResultBoards(kMaxResultScenes * kHeight);
    DeviceBuffer<int> dResultStateIds(kMaxResultScenes);
    DeviceBuffer<int> dResultYs(kMaxResultScenes);
    DeviceBuffer<int> dResultMoveCounts(kMaxResultScenes);
    DeviceBuffer<int> dResultCount(1);
    DeviceBuffer<int> dError(1);
    DeviceBuffer<unsigned long long> dPhaseCycles(4);
    DeviceBuffer<unsigned int> dQueueOverflow(1);
    check_cuda(cudaMemset(dQueueOverflow.get(), 0, sizeof(unsigned int)), "clear BFS queue overflow flag");

    cudaEvent_t startEvent{}, stopEvent{};
    check_cuda(cudaEventCreate(&startEvent), "create search start event");
    check_cuda(cudaEventCreate(&stopEvent), "create search stop event");
    check_cuda(cudaEventRecord(startEvent), "record search start event");
    search_scene_kernel<<<1, threadsPerBlock>>>(dInputBoards.get(), piece,
        spawnX, spawnY, spawnRotation, 1,
        dParent.get(), dResultBoards.get(), dResultStateIds.get(),
        dResultYs.get(), dResultMoveCounts.get(), dResultCount.get(), dError.get(),
        dPhaseCycles.get(), nullptr, nullptr, nullptr, dQueueOverflow.get());
    check_cuda(cudaGetLastError(), "launch scene search");
    check_cuda(cudaEventRecord(stopEvent), "record search stop event");
    check_cuda(cudaEventSynchronize(stopEvent), "synchronize scene search");
    float kernelMilliseconds = 0.0f;
    check_cuda(cudaEventElapsedTime(&kernelMilliseconds, startEvent, stopEvent), "measure scene search");
    elapsedMilliseconds = kernelMilliseconds;

    unsigned long long phaseCycles[4]{};
    check_cuda(cudaMemcpy(phaseCycles, dPhaseCycles.get(), sizeof(phaseCycles), cudaMemcpyDeviceToHost),
               "read search phase timings");
    if (peakQueueSize) *peakQueueSize = static_cast<int>(phaseCycles[3]);
    std::cerr << "GPU phases (BFS / hard-drop + dedupe / path copy): "
              << phaseCycles[0] / 1.0e6 << " / "
              << phaseCycles[1] / 1.0e6 << " / "
              << phaseCycles[2] / 1.0e6 << " ms; peak BFS queue: "
              << phaseCycles[3] << " states / " << kFrontierCapacity << " capacity\n";
    cudaEventDestroy(stopEvent);
    cudaEventDestroy(startEvent);

    int resultCount = 0;
    int error = 0;
    unsigned int queueOverflow = 0;
    check_cuda(cudaMemcpy(&queueOverflow, dQueueOverflow.get(), sizeof(queueOverflow), cudaMemcpyDeviceToHost),
               "read BFS queue capacity status");
    if (queueOverflow) {
        throw std::runtime_error("BFS frontier exceeded the per-scene state-count capacity; results are incomplete.");
    }
    check_cuda(cudaMemcpy(&resultCount, dResultCount.get(), sizeof(int), cudaMemcpyDeviceToHost), "read result count");
    check_cuda(cudaMemcpy(&error, dError.get(), sizeof(int), cudaMemcpyDeviceToHost), "read search status");
    if (error || resultCount < 0 || resultCount > kMaxResultScenes) {
        throw std::runtime_error("GPU result map capacity exceeded or search failed.");
    }
    std::vector<std::uint16_t> resultBoards(resultCount * kHeight);
    std::vector<int> resultStateIds(resultCount);
    std::vector<int> resultYs(resultCount);
    std::vector<int> resultMoveCounts(resultCount);
    std::vector<std::uint16_t> parent(kStateCount);
    if (resultCount > 0) {
        check_cuda(cudaMemcpy(resultBoards.data(), dResultBoards.get(), resultBoards.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost), "copy result boards");
        check_cuda(cudaMemcpy(resultStateIds.data(), dResultStateIds.get(), resultCount * sizeof(int), cudaMemcpyDeviceToHost), "copy result state ids");
        check_cuda(cudaMemcpy(resultYs.data(), dResultYs.get(), resultCount * sizeof(int), cudaMemcpyDeviceToHost), "copy result y positions");
        check_cuda(cudaMemcpy(resultMoveCounts.data(), dResultMoveCounts.get(), resultCount * sizeof(int), cudaMemcpyDeviceToHost), "copy result move counts");
    }
    check_cuda(cudaMemcpy(parent.data(), dParent.get(), kStateCount * sizeof(std::uint16_t), cudaMemcpyDeviceToHost), "copy parents");

    results.clear();
    results.reserve(resultCount);
    for (int i = 0; i < resultCount; ++i) {
        const int id = resultStateIds[i];
        const State state = decode_state(id);
        SceneResult result{};
        for (int row = 0; row < kHeight; ++row) {
            result.board.rows[row] = resultBoards[i * kHeight + row];
        }
        result.x = state.x;
        result.y = resultYs[i];
        result.rotation = state.rotation;
        result.moves = reconstruct_path(id, parent);
        const int groundedId = state_id(state.x, result.y, state.rotation);
        result.trailingDownAsHardDrop = reconstruct_path(groundedId, parent, false);
        while (!result.trailingDownAsHardDrop.empty() &&
               result.trailingDownAsHardDrop.back() == 'D') {
            result.trailingDownAsHardDrop.pop_back();
            ++result.compressedDownSteps;
        }
        result.trailingDownAsHardDrop.push_back('H');
        if (static_cast<int>(result.moves.size()) != resultMoveCounts[i]) {
            throw std::runtime_error("GPU move count does not match reconstructed path.");
        }
        State replay{spawnX, searchStartY, spawnRotation};
        for (std::size_t moveIndex = 0; moveIndex + 1 < result.moves.size(); ++moveIndex) {
            switch (result.moves[moveIndex]) {
            case 'L': --replay.x; break;
            case 'R': ++replay.x; break;
            case 'D': ++replay.y; break;
            case 'C': replay.rotation = (replay.rotation + 1) & 3; break;
            case 'A': replay.rotation = (replay.rotation + 3) & 3; break;
            default: throw std::runtime_error("Unknown action in reconstructed BFS path.");
            }
            if (!can_place(board, piece, replay.x, replay.y, replay.rotation))
                throw std::runtime_error("Reconstructed BFS path contains an illegal transition.");
        }
        if (result.moves.empty() || result.moves.back() != 'H')
            throw std::runtime_error("Reconstructed BFS path is missing its hard-drop action.");
        int replayLandingY = replay.y;
        while (can_place(board, piece, replay.x, replayLandingY + 1, replay.rotation))
            ++replayLandingY;
        if (replay.x != result.x || replay.rotation != result.rotation || replayLandingY != result.y)
            throw std::runtime_error("Reconstructed BFS path does not reach the reported landing.");
        results.push_back(std::move(result));
    }
}

std::vector<LayerMetrics> benchmark_scene_tree(const std::vector<SceneResult>& initialScenes,
                                              int maxLayers, std::uint64_t maxCandidates,
                                              int searchThreads, const Piece* selectedPiece) {
    if (searchThreads <= 0 || searchThreads > 1024) {
        throw std::invalid_argument("searchThreads must be in [1, 1024].");
    }
    std::vector<LayerMetrics> metrics;
    if (initialScenes.empty() || maxLayers <= 0) return metrics;
    std::vector<Board> seedBoards;
    seedBoards.reserve(initialScenes.size());
    for (const auto& scene : initialScenes) seedBoards.push_back(scene.board);
    auto currentBoards = std::make_unique<DeviceBuffer<Board>>(seedBoards.size());
    check_cuda(cudaMemcpy(currentBoards->get(), seedBoards.data(), seedBoards.size() * sizeof(Board),
                          cudaMemcpyHostToDevice), "upload initial scene batch");
    std::uint64_t currentCount = seedBoards.size();
    const Piece piece = selectedPiece ? *selectedPiece : make_t_piece();

    for (int layer = 1; layer <= maxLayers && currentCount != 0; ++layer) {
        const std::uint64_t candidateSlots = currentCount * kMaxResultScenes;
        std::size_t freeBytes = 0, totalBytes = 0;
        check_cuda(cudaMemGetInfo(&freeBytes, &totalBytes), "query GPU memory");
        const std::uint64_t chunkSceneLimit = std::max<std::uint64_t>(1,
            std::min<std::uint64_t>(131072, maxCandidates / kMaxResultScenes));
        const std::uint64_t chunkScenes = std::min(currentCount, chunkSceneLimit);
        const std::size_t chunkSlots = static_cast<std::size_t>(chunkScenes) * kMaxResultScenes;
        const std::size_t chunkBytes = chunkSlots * (sizeof(Board) + sizeof(int));
        const std::size_t currentBytes = static_cast<std::size_t>(currentCount) * sizeof(Board);

        // The open-address table plus worst-case unique board storage is 32 bytes
        // per table slot. Grow it to the largest power of two that fits the VRAM budget.
        const std::size_t memoryBudget = freeBytes * 7 / 10;
        const std::size_t memoryTableLimit = memoryBudget > chunkBytes + currentBytes
            ? (memoryBudget - chunkBytes - currentBytes) / 28 : 0;
        const std::size_t desiredUnique = static_cast<std::size_t>(std::min<std::uint64_t>(candidateSlots,
            memoryTableLimit / 2));
        std::size_t hashCapacity = 2;
        while (hashCapacity < desiredUnique * 2) hashCapacity <<= 1;
        while (hashCapacity > memoryTableLimit && hashCapacity > 2) hashCapacity >>= 1;
        const std::size_t uniqueCapacity = std::min<std::size_t>(candidateSlots, hashCapacity / 2);
        if (hashCapacity > static_cast<std::size_t>(INT_MAX) || uniqueCapacity == 0) {
            std::cerr << "Layer " << layer << " stopped: not enough VRAM for a dedupe table.\n";
            break;
        }
        const std::size_t tempBytes = currentBytes + chunkBytes + hashCapacity * sizeof(unsigned long long) +
            uniqueCapacity * sizeof(Board);
        if (tempBytes > freeBytes * 7 / 10) {
            std::cerr << "Layer " << layer << " stopped: estimated "
                      << (tempBytes / (1024.0 * 1024.0)) << " MiB needed, "
                      << (freeBytes / (1024.0 * 1024.0)) << " MiB free.\n";
            break;
        }

        try {
            DeviceBuffer<std::uint16_t> candidates(chunkSlots * kHeight);
            DeviceBuffer<int> perSceneCounts(static_cast<std::size_t>(chunkScenes));
            DeviceBuffer<unsigned int> layerCandidateCount(1);
            DeviceBuffer<unsigned int> layerInputCount(1);
            DeviceBuffer<unsigned int> overflow(1);
            DeviceBuffer<unsigned int> queueOverflow(1);
            DeviceBuffer<unsigned long long> hashTable(hashCapacity);
            DeviceBuffer<Board> nextBoards(uniqueCapacity);
            DeviceBuffer<unsigned int> nextCount(1);
            check_cuda(cudaMemset(hashTable.get(), 0, hashCapacity * sizeof(std::uint64_t)), "clear layer hash table");
            check_cuda(cudaMemset(nextCount.get(), 0, sizeof(unsigned int)), "clear layer unique count");
            check_cuda(cudaMemset(layerCandidateCount.get(), 0, sizeof(unsigned int)), "clear candidate count");
            check_cuda(cudaMemset(layerInputCount.get(), 0, sizeof(unsigned int)), "clear input count");
            check_cuda(cudaMemset(overflow.get(), 0, sizeof(unsigned int)), "clear layer overflow flag");
            check_cuda(cudaMemset(queueOverflow.get(), 0, sizeof(unsigned int)), "clear BFS queue overflow flag");

            cudaEvent_t startEvent{}, stopEvent{};
            check_cuda(cudaEventCreate(&startEvent), "create layer start event");
            check_cuda(cudaEventCreate(&stopEvent), "create layer stop event");
            std::vector<std::pair<cudaEvent_t, cudaEvent_t>> stageEvents;
            const std::uint64_t chunkCount = (currentCount + chunkSceneLimit - 1) / chunkSceneLimit;
            stageEvents.reserve(static_cast<std::size_t>(chunkCount) * 2);
            check_cuda(cudaEventRecord(startEvent), "record layer start event");
            constexpr int dedupeThreads = 256;
            for (std::uint64_t offset = 0; offset < currentCount; offset += chunkSceneLimit) {
                const int count = static_cast<int>(std::min(chunkSceneLimit, currentCount - offset));
                cudaEvent_t searchStart{}, searchStop{}, dedupeStart{}, dedupeStop{};
                check_cuda(cudaEventCreate(&searchStart), "create search timing event");
                check_cuda(cudaEventCreate(&searchStop), "create search timing event");
                check_cuda(cudaEventCreate(&dedupeStart), "create dedupe timing event");
                check_cuda(cudaEventCreate(&dedupeStop), "create dedupe timing event");
                check_cuda(cudaEventRecord(searchStart), "record search timing start");
                search_scene_kernel<<<count, searchThreads>>>(
                    currentBoards->get() + offset, piece, 3, kMinY, 0, count,
                    nullptr, candidates.get(), nullptr, nullptr, nullptr,
                    perSceneCounts.get(), nullptr, nullptr, layerCandidateCount.get(),
                    overflow.get(), layerInputCount.get(), queueOverflow.get());
                check_cuda(cudaGetLastError(), "launch scene batch search");
                check_cuda(cudaEventRecord(searchStop), "record search timing stop");
                const std::uint64_t slots = static_cast<std::uint64_t>(count) * kMaxResultScenes;
                const unsigned int dedupeBlocks = static_cast<unsigned int>((slots + dedupeThreads - 1) / dedupeThreads);
                check_cuda(cudaEventRecord(dedupeStart), "record dedupe timing start");
                dedupe_layer_kernel<<<dedupeBlocks, dedupeThreads>>>(candidates.get(), perSceneCounts.get(),
                    count, hashTable.get(), static_cast<int>(hashCapacity),
                    reinterpret_cast<std::uint16_t*>(nextBoards.get()), static_cast<unsigned int>(uniqueCapacity),
                    nextCount.get(), overflow.get(), queueOverflow.get());
                check_cuda(cudaGetLastError(), "launch layer dedupe");
                check_cuda(cudaEventRecord(dedupeStop), "record dedupe timing stop");
                stageEvents.emplace_back(searchStart, searchStop);
                stageEvents.emplace_back(dedupeStart, dedupeStop);
            }
            check_cuda(cudaEventRecord(stopEvent), "record layer stop event");
            check_cuda(cudaEventSynchronize(stopEvent), "synchronize scene layer");
            float layerMilliseconds = 0.0f;
            check_cuda(cudaEventElapsedTime(&layerMilliseconds, startEvent, stopEvent), "measure scene layer");
            float searchMilliseconds = 0.0f, dedupeMilliseconds = 0.0f;
            for (std::size_t i = 0; i < stageEvents.size(); i += 2) {
                float elapsed = 0.0f;
                check_cuda(cudaEventElapsedTime(&elapsed, stageEvents[i].first, stageEvents[i].second), "measure search stage");
                searchMilliseconds += elapsed;
                check_cuda(cudaEventElapsedTime(&elapsed, stageEvents[i + 1].first, stageEvents[i + 1].second), "measure dedupe stage");
                dedupeMilliseconds += elapsed;
            }
            for (const auto& events : stageEvents) {
                cudaEventDestroy(events.first);
                cudaEventDestroy(events.second);
            }
            cudaEventDestroy(stopEvent);
            cudaEventDestroy(startEvent);

            unsigned int actualCandidates = 0, uniqueCount = 0, overflowed = 0, processedInputs = 0;
            unsigned int queueOverflowed = 0;
            check_cuda(cudaMemcpy(&actualCandidates, layerCandidateCount.get(), sizeof(actualCandidates), cudaMemcpyDeviceToHost),
                       "read layer candidate count");
            check_cuda(cudaMemcpy(&uniqueCount, nextCount.get(), sizeof(uniqueCount), cudaMemcpyDeviceToHost),
                       "read next layer scene count");
            check_cuda(cudaMemcpy(&overflowed, overflow.get(), sizeof(overflowed), cudaMemcpyDeviceToHost),
                       "read layer capacity status");
            check_cuda(cudaMemcpy(&processedInputs, layerInputCount.get(), sizeof(processedInputs), cudaMemcpyDeviceToHost),
                       "read processed scene count");
            check_cuda(cudaMemcpy(&queueOverflowed, queueOverflow.get(), sizeof(queueOverflowed), cudaMemcpyDeviceToHost),
                       "read BFS queue capacity status");
            if (queueOverflowed) {
                std::cerr << "Layer " << layer << " stopped: a BFS frontier exceeded the fixed capacity of "
                          << kFrontierCapacity << " states; results for this layer are incomplete.\n";
                break;
            }
            metrics.push_back({currentCount, actualCandidates, uniqueCount, layerMilliseconds,
                               searchMilliseconds, dedupeMilliseconds, tempBytes});
            if (overflowed) {
                std::cout << "Layer " << layer << ": VRAM capacity reached after " << processedInputs
                          << " / " << currentCount << " inputs and " << actualCandidates
                          << " candidate boards; at least " << uniqueCapacity << " unique boards stored "
                          << "(counter " << uniqueCount << "); "
                          << layerMilliseconds << " ms total (search " << searchMilliseconds
                          << " ms, dedupe " << dedupeMilliseconds << " ms); temporary "
                          << (tempBytes / (1024.0 * 1024.0)) << " MiB including input; free before layer "
                          << (freeBytes / (1024.0 * 1024.0)) << " MiB." << std::endl;
            } else {
                std::cout << "Layer " << layer << ": " << processedInputs << " inputs -> " << actualCandidates
                          << " candidate boards -> " << uniqueCount << " unique boards; "
                          << layerMilliseconds << " ms total (search " << searchMilliseconds
                          << " ms, dedupe " << dedupeMilliseconds << " ms); temporary "
                          << (tempBytes / (1024.0 * 1024.0)) << " MiB; free before layer "
                          << (freeBytes / (1024.0 * 1024.0)) << " MiB." << std::endl;
            }
            if (overflowed) {
                std::cerr << "Layer " << layer << " reached the VRAM-sized unique-board capacity of "
                          << uniqueCapacity << "; stopped safely before writing beyond the buffer.\n";
                break;
            }
            // Preserve the deduplicated layer on the GPU; only its count crosses to the host.
            auto next = std::make_unique<DeviceBuffer<Board>>(std::move(nextBoards));
            currentBoards = std::move(next);
            currentCount = uniqueCount;
        } catch (const std::exception& error) {
            std::cerr << "Layer " << layer << " stopped during allocation/execution: " << error.what() << '\n';
            break;
        }
    }
    return metrics;
}

CrossPieceDedupeMetrics benchmark_cross_piece_dedupe(
    const std::vector<SceneResult>& parentScenes, int searchThreads, int repetitions) {
    if (searchThreads <= 0 || searchThreads > 1024)
        throw std::invalid_argument("searchThreads must be in [1, 1024].");
    if (repetitions < 3) throw std::invalid_argument("repetitions must be at least 3.");
    CrossPieceDedupeMetrics result{};
    if (parentScenes.empty()) return result;

    std::vector<Board> hostBoards;
    hostBoards.reserve(parentScenes.size());
    for (const auto& scene : parentScenes) hostBoards.push_back(scene.board);
    const std::size_t parentCount = hostBoards.size();
    const std::size_t sceneCount = parentCount * 7;
    const std::size_t slotsPerPiece = parentCount * kMaxResultScenes;
    const std::size_t candidateSlots = sceneCount * kMaxResultScenes;

    DeviceBuffer<Board> inputBoards(parentCount);
    DeviceBuffer<std::uint16_t> candidates(candidateSlots * kHeight);
    DeviceBuffer<int> perSceneCounts(sceneCount);
    DeviceBuffer<unsigned int> queueOverflow(1);
    check_cuda(cudaMemcpy(inputBoards.get(), hostBoards.data(), parentCount * sizeof(Board),
                          cudaMemcpyHostToDevice), "upload cross-piece benchmark inputs");

    cudaEvent_t start{}, stop{};
    check_cuda(cudaEventCreate(&start), "create cross-piece benchmark event");
    check_cuda(cudaEventCreate(&stop), "create cross-piece benchmark event");
    std::vector<float> searchSamples, dedupeSamples;
    searchSamples.reserve(repetitions);
    dedupeSamples.reserve(repetitions);
    constexpr int warmupRuns = 3;

    for (int iteration = -warmupRuns; iteration < repetitions; ++iteration) {
        check_cuda(cudaMemset(queueOverflow.get(), 0, sizeof(unsigned int)), "reset cross-piece queue overflow");

        check_cuda(cudaEventRecord(start), "record cross-piece search start");
        for (int pieceIndex = 0; pieceIndex < 7; ++pieceIndex) {
            search_scene_kernel<<<static_cast<unsigned int>(parentCount), searchThreads>>>(
                inputBoards.get(), make_piece(pieceIndex), 3, kMinY, 0, static_cast<int>(parentCount),
                nullptr,
                candidates.get() + static_cast<std::size_t>(pieceIndex) * slotsPerPiece * kHeight,
                nullptr, nullptr, nullptr,
                perSceneCounts.get() + static_cast<std::size_t>(pieceIndex) * parentCount,
                nullptr, nullptr, nullptr, nullptr, nullptr, queueOverflow.get());
            check_cuda(cudaGetLastError(), "launch cross-piece scene search");
        }
        check_cuda(cudaEventRecord(stop), "record cross-piece search stop");
        check_cuda(cudaEventSynchronize(stop), "synchronize cross-piece search");
        float searchMs = 0.0f;
        check_cuda(cudaEventElapsedTime(&searchMs, start, stop), "measure cross-piece search");
        if (iteration >= 0) {
            searchSamples.push_back(searchMs);
        }
    }

    unsigned int queueOverflowed = 0;
    check_cuda(cudaMemcpy(&queueOverflowed, queueOverflow.get(), sizeof(queueOverflowed), cudaMemcpyDeviceToHost),
               "read cross-piece queue status");
    if (queueOverflowed) throw std::runtime_error("Cross-piece benchmark BFS frontier overflowed.");
    std::vector<int> counts(sceneCount);
    check_cuda(cudaMemcpy(counts.data(), perSceneCounts.get(), sceneCount * sizeof(int),
                          cudaMemcpyDeviceToHost), "read cross-piece candidate counts");
    for (const int sceneCandidates : counts) result.candidateScenes += sceneCandidates;

    // Remove same-mino repeats first, then benchmark only the extra cross-mino pass.
    std::vector<std::uint16_t> hostCandidates(candidateSlots * kHeight);
    check_cuda(cudaMemcpy(hostCandidates.data(), candidates.get(), hostCandidates.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost), "copy candidates for same-mino grouping");
    auto boardLess = [](const Board& lhs, const Board& rhs) {
        for (int row = 0; row < kHeight; ++row) {
            if (lhs.rows[row] != rhs.rows[row]) return lhs.rows[row] < rhs.rows[row];
        }
        return false;
    };
    auto boardEqual = [](const Board& lhs, const Board& rhs) {
        for (int row = 0; row < kHeight; ++row)
            if (lhs.rows[row] != rhs.rows[row]) return false;
        return true;
    };
    std::vector<std::vector<Board>> perPieceBoards(7);
    for (int pieceIndex = 0; pieceIndex < 7; ++pieceIndex) {
        auto& boards = perPieceBoards[pieceIndex];
        for (std::size_t scene = 0; scene < parentCount; ++scene) {
            const std::size_t sceneIndex = static_cast<std::size_t>(pieceIndex) * parentCount + scene;
            for (int local = 0; local < counts[sceneIndex]; ++local) {
                Board board{};
                const std::size_t source = (sceneIndex * kMaxResultScenes + local) * kHeight;
                for (int row = 0; row < kHeight; ++row) board.rows[row] = hostCandidates[source + row];
                boards.push_back(board);
            }
        }
        std::sort(boards.begin(), boards.end(), boardLess);
        boards.erase(std::unique(boards.begin(), boards.end(), boardEqual), boards.end());
        result.withinPieceUniqueScenes += boards.size();
    }
    std::vector<std::uint16_t> crossPieceCandidates;
    crossPieceCandidates.reserve(result.withinPieceUniqueScenes * kHeight);
    for (const auto& boards : perPieceBoards)
        for (const Board& board : boards)
            crossPieceCandidates.insert(crossPieceCandidates.end(), std::begin(board.rows), std::end(board.rows));

    std::size_t hashCapacity = 2;
    while (hashCapacity < result.withinPieceUniqueScenes * 2) hashCapacity <<= 1;
    DeviceBuffer<std::uint16_t> deviceCrossCandidates(crossPieceCandidates.size());
    DeviceBuffer<unsigned long long> hashTable(hashCapacity);
    DeviceBuffer<Board> uniqueBoards(result.withinPieceUniqueScenes);
    DeviceBuffer<unsigned int> uniqueCount(1);
    DeviceBuffer<unsigned int> dedupeOverflow(1);
    check_cuda(cudaMemcpy(deviceCrossCandidates.get(), crossPieceCandidates.data(),
                          crossPieceCandidates.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice),
               "upload within-mino unique boards");
    constexpr int dedupeThreads = 256;
    const int dedupeBlocks = static_cast<int>((result.withinPieceUniqueScenes + dedupeThreads - 1) / dedupeThreads);
    for (int iteration = -warmupRuns; iteration < repetitions; ++iteration) {
        check_cuda(cudaMemset(hashTable.get(), 0, hashCapacity * sizeof(unsigned long long)), "reset cross-piece hash table");
        check_cuda(cudaMemset(uniqueCount.get(), 0, sizeof(unsigned int)), "reset cross-piece unique count");
        check_cuda(cudaMemset(dedupeOverflow.get(), 0, sizeof(unsigned int)), "reset cross-piece dedupe overflow");
        check_cuda(cudaEventRecord(start), "record cross-piece dedupe start");
        dedupe_flat_boards_kernel<<<dedupeBlocks, dedupeThreads>>>(
            deviceCrossCandidates.get(), static_cast<unsigned int>(result.withinPieceUniqueScenes),
            hashTable.get(), static_cast<int>(hashCapacity),
            reinterpret_cast<std::uint16_t*>(uniqueBoards.get()),
            static_cast<unsigned int>(result.withinPieceUniqueScenes), uniqueCount.get(), dedupeOverflow.get());
        check_cuda(cudaGetLastError(), "launch cross-piece-only dedupe");
        check_cuda(cudaEventRecord(stop), "record cross-piece dedupe stop");
        check_cuda(cudaEventSynchronize(stop), "synchronize cross-piece dedupe");
        float dedupeMs = 0.0f;
        check_cuda(cudaEventElapsedTime(&dedupeMs, start, stop), "measure cross-piece dedupe");
        if (iteration >= 0) dedupeSamples.push_back(dedupeMs);
    }

    unsigned int count = 0, overflowed = 0;
    check_cuda(cudaMemcpy(&count, uniqueCount.get(), sizeof(count), cudaMemcpyDeviceToHost),
               "read cross-piece unique count");
    check_cuda(cudaMemcpy(&overflowed, dedupeOverflow.get(), sizeof(overflowed), cudaMemcpyDeviceToHost),
               "read cross-piece dedupe status");
    if (overflowed) throw std::runtime_error("Cross-piece dedupe output capacity overflowed.");
    result.uniqueScenes = count;
    const auto median = [](std::vector<float>& values) {
        std::sort(values.begin(), values.end());
        return static_cast<double>(values[values.size() / 2]);
    };
    result.searchOnlyMilliseconds = median(searchSamples);
    result.dedupeMilliseconds = median(dedupeSamples);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return result;
}

BatchSearchBenchmark benchmark_scene_batch_sizes(
    const std::vector<SceneResult>& parentScenes, int searchThreads, int repetitions) {
    if (searchThreads <= 0 || searchThreads > 1024)
        throw std::invalid_argument("searchThreads must be in [1, 1024].");
    if (repetitions < 3) throw std::invalid_argument("repetitions must be at least 3.");
    BatchSearchBenchmark result{};
    if (parentScenes.empty()) return result;

    int device = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&device), "get active CUDA device");
    check_cuda(cudaGetDeviceProperties(&properties, device), "get CUDA device properties");
    result.multiprocessorCount = properties.multiProcessorCount;
    result.maxThreadsPerMultiprocessor = properties.maxThreadsPerMultiProcessor;
    cudaFuncAttributes attributes{};
    check_cuda(cudaFuncGetAttributes(&attributes, search_scene_kernel), "query search kernel resources");
    result.registersPerThread = attributes.numRegs;
    result.staticSharedBytesPerBlock = attributes.sharedSizeBytes;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &result.residentBlocksPerMultiprocessor, search_scene_kernel, searchThreads, 0),
        "estimate resident scene blocks");

    std::vector<Board> seeds;
    seeds.reserve(parentScenes.size());
    for (const auto& scene : parentScenes) seeds.push_back(scene.board);
    const std::vector<std::uint64_t> batchSizes{1, 34, 68, 136, 272, 544, 1088, 2176, 4096};
    constexpr int warmupRuns = 3;
    for (const std::uint64_t sceneCount : batchSizes) {
        std::vector<Board> hostInputs(static_cast<std::size_t>(sceneCount));
        for (std::size_t i = 0; i < hostInputs.size(); ++i) hostInputs[i] = seeds[i % seeds.size()];
        const std::size_t slots = static_cast<std::size_t>(sceneCount) * kMaxResultScenes;
        DeviceBuffer<Board> inputBoards(static_cast<std::size_t>(sceneCount));
        DeviceBuffer<std::uint16_t> outputBoards(slots * kHeight);
        DeviceBuffer<int> outputCounts(static_cast<std::size_t>(sceneCount));
        DeviceBuffer<unsigned int> queueOverflow(1);
        check_cuda(cudaMemcpy(inputBoards.get(), hostInputs.data(), hostInputs.size() * sizeof(Board),
                              cudaMemcpyHostToDevice), "upload scene batch inputs");
        check_cuda(cudaMemset(queueOverflow.get(), 0, sizeof(unsigned int)), "clear scene batch overflow");
        cudaEvent_t start{}, stop{};
        check_cuda(cudaEventCreate(&start), "create scene batch timing event");
        check_cuda(cudaEventCreate(&stop), "create scene batch timing event");
        std::vector<float> samples;
        samples.reserve(repetitions);
        for (int iteration = -warmupRuns; iteration < repetitions; ++iteration) {
            check_cuda(cudaEventRecord(start), "record scene batch start");
            search_scene_kernel<<<static_cast<unsigned int>(sceneCount), searchThreads>>>(
                inputBoards.get(), make_t_piece(), 3, kMinY, 0, static_cast<int>(sceneCount),
                nullptr, outputBoards.get(), nullptr, nullptr, nullptr,
                outputCounts.get(), nullptr, nullptr, nullptr, nullptr, nullptr, queueOverflow.get());
            check_cuda(cudaGetLastError(), "launch scene batch search");
            check_cuda(cudaEventRecord(stop), "record scene batch stop");
            check_cuda(cudaEventSynchronize(stop), "synchronize scene batch search");
            if (iteration >= 0) {
                float elapsed = 0.0f;
                check_cuda(cudaEventElapsedTime(&elapsed, start, stop), "measure scene batch search");
                samples.push_back(elapsed);
            }
        }
        std::vector<int> hostCounts(static_cast<std::size_t>(sceneCount));
        unsigned int overflowed = 0;
        check_cuda(cudaMemcpy(hostCounts.data(), outputCounts.get(), hostCounts.size() * sizeof(int),
                              cudaMemcpyDeviceToHost), "read scene batch results");
        check_cuda(cudaMemcpy(&overflowed, queueOverflow.get(), sizeof(overflowed), cudaMemcpyDeviceToHost),
                   "read scene batch overflow");
        if (overflowed) throw std::runtime_error("A scene batch exceeded the fixed BFS frontier capacity.");
        BatchSearchTiming timing{};
        timing.sceneCount = sceneCount;
        for (const int count : hostCounts) timing.candidateCount += count;
        std::sort(samples.begin(), samples.end());
        timing.medianMilliseconds = samples[samples.size() / 2];
        result.timings.push_back(timing);
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
    }
    return result;
}

} // namespace tetris
