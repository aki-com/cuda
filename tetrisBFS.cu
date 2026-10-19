#include "tetris_types.cuh"
#include "packed_benchmark.cuh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace {

struct BoardKey {
    std::array<std::uint16_t, tetris::kHeight> rows{};
    bool operator==(const BoardKey& other) const { return rows == other.rows; }
};

struct BoardKeyHash {
    std::size_t operator()(const BoardKey& key) const noexcept {
        std::size_t hash = 1469598103934665603ull;
        for (const auto row : key.rows) {
            hash ^= row;
            hash *= 1099511628211ull;
        }
        return hash;
    }
};

BoardKey board_key(const tetris::Board& board) {
    BoardKey key{};
    std::copy(std::begin(board.rows), std::end(board.rows), key.rows.begin());
    return key;
}

int bit_count(unsigned int bits) {
    int count = 0;
    while (bits) { count += bits & 1u; bits >>= 1; }
    return count;
}

void print_board(const tetris::Board& board) {
    for (int y = 0; y < tetris::kHeight; ++y) {
        for (int x = 0; x < tetris::kWidth; ++x) {
            const bool occupied = (board.rows[y] & (1u << x)) != 0;
            std::cout << (occupied ? '#' : '.');
        }
        std::cout << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        int searchThreads = 48;
        bool singleOnly = false;
        bool allPieces = false;
        bool crossPieceBench = false;
        bool batchBench = false;
        bool threadBench = false;
        int queueSampleDepth = 0;
        int layerLimit = 6;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--threads" && i + 1 < argc) {
                searchThreads = std::stoi(argv[++i]);
            } else if (arg == "--single-only") {
                singleOnly = true;
            } else if (arg == "--all-pieces") {
                allPieces = true;
            } else if (arg == "--cross-piece-bench") {
                crossPieceBench = true;
            } else if (arg == "--batch-bench") {
                batchBench = true;
            } else if (arg == "--thread-bench") {
                threadBench = true;
            } else if (arg == "--queue-sample") {
                queueSampleDepth = 1;
            } else if (arg == "--queue-sample-depth" && i + 1 < argc) {
                queueSampleDepth = std::stoi(argv[++i]);
            } else if (arg == "--layers" && i + 1 < argc) {
                layerLimit = std::stoi(argv[++i]);
            } else {
                throw std::invalid_argument("Usage: tetrisBFS [--threads N] [--single-only] [--all-pieces] [--cross-piece-bench] [--batch-bench] [--queue-sample-depth N] [--layers N]");
            }
        }
        // One input scene for the first run. This can later be replaced by a scene batch.
        tetris::Board board{};
        constexpr int spawnX = 3;
        constexpr int spawnY = tetris::kMinY;
        constexpr int spawnRotation = 0;

        if (crossPieceBench) {
            std::vector<tetris::SceneResult> parentScenes;
            double parentMilliseconds = 0.0;
            tetris::search_one_scene(board, tetris::make_t_piece(), spawnX, spawnY, spawnRotation,
                parentScenes, parentMilliseconds, searchThreads);
            const auto metrics = tetris::benchmark_cross_piece_dedupe(parentScenes, searchThreads, 21);
            std::cout << "Cross-mino benchmark (" << parentScenes.size()
                      << " parent boards, all 7 minos, median of 21 GPU runs):\n"
                      << "  candidates: " << metrics.candidateScenes << "\n"
                      << "  unique within each mino: " << metrics.withinPieceUniqueScenes << "\n"
                      << "  unique after cross-mino dedupe: " << metrics.uniqueScenes << "\n"
                      << "  search only: " << metrics.searchOnlyMilliseconds << " ms\n"
                      << "  cross-mino dedupe: " << metrics.dedupeMilliseconds << " ms\n"
                      << "  search + dedupe: " << metrics.searchOnlyMilliseconds + metrics.dedupeMilliseconds
                      << " ms\n"
                      << "  estimated speedup without dedupe: "
                      << (metrics.searchOnlyMilliseconds > 0.0
                            ? (metrics.searchOnlyMilliseconds + metrics.dedupeMilliseconds) /
                                  metrics.searchOnlyMilliseconds
                            : 1.0)
                      << "x\n";
            return 0;
        }

        if (batchBench) {
            std::vector<tetris::SceneResult> parentScenes;
            double parentMilliseconds = 0.0;
            tetris::search_one_scene(board, tetris::make_t_piece(), spawnX, spawnY, spawnRotation,
                parentScenes, parentMilliseconds, searchThreads);
            const auto benchmark = tetris::benchmark_scene_batch_sizes(parentScenes, searchThreads, 15);
            std::cout << "Batched T-mino search (median of 15 GPU runs; same 34 parent boards, repeated for larger batches):\n"
                      << "  GPU SMs: " << benchmark.multiprocessorCount
                      << "; max threads/SM: " << benchmark.maxThreadsPerMultiprocessor << '\n'
                      << "  search CTA resources: " << benchmark.registersPerThread << " registers/thread, "
                      << benchmark.staticSharedBytesPerBlock << " static shared-memory bytes/block\n"
                      << "  occupancy estimate: " << benchmark.residentBlocksPerMultiprocessor
                      << " resident blocks/SM => "
                      << benchmark.residentBlocksPerMultiprocessor * benchmark.multiprocessorCount
                      << " simultaneously resident blocks (upper-bound estimate)\n";
            for (const auto& timing : benchmark.timings) {
                std::cout << "  " << timing.sceneCount << " scenes: " << timing.medianMilliseconds
                          << " ms; " << timing.candidateCount << " candidate boards\n";
            }
            return 0;
        }

        if (allPieces) {
            const tetris::Piece parentPiece = tetris::make_t_piece();
            std::vector<tetris::SceneResult> parentScenes;
            double parentMilliseconds = 0.0;
            int parentPeakQueue = 0;
            tetris::search_one_scene(board, parentPiece, spawnX, spawnY, spawnRotation,
                parentScenes, parentMilliseconds, searchThreads, &parentPeakQueue);
            std::cout << "Parent set: " << parentScenes.size()
                      << " unique boards after placing one T mino on the empty field.\n";

            std::unordered_map<BoardKey, unsigned int, BoardKeyHash> pieceMasks;
            std::array<std::unordered_set<BoardKey, BoardKeyHash>, 7> uniqueByPiece;
            std::array<std::uint64_t, 7> candidatesByPiece{};
            for (int pieceIndex = 0; pieceIndex < 7; ++pieceIndex) {
                const tetris::Piece piece = tetris::make_piece(pieceIndex);
                for (const auto& parent : parentScenes) {
                    std::vector<tetris::SceneResult> childScenes;
                    double elapsedMilliseconds = 0.0;
                    int peakQueueSize = 0;
                    tetris::search_one_scene(parent.board, piece, spawnX, spawnY, spawnRotation,
                        childScenes, elapsedMilliseconds, searchThreads, &peakQueueSize);
                    candidatesByPiece[pieceIndex] += childScenes.size();
                    for (const auto& child : childScenes) {
                        const BoardKey key = board_key(child.board);
                        uniqueByPiece[pieceIndex].insert(key);
                        pieceMasks[key] |= 1u << pieceIndex;
                    }
                }
                const auto uniqueCount = uniqueByPiece[pieceIndex].size();
                const double duplicatePercent = candidatesByPiece[pieceIndex] == 0 ? 0.0 :
                    100.0 * (candidatesByPiece[pieceIndex] - uniqueCount) / candidatesByPiece[pieceIndex];
                std::cout << tetris::piece_name(pieceIndex) << ": " << candidatesByPiece[pieceIndex]
                          << " candidates across parent boards -> " << uniqueCount
                          << " unique for this mino (" << duplicatePercent
                          << "% same-mino overlap across parents)\n";
            }

            std::size_t boardsSharedAcrossPieces = 0;
            std::size_t crossPieceDuplicateOccurrences = 0;
            for (const auto& [key, mask] : pieceMasks) {
                (void)key;
                const int types = bit_count(mask);
                if (types > 1) {
                    ++boardsSharedAcrossPieces;
                    crossPieceDuplicateOccurrences += types - 1;
                }
            }
            std::uint64_t candidateTotal = 0;
            for (const auto count : candidatesByPiece) candidateTotal += count;
            std::cout << "Across all seven minos: " << candidateTotal << " candidates; "
                      << pieceMasks.size() << " unique boards; " << boardsSharedAcrossPieces
                      << " boards appear for multiple mino types (" << crossPieceDuplicateOccurrences
                      << " extra cross-mino occurrences).\n";
            return 0;
        }

        const tetris::Piece piece = tetris::make_t_piece();

        if (threadBench) {
            // One fixed input scene; only the cooperative thread count changes.
            // search_one_scene reports CUDA-event kernel time, excluding host copies.
            constexpr int warmups = 3;
            constexpr int repetitions = 21;
            const int threadCounts[] = {1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512, 1024};
            std::unordered_set<BoardKey, BoardKeyHash> referenceBoards;
            bool haveReference = false;
            std::ostringstream suppressedDiagnostics;
            auto* originalCerr = std::cerr.rdbuf(suppressedDiagnostics.rdbuf());
            std::cout << "One-scene T-mino thread-count benchmark (empty board; median of "
                      << repetitions << " CUDA-event kernel timings; same 1 CTA):\n";
            try {
                for (const int threads : threadCounts) {
                    std::vector<double> samples;
                    samples.reserve(repetitions);
                    int peakQueue = 0;
                    std::size_t resultCount = 0;
                    for (int iteration = -warmups; iteration < repetitions; ++iteration) {
                        std::vector<tetris::SceneResult> results;
                        double milliseconds = 0.0;
                        int currentPeak = 0;
                        tetris::search_one_scene(board, piece, spawnX, spawnY, spawnRotation,
                            results, milliseconds, threads, &currentPeak);
                        peakQueue = std::max(peakQueue, currentPeak);
                        resultCount = results.size();
                        if (iteration >= 0) samples.push_back(milliseconds * 1000.0);

                        std::unordered_set<BoardKey, BoardKeyHash> currentBoards;
                        for (const auto& result : results) currentBoards.insert(board_key(result.board));
                        if (!haveReference) {
                            referenceBoards = std::move(currentBoards);
                            haveReference = true;
                        } else if (currentBoards != referenceBoards) {
                            throw std::runtime_error("Thread-count benchmark produced different result boards.");
                        }
                    }
                    std::sort(samples.begin(), samples.end());
                    std::cout << "  " << threads << " threads: " << samples[samples.size() / 2]
                              << " us; results=" << resultCount << "; peak BFS queue=" << peakQueue
                              << '/' << tetris::kFrontierCapacity << "\n";
                }
            } catch (...) {
                std::cerr.rdbuf(originalCerr);
                throw;
            }
            std::cerr.rdbuf(originalCerr);
            return 0;
        }

        std::vector<tetris::SceneResult> results;
        double elapsedMilliseconds = 0.0;
        int peakQueueSize = 0;
        tetris::search_one_scene(board, piece, spawnX, spawnY, spawnRotation,
                                 results, elapsedMilliseconds, searchThreads, &peakQueueSize);

        std::cout << "Unique result scenes: " << results.size() << '\n';
        std::cout << "GPU search time: " << elapsedMilliseconds << " ms\n";
        std::cout << "Search threads: " << searchThreads
                  << "; peak BFS queue: " << peakQueueSize << " / " << tetris::kFrontierCapacity << '\n';
        if (queueSampleDepth > 0) {
            int maximumQueue = peakQueueSize;
            std::vector<tetris::SceneResult> sampleScenes = results;
            for (int depth = 1; depth <= queueSampleDepth && !sampleScenes.empty(); ++depth) {
                std::vector<tetris::SceneResult> nextScenes;
                nextScenes.reserve(sampleScenes.size() * tetris::kXCount * tetris::kRotationCount);
                int depthMaximum = 0;
                for (const auto& scene : sampleScenes) {
                    std::vector<tetris::SceneResult> childResults;
                    double childMilliseconds = 0.0;
                    int childPeakQueue = 0;
                    tetris::search_one_scene(scene.board, piece, spawnX, spawnY, spawnRotation,
                        childResults, childMilliseconds, searchThreads, &childPeakQueue);
                    depthMaximum = std::max(depthMaximum, childPeakQueue);
                    nextScenes.insert(nextScenes.end(), childResults.begin(), childResults.end());
                }
                maximumQueue = std::max(maximumQueue, depthMaximum);
                std::cout << "Queue sample depth " << depth << ": " << sampleScenes.size()
                          << " boards; peak " << depthMaximum << " / " << tetris::kFrontierCapacity
                          << " queue states\n";
                sampleScenes = std::move(nextScenes);
            }
            std::cout << "Maximum queue over sampled searches: " << maximumQueue << " / "
                      << tetris::kFrontierCapacity << " queue states\n";
            return 0;
        }
        if (singleOnly) return 0;
        std::cout << "\nGPU-resident multi-layer scene expansion (start from the "
                  << results.size() << " unique first-layer boards):\n";
        const auto layerMetrics = tetris::benchmark_scene_tree(results, layerLimit, 20'000'000, searchThreads);
        if (layerMetrics.empty()) std::cout << "  No layers were executed.\n";
        if (!results.empty()) {
            const auto demoIt = std::max_element(results.begin(), results.end(),
                [](const auto& lhs, const auto& rhs) {
                    return lhs.compressedDownSteps < rhs.compressedDownSteps;
                });
            const auto& demo = *demoIt;
            std::cout << "\nDrop-suffix demo (paths start at the fast-forward search height; same landing board):\n  current: ";
            for (char move : demo.moves) std::cout << move;
            std::cout << " (" << demo.moves.size() << " actions)\n  grounded path, trailing D -> H: ";
            for (char move : demo.trailingDownAsHardDrop) std::cout << move;
            std::cout << " (" << demo.trailingDownAsHardDrop.size() << " actions; compressed "
                      << demo.compressedDownSteps << " D actions)\n";
        }
        const auto benchmark = tetris::benchmark_packed_movement(30);
        std::cout << "\nMovement microbenchmark (4096 scenes, 256 moves each; average of 30 GPU runs):\n"
                  << "  coordinate + 4-cell collision: " << benchmark.coordinateMilliseconds << " ms\n"
                  << "  compact row-mask collision:    " << benchmark.rowMaskMilliseconds << " ms\n"
                  << "  accepted moves (coordinate / row-mask): "
                  << benchmark.coordinateAccepted << " / " << benchmark.rowMaskAccepted << '\n';
        for (std::size_t i = 0; i < results.size(); ++i) {
            const auto& scene = results[i];
            std::cout << "\nScene " << i
                      << " (x=" << scene.x
                      << ", y=" << scene.y
                      << ", rotation=" << scene.rotation
                      << ", moves-after-fast-forward=";
            for (char move : scene.moves) std::cout << move;
            std::cout << ")\n";
            print_board(scene.board);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Tetris BFS failed: " << error.what() << '\n';
        return 1;
    }
}
