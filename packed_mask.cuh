#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace tetris::packed {

constexpr int kRows = 32;
constexpr int kColumns = 16;

// Equivalent storage to Rust's u16x32: one 16-bit row per SIMD lane.
struct Mask {
    std::uint16_t row[kRows];
};

// A moving tetromino only touches a few consecutive rows. Keep those rows
// compact and track their absolute lane instead of shifting all 32 lanes.
struct SparseMino {
    std::uint16_t row[4];
    int firstRow;
    int rowCount;
};

struct ShapeRows {
    std::uint16_t row[4];
    int firstRow;
    int rowCount;
};

__device__ inline ShapeRows make_shape_rows(std::uint16_t shape) {
    ShapeRows result{};
    int first = 4;
    int last = -1;
    std::uint16_t source[4];
    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        source[row] = static_cast<std::uint16_t>((shape >> (row * 4)) & 0xfu);
        if (source[row] != 0) {
            if (first == 4) first = row;
            last = row;
        }
    }
    if (last < first) return result;
    result.firstRow = first;
    result.rowCount = last - first + 1;
    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        if (row < result.rowCount) result.row[row] = source[first + row];
    }
    return result;
}

// Test a pre-shifted-by-x shape without copying or shifting its row masks.
// Only the rows containing mino blocks are loaded and checked.
__device__ inline bool can_place(const ShapeRows& shape, const Mask& field,
                                 int x, int y, int minX, int maxX,
                                 int rowOffset, int columnOffset) {
    if (shape.rowCount == 0 || x < minX || x > maxX) return false;
    const int bitShift = x + columnOffset;
    std::uint16_t discardedMask = bitShift < 0
        ? static_cast<std::uint16_t>((1u << -bitShift) - 1u) : 0;
    std::uint16_t anyOverlap = 0;
    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        if (row >= shape.rowCount) break;
        const std::uint16_t bits = shape.row[row];
        if (bitShift < 0 && (bits & discardedMask) != 0) return false;
        const std::uint16_t positioned = bitShift < 0
            ? static_cast<std::uint16_t>(bits >> -bitShift)
            : static_cast<std::uint16_t>(bits << bitShift);
        const int fieldRow = y + rowOffset + shape.firstRow + row;
        if (fieldRow < 0 || fieldRow >= kRows) return false;
        anyOverlap |= positioned & field.row[fieldRow];
    }
    return anyOverlap == 0;
}

__device__ inline SparseMino shift_x(SparseMino value, int dx) {
    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        if (row >= value.rowCount) break;
        if (dx > 0) value.row[row] = static_cast<std::uint16_t>(value.row[row] << dx);
        else if (dx < 0) value.row[row] = static_cast<std::uint16_t>(value.row[row] >> -dx);
    }
    return value;
}

__device__ inline SparseMino shift_y(SparseMino value, int dy) {
    value.firstRow += dy;
    return value;
}

__device__ inline bool overlaps(const SparseMino& mino, const Mask& field) {
    std::uint16_t any = 0;
    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        if (row >= mino.rowCount) break;
        const int fieldRow = mino.firstRow + row;
        if (fieldRow < 0 || fieldRow >= kRows) return true;
        any |= mino.row[row] & field.row[fieldRow];
    }
    return any != 0;
}

__device__ inline void add(SparseMino mino, Mask& field) {
    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        if (row >= mino.rowCount) break;
        const int fieldRow = mino.firstRow + row;
        if (fieldRow >= 0 && fieldRow < kRows) field.row[fieldRow] |= mino.row[row];
    }
}

__device__ inline Mask shift_x(Mask value, int dx) {
    for (int row = 0; row < kRows; ++row) {
        if (dx > 0) value.row[row] = static_cast<std::uint16_t>(value.row[row] << dx);
        else if (dx < 0) value.row[row] = static_cast<std::uint16_t>(value.row[row] >> -dx);
    }
    return value;
}

// Rust rotate_elements_right::<1>() semantics: every lane moves down one,
// and the last lane wraps to lane zero. Wall/floor and bounds checks reject
// wrapped states that leave the playable region.
__device__ inline Mask shift_y(Mask value, int dy) {
    int count = dy % kRows;
    if (count < 0) count += kRows;
    while (count-- > 0) {
        const std::uint16_t last = value.row[kRows - 1];
        for (int row = kRows - 1; row > 0; --row) value.row[row] = value.row[row - 1];
        value.row[0] = last;
    }
    return value;
}

__device__ inline bool overlaps(const Mask& mino, const Mask& field) {
    std::uint16_t any = 0;
    #pragma unroll
    for (int row = 0; row < kRows; ++row) any |= mino.row[row] & field.row[row];
    return any != 0;
}

__device__ inline Mask add(const Mask& mino, const Mask& field) {
    Mask result{};
    #pragma unroll
    for (int row = 0; row < kRows; ++row) result.row[row] = mino.row[row] | field.row[row];
    return result;
}

} // namespace tetris::packed
