#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

#include "plamatrix/internal/core/device.h"

namespace plamatrix::internal
{

inline namespace v1
{

struct Shape2
{
    Index rows = 0;
    Index cols = 0;

    Shape2() = default;

    Shape2(Index row_count, Index col_count)
        : rows(row_count)
        , cols(col_count)
    {
        validate();
    }

    std::size_t elementCount() const
    {
        if (rows == 0 || cols == 0)
        {
            return 0;
        }
        const auto row_count = static_cast<std::size_t>(rows);
        const auto col_count = static_cast<std::size_t>(cols);
        if (row_count > std::numeric_limits<std::size_t>::max() / col_count)
        {
            throw std::overflow_error("Shape2 element count overflows size_t");
        }
        return row_count * col_count;
    }

private:
    void validate() const
    {
        if (rows < 0 || cols < 0)
        {
            throw std::invalid_argument("Shape2 dimensions must be non-negative");
        }
    }
};

struct Shape3
{
    Index first = 0;
    Index second = 0;
    Index third = 0;

    Shape3() = default;

    Shape3(Index first_extent, Index second_extent, Index third_extent)
        : first(first_extent)
        , second(second_extent)
        , third(third_extent)
    {
        if (first < 0 || second < 0 || third < 0)
        {
            throw std::invalid_argument("Shape3 extents must be non-negative");
        }
    }
};

} // namespace v1

} // namespace plamatrix::internal
