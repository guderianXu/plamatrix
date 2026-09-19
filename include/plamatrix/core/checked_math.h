#pragma once

#include <limits>
#include <stdexcept>
#include <string>

#include "plamatrix/core/types.h"

namespace plamatrix::detail
{

inline Index checkedIndexMul(Index lhs, Index rhs, const char* what)
{
    if (lhs < 0 || rhs < 0 || (rhs != 0 && lhs > std::numeric_limits<Index>::max() / rhs))
    {
        throw std::overflow_error(std::string(what) + " overflows Index");
    }
    return lhs * rhs;
}

inline std::size_t checkedSizeMul(std::size_t lhs, std::size_t rhs, const char* what)
{
    if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs)
    {
        throw std::overflow_error(std::string(what) + " overflows size_t");
    }
    return lhs * rhs;
}

} // namespace plamatrix::detail
