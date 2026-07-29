#include "small_matrix_detail.h"

#ifdef PLAMATRIX_SMALL_MATRIX_TEST_HOOKS

namespace plamatrix
{
namespace
{
Index forced_basis_failure_row = -1;
}

void small_matrix_detail::setForcedBasisFailureRow(Index row) noexcept
{
    forced_basis_failure_row = row;
}

Index small_matrix_detail::forcedBasisFailureRow() noexcept
{
    return forced_basis_failure_row;
}

} // namespace plamatrix

#endif
