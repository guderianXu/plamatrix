#pragma once

#include "plamatrix/ops/small_matrix.h"

namespace plamatrix
{
namespace small_matrix_detail
{

struct DeviceStatus
{
    Index nonFiniteRow;
    Index basisFailureRow;
};

struct SymmetricEigh3x3WorkspaceAccess
{
    static bool beginStatusBatch(SymmetricEigh3x3Workspace& workspace) noexcept;
};

#ifdef PLAMATRIX_SMALL_MATRIX_TEST_HOOKS
void setForcedBasisFailureRow(Index row) noexcept;
Index forcedBasisFailureRow() noexcept;
#endif

template <typename Scalar>
void launchSymmetricEigh3x3(
    const DenseMatrix<Scalar, Device::GPU>& input,
    DenseMatrix<Scalar, Device::GPU>& eigenvalues,
    DenseMatrix<Scalar, Device::GPU>& eigenvectors,
    SymmetricEigh3x3Workspace& workspace,
    cudaStream_t stream);

} // namespace small_matrix_detail
} // namespace plamatrix
