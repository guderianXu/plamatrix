#pragma once

#include "plamatrix/internal/ops/small_matrix.h"

namespace plamatrix::internal
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
    const DenseStorage<Scalar, Device::GPU>& input,
    DenseStorage<Scalar, Device::GPU>& eigenvalues,
    DenseStorage<Scalar, Device::GPU>& eigenvectors,
    SymmetricEigh3x3Workspace& workspace,
    cudaStream_t stream);

template <typename Scalar>
void launchSymmetricEigh3x3(
    ConstMatrixView<Scalar, Device::GPU> input,
    MatrixView<Scalar, Device::GPU> eigenvalues,
    MatrixView<Scalar, Device::GPU> eigenvectors,
    SymmetricEigh3x3Workspace& workspace,
    cudaStream_t stream);

} // namespace small_matrix_detail
} // namespace plamatrix::internal
