#include <sstream>
#include <stdexcept>
#include <string>

#include "small_matrix_detail.h"

namespace plamatrix
{
namespace
{

template <typename Scalar>
void validateShapes(
    const char* operation,
    const DenseMatrix<Scalar, Device::GPU>& input,
    const DenseMatrix<Scalar, Device::GPU>* eigenvalues,
    const DenseMatrix<Scalar, Device::GPU>* eigenvectors)
{
    if (input.cols() != 6)
    {
        std::ostringstream message;
        message << operation << ": expected an N x 6 compact matrix, got "
                << input.rows() << " x " << input.cols();
        throw std::invalid_argument(message.str());
    }
    if (eigenvalues != nullptr &&
        (eigenvalues->rows() != input.rows() || eigenvalues->cols() != 3))
    {
        throw std::invalid_argument(std::string(operation) +
                                    ": eigenvalues must have shape N x 3");
    }
    if (eigenvectors != nullptr &&
        (eigenvectors->rows() != input.rows() || eigenvectors->cols() != 9))
    {
        throw std::invalid_argument(std::string(operation) +
                                    ": eigenvectors must have shape N x 9");
    }
}

} // namespace

template <typename Scalar>
SymmetricEigh3x3Result<Scalar, Device::GPU> symmetricEigh3x3Batched(
    const DenseMatrix<Scalar, Device::GPU>& compact_matrices)
{
    validateShapes<Scalar>("symmetricEigh3x3Batched", compact_matrices, nullptr, nullptr);
    SymmetricEigh3x3Result<Scalar, Device::GPU> result{
        DenseMatrix<Scalar, Device::GPU>(compact_matrices.rows(), 3),
        DenseMatrix<Scalar, Device::GPU>(compact_matrices.rows(), 9)
    };
    SymmetricEigh3x3Workspace workspace;
    small_matrix_detail::launchSymmetricEigh3x3(
        compact_matrices, result.eigenvalues, result.eigenvectors, workspace, nullptr);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(nullptr));
    workspace.checkStatus("symmetricEigh3x3Batched");
    return result;
}

template <typename Scalar>
void symmetricEigh3x3Batched(
    const DenseMatrix<Scalar, Device::GPU>& compact_matrices,
    DenseMatrix<Scalar, Device::GPU>& eigenvalues,
    DenseMatrix<Scalar, Device::GPU>& eigenvectors,
    SymmetricEigh3x3Workspace& workspace,
    cudaStream_t stream)
{
    validateShapes("symmetricEigh3x3Batched", compact_matrices, &eigenvalues, &eigenvectors);
    small_matrix_detail::launchSymmetricEigh3x3(
        compact_matrices, eigenvalues, eigenvectors, workspace, stream);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    workspace.checkStatus("symmetricEigh3x3Batched");
}

template <typename Scalar>
void symmetricEigh3x3BatchedAsync(
    const DenseMatrix<Scalar, Device::GPU>& compact_matrices,
    DenseMatrix<Scalar, Device::GPU>& eigenvalues,
    DenseMatrix<Scalar, Device::GPU>& eigenvectors,
    SymmetricEigh3x3Workspace& workspace,
    cudaStream_t stream)
{
    validateShapes(
        "symmetricEigh3x3BatchedAsync", compact_matrices, &eigenvalues, &eigenvectors);
    small_matrix_detail::launchSymmetricEigh3x3(
        compact_matrices, eigenvalues, eigenvectors, workspace, stream);
}

#define PLAMATRIX_INSTANTIATE_SMALL_MATRIX(Scalar)                                      \
    template SymmetricEigh3x3Result<Scalar, Device::GPU> symmetricEigh3x3Batched(       \
        const DenseMatrix<Scalar, Device::GPU>&);                                       \
    template void symmetricEigh3x3Batched(                                              \
        const DenseMatrix<Scalar, Device::GPU>&,                                        \
        DenseMatrix<Scalar, Device::GPU>&, DenseMatrix<Scalar, Device::GPU>&,            \
        SymmetricEigh3x3Workspace&, cudaStream_t);                                      \
    template void symmetricEigh3x3BatchedAsync(                                         \
        const DenseMatrix<Scalar, Device::GPU>&,                                        \
        DenseMatrix<Scalar, Device::GPU>&, DenseMatrix<Scalar, Device::GPU>&,            \
        SymmetricEigh3x3Workspace&, cudaStream_t)

#ifdef PLAMATRIX_USE_FLOAT
PLAMATRIX_INSTANTIATE_SMALL_MATRIX(float);
#endif
#ifdef PLAMATRIX_USE_DOUBLE
PLAMATRIX_INSTANTIATE_SMALL_MATRIX(double);
#endif

#undef PLAMATRIX_INSTANTIATE_SMALL_MATRIX

} // namespace plamatrix
