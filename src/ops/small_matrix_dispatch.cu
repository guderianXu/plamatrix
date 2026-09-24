#include <array>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include "small_matrix_detail.h"

namespace plamatrix::internal
{
namespace
{

template <typename Scalar>
void validateShapes(
    const char* operation,
    const DenseStorage<Scalar, Device::GPU>& input,
    const DenseStorage<Scalar, Device::GPU>* eigenvalues,
    const DenseStorage<Scalar, Device::GPU>* eigenvectors)
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

template <typename Scalar>
void validateViewShapes(
    const char* operation,
    ConstMatrixView<Scalar, Device::GPU> input,
    MatrixView<Scalar, Device::GPU> eigenvalues,
    MatrixView<Scalar, Device::GPU> eigenvectors)
{
    if (input.cols() != 6)
    {
        std::ostringstream message;
        message << operation << ": expected an N x 6 compact matrix, got "
                << input.rows() << " x " << input.cols();
        throw std::invalid_argument(message.str());
    }
    if (eigenvalues.rows() != input.rows() || eigenvalues.cols() != 3)
    {
        throw std::invalid_argument(std::string(operation) + ": eigenvalues must have shape N x 3");
    }
    if (eigenvectors.rows() != input.rows() || eigenvectors.cols() != 9)
    {
        throw std::invalid_argument(std::string(operation) + ": eigenvectors must have shape N x 9");
    }
    if (!input.isContiguousColumnMajor() || !eigenvalues.isContiguousColumnMajor() ||
        !eigenvectors.isContiguousColumnMajor())
    {
        throw std::invalid_argument(std::string(operation) + ": views must be contiguous column-major");
    }

    struct Range
    {
        std::uintptr_t begin;
        std::uintptr_t end;
    };
    const auto range = [operation](const Scalar* data, Index count) {
        const auto begin = reinterpret_cast<std::uintptr_t>(data);
        const auto bytes = detail::checkedAllocationBytes<Scalar>(count);
        if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin)
        {
            throw std::overflow_error(std::string(operation) + ": view storage range overflows");
        }
        return Range{begin, begin + bytes};
    };
    const std::array<Range, 3> ranges{
        range(input.data(), input.size()),
        range(eigenvalues.data(), eigenvalues.size()),
        range(eigenvectors.data(), eigenvectors.size())
    };
    for (std::size_t first = 0; first < ranges.size(); ++first)
    {
        for (std::size_t second = first + 1; second < ranges.size(); ++second)
        {
            if (ranges[first].begin < ranges[first].end && ranges[second].begin < ranges[second].end &&
                ranges[first].begin < ranges[second].end && ranges[second].begin < ranges[first].end)
            {
                throw std::invalid_argument(std::string(operation) + ": views must not overlap");
            }
        }
    }
}

} // namespace

template <typename Scalar>
SymmetricEigh3x3Result<Scalar, Device::GPU> symmetricEigh3x3Batched(
    const DenseStorage<Scalar, Device::GPU>& compact_matrices)
{
    validateShapes<Scalar>("symmetricEigh3x3Batched", compact_matrices, nullptr, nullptr);
    SymmetricEigh3x3Result<Scalar, Device::GPU> result{
        DenseStorage<Scalar, Device::GPU>(compact_matrices.rows(), 3),
        DenseStorage<Scalar, Device::GPU>(compact_matrices.rows(), 9)
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
    const DenseStorage<Scalar, Device::GPU>& compact_matrices,
    DenseStorage<Scalar, Device::GPU>& eigenvalues,
    DenseStorage<Scalar, Device::GPU>& eigenvectors,
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
    const DenseStorage<Scalar, Device::GPU>& compact_matrices,
    DenseStorage<Scalar, Device::GPU>& eigenvalues,
    DenseStorage<Scalar, Device::GPU>& eigenvectors,
    SymmetricEigh3x3Workspace& workspace,
    cudaStream_t stream)
{
    validateShapes(
        "symmetricEigh3x3BatchedAsync", compact_matrices, &eigenvalues, &eigenvectors);
    small_matrix_detail::launchSymmetricEigh3x3(
        compact_matrices, eigenvalues, eigenvectors, workspace, stream);
}

template <typename Scalar>
void symmetricEigh3x3Batched(
    ConstMatrixView<Scalar, Device::GPU> compact_matrices,
    MatrixView<Scalar, Device::GPU> eigenvalues,
    MatrixView<Scalar, Device::GPU> eigenvectors,
    SymmetricEigh3x3Workspace& workspace,
    cudaStream_t stream)
{
    validateViewShapes("symmetricEigh3x3Batched", compact_matrices, eigenvalues, eigenvectors);
    small_matrix_detail::launchSymmetricEigh3x3(
        compact_matrices, eigenvalues, eigenvectors, workspace, stream);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    workspace.checkStatus("symmetricEigh3x3Batched");
}

template <typename Scalar>
void symmetricEigh3x3BatchedAsync(
    ConstMatrixView<Scalar, Device::GPU> compact_matrices,
    MatrixView<Scalar, Device::GPU> eigenvalues,
    MatrixView<Scalar, Device::GPU> eigenvectors,
    SymmetricEigh3x3Workspace& workspace,
    cudaStream_t stream)
{
    validateViewShapes("symmetricEigh3x3BatchedAsync", compact_matrices, eigenvalues, eigenvectors);
    small_matrix_detail::launchSymmetricEigh3x3(
        compact_matrices, eigenvalues, eigenvectors, workspace, stream);
}

#define PLAMATRIX_INSTANTIATE_SMALL_MATRIX(Scalar)                                      \
    template SymmetricEigh3x3Result<Scalar, Device::GPU> symmetricEigh3x3Batched(       \
        const DenseStorage<Scalar, Device::GPU>&);                                       \
    template void symmetricEigh3x3Batched(                                              \
        const DenseStorage<Scalar, Device::GPU>&,                                        \
        DenseStorage<Scalar, Device::GPU>&, DenseStorage<Scalar, Device::GPU>&,            \
        SymmetricEigh3x3Workspace&, cudaStream_t);                                      \
    template void symmetricEigh3x3BatchedAsync(                                         \
        const DenseStorage<Scalar, Device::GPU>&,                                        \
        DenseStorage<Scalar, Device::GPU>&, DenseStorage<Scalar, Device::GPU>&,            \
        SymmetricEigh3x3Workspace&, cudaStream_t);                                      \
    template void symmetricEigh3x3Batched(                                              \
        ConstMatrixView<Scalar, Device::GPU>, MatrixView<Scalar, Device::GPU>,           \
        MatrixView<Scalar, Device::GPU>, SymmetricEigh3x3Workspace&, cudaStream_t);     \
    template void symmetricEigh3x3BatchedAsync(                                         \
        ConstMatrixView<Scalar, Device::GPU>, MatrixView<Scalar, Device::GPU>,           \
        MatrixView<Scalar, Device::GPU>, SymmetricEigh3x3Workspace&, cudaStream_t)

#ifdef PLAMATRIX_USE_FLOAT
PLAMATRIX_INSTANTIATE_SMALL_MATRIX(float);
#endif
#ifdef PLAMATRIX_USE_DOUBLE
PLAMATRIX_INSTANTIATE_SMALL_MATRIX(double);
#endif

#undef PLAMATRIX_INSTANTIATE_SMALL_MATRIX

} // namespace plamatrix::internal
