#include <sstream>
#include <stdexcept>
#include <type_traits>

#include <cusolverDn.h>

#include "plamatrix/core/error_check.h"
#include "plamatrix/ops/solver.h"

namespace plamatrix
{

namespace
{

/// Lazily initialized cuSOLVER handle
cusolverDnHandle_t getCusolverHandle()
{
    static cusolverDnHandle_t handle = nullptr;
    if (handle == nullptr)
    {
        PLAMATRIX_CHECK_CUSOLVER(cusolverDnCreate(&handle));
    }
    return handle;
}

/// GPU linear solve via cuSOLVER getrf/getrs.
/// @tparam Scalar  Element type (float or double)
/// @param A  Square coefficient matrix on GPU (n x n)
/// @param B  Right-hand side matrix on GPU (n x nrhs)
/// @return  Solution matrix X on GPU (n x nrhs)
template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> solveGpuImpl(const DenseMatrix<Scalar, Device::GPU>& A,
                                                const DenseMatrix<Scalar, Device::GPU>& B)
{
    Index n = A.rows();
    Index nrhs = B.cols();

    if (n == 0 || nrhs == 0)
    {
        throw std::runtime_error("Solve: input matrix has zero dimensions");
    }
    if (A.cols() != n)
    {
        throw std::runtime_error("Solve: coefficient matrix A must be square");
    }
    if (B.rows() != n)
    {
        std::ostringstream oss;
        oss << "Solve: dimension mismatch. A is " << n << "x" << n
            << ", B is " << B.rows() << "x" << nrhs;
        throw std::runtime_error(oss.str());
    }

    cusolverDnHandle_t handle = getCusolverHandle();

    int n_int     = static_cast<int>(n);
    int nrhs_int  = static_cast<int>(nrhs);
    int lda       = n_int;
    int ldb       = n_int;

    // Copy A to device work matrix (getrf overwrites A)
    DenseMatrix<Scalar, Device::GPU> A_work(n, n);
    PLAMATRIX_CHECK_CUDA(
        cudaMemcpy(A_work.data(), A.data(),
                   static_cast<std::size_t>(n * n) * sizeof(Scalar),
                   cudaMemcpyDeviceToDevice));

    // Copy B to device work matrix (getrs overwrites B)
    DenseMatrix<Scalar, Device::GPU> B_work(n, nrhs);
    PLAMATRIX_CHECK_CUDA(
        cudaMemcpy(B_work.data(), B.data(),
                   static_cast<std::size_t>(n * nrhs) * sizeof(Scalar),
                   cudaMemcpyDeviceToDevice));

    // Allocate pivot array on GPU
    int* d_pivot = nullptr;
    PLAMATRIX_CHECK_CUDA(cudaMalloc(&d_pivot, static_cast<std::size_t>(n) * sizeof(int)));

    // Allocate dev_info on GPU
    int* d_dev_info = nullptr;
    PLAMATRIX_CHECK_CUDA(cudaMalloc(&d_dev_info, sizeof(int)));

    // Query workspace size for getrf
    int lwork = 0;
    if constexpr (std::is_same_v<Scalar, float>)
    {
        PLAMATRIX_CHECK_CUSOLVER(
            cusolverDnSgetrf_bufferSize(handle, n_int, n_int, A_work.data(), lda, &lwork));
    }
    else
    {
        PLAMATRIX_CHECK_CUSOLVER(
            cusolverDnDgetrf_bufferSize(handle, n_int, n_int, A_work.data(), lda, &lwork));
    }

    // Allocate workspace
    Scalar* d_work = nullptr;
    PLAMATRIX_CHECK_CUDA(cudaMalloc(&d_work, static_cast<std::size_t>(lwork) * sizeof(Scalar)));

    // Call getrf (LU factorization)
    if constexpr (std::is_same_v<Scalar, float>)
    {
        PLAMATRIX_CHECK_CUSOLVER(
            cusolverDnSgetrf(handle, n_int, n_int, A_work.data(), lda, d_work, d_pivot, d_dev_info));
    }
    else
    {
        PLAMATRIX_CHECK_CUSOLVER(
            cusolverDnDgetrf(handle, n_int, n_int, A_work.data(), lda, d_work, d_pivot, d_dev_info));
    }

    // Check dev_info from getrf
    int host_dev_info = 0;
    PLAMATRIX_CHECK_CUDA(cudaMemcpy(&host_dev_info, d_dev_info, sizeof(int), cudaMemcpyDeviceToHost));

    if (host_dev_info < 0)
    {
        PLAMATRIX_CHECK_CUDA(cudaFree(d_work));
        PLAMATRIX_CHECK_CUDA(cudaFree(d_pivot));
        PLAMATRIX_CHECK_CUDA(cudaFree(d_dev_info));
        std::ostringstream oss;
        oss << "Solve: invalid argument at getrf parameter " << -host_dev_info;
        throw std::runtime_error(oss.str());
    }
    if (host_dev_info > 0)
    {
        PLAMATRIX_CHECK_CUDA(cudaFree(d_work));
        PLAMATRIX_CHECK_CUDA(cudaFree(d_pivot));
        PLAMATRIX_CHECK_CUDA(cudaFree(d_dev_info));
        std::ostringstream oss;
        oss << "Solve: matrix is singular (U[" << (host_dev_info - 1) << "," << (host_dev_info - 1) << "] is zero)";
        throw std::runtime_error(oss.str());
    }

    // Call getrs (solve)
    if constexpr (std::is_same_v<Scalar, float>)
    {
        PLAMATRIX_CHECK_CUSOLVER(
            cusolverDnSgetrs(handle, CUBLAS_OP_N, n_int, nrhs_int,
                             A_work.data(), lda, d_pivot,
                             B_work.data(), ldb, d_dev_info));
    }
    else
    {
        PLAMATRIX_CHECK_CUSOLVER(
            cusolverDnDgetrs(handle, CUBLAS_OP_N, n_int, nrhs_int,
                             A_work.data(), lda, d_pivot,
                             B_work.data(), ldb, d_dev_info));
    }

    // Check dev_info from getrs
    PLAMATRIX_CHECK_CUDA(cudaMemcpy(&host_dev_info, d_dev_info, sizeof(int), cudaMemcpyDeviceToHost));

    // Free resources
    PLAMATRIX_CHECK_CUDA(cudaFree(d_work));
    PLAMATRIX_CHECK_CUDA(cudaFree(d_pivot));
    PLAMATRIX_CHECK_CUDA(cudaFree(d_dev_info));

    if (host_dev_info < 0)
    {
        std::ostringstream oss;
        oss << "Solve: invalid argument at getrs parameter " << -host_dev_info;
        throw std::runtime_error(oss.str());
    }

    return B_work;
}

} // anonymous namespace

// Explicit specializations for GPU
template <>
DenseMatrix<float, Device::GPU> solve<float, Device::GPU>(const DenseMatrix<float, Device::GPU>& A,
                                                            const DenseMatrix<float, Device::GPU>& B)
{
    return solveGpuImpl(A, B);
}

template <>
DenseMatrix<double, Device::GPU> solve<double, Device::GPU>(const DenseMatrix<double, Device::GPU>& A,
                                                              const DenseMatrix<double, Device::GPU>& B)
{
    return solveGpuImpl(A, B);
}

} // namespace plamatrix
