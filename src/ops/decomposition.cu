#include <sstream>
#include <stdexcept>
#include <type_traits>

#include <cublas_v2.h>
#include <cusolverDn.h>

#include "plamatrix/core/error_check.h"
#include "plamatrix/ops/decomposition.h"

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

template <typename Scalar>
std::tuple<DenseMatrix<Scalar, Device::GPU>, DenseMatrix<Scalar, Device::GPU>, DenseMatrix<Scalar, Device::GPU>>
svdGpuImpl(const DenseMatrix<Scalar, Device::GPU>& A)
{
    Index m = A.rows();
    Index n = A.cols();

    if (m == 0 || n == 0)
    {
        throw std::runtime_error("SVD: input matrix has zero dimensions");
    }

    cusolverDnHandle_t handle = getCusolverHandle();

    signed char jobu  = 'A';  // All m columns of U
    signed char jobvt = 'A';  // All n rows of Vt

    int m_int    = static_cast<int>(m);
    int n_int    = static_cast<int>(n);
    int lda      = m_int;
    int ldu      = m_int;
    int ldvt     = n_int;
    int min_mn   = (m_int < n_int) ? m_int : n_int;  // std::min with plain ints

    // Copy A to device (gesvd overwrites A)
    DenseMatrix<Scalar, Device::GPU> A_copy(m, n);
    PLAMATRIX_CHECK_CUDA(
        cudaMemcpy(A_copy.data(), A.data(),
                   static_cast<std::size_t>(m * n) * sizeof(Scalar),
                   cudaMemcpyDeviceToDevice));

    // Output matrices on GPU
    DenseMatrix<Scalar, Device::GPU> U_gpu(m, m);
    DenseMatrix<Scalar, Device::GPU> S_gpu(min_mn, 1);
    DenseMatrix<Scalar, Device::GPU> Vt_gpu(n, n);

    // Query workspace size
    int lwork = 0;
    if constexpr (std::is_same_v<Scalar, float>)
    {
        PLAMATRIX_CHECK_CUSOLVER(
            cusolverDnSgesvd_bufferSize(handle, m_int, n_int, &lwork));
    }
    else
    {
        PLAMATRIX_CHECK_CUSOLVER(
            cusolverDnDgesvd_bufferSize(handle, m_int, n_int, &lwork));
    }

    // Allocate workspace, rwork, and dev_info on GPU
    Scalar* d_work = nullptr;
    Scalar* d_rwork = nullptr;
    int* d_dev_info = nullptr;

    int rwork_size = (min_mn > 1) ? (min_mn - 1) : 1;

    PLAMATRIX_CHECK_CUDA(cudaMalloc(&d_work, static_cast<std::size_t>(lwork) * sizeof(Scalar)));
    PLAMATRIX_CHECK_CUDA(cudaMalloc(&d_rwork, static_cast<std::size_t>(rwork_size) * sizeof(Scalar)));
    PLAMATRIX_CHECK_CUDA(cudaMalloc(&d_dev_info, sizeof(int)));

    // Call gesvd
    if constexpr (std::is_same_v<Scalar, float>)
    {
        PLAMATRIX_CHECK_CUSOLVER(
            cusolverDnSgesvd(handle, jobu, jobvt,
                             m_int, n_int,
                             A_copy.data(), lda,
                             S_gpu.data(),
                             U_gpu.data(), ldu,
                             Vt_gpu.data(), ldvt,
                             d_work, lwork,
                             d_rwork,
                             d_dev_info));
    }
    else
    {
        PLAMATRIX_CHECK_CUSOLVER(
            cusolverDnDgesvd(handle, jobu, jobvt,
                             m_int, n_int,
                             A_copy.data(), lda,
                             S_gpu.data(),
                             U_gpu.data(), ldu,
                             Vt_gpu.data(), ldvt,
                             d_work, lwork,
                             d_rwork,
                             d_dev_info));
    }

    // Check dev_info
    int host_dev_info = 0;
    PLAMATRIX_CHECK_CUDA(cudaMemcpy(&host_dev_info, d_dev_info, sizeof(int), cudaMemcpyDeviceToHost));

    // Free workspace, rwork, and dev_info
    PLAMATRIX_CHECK_CUDA(cudaFree(d_work));
    PLAMATRIX_CHECK_CUDA(cudaFree(d_rwork));
    PLAMATRIX_CHECK_CUDA(cudaFree(d_dev_info));

    if (host_dev_info < 0)
    {
        std::ostringstream oss;
        oss << "SVD: invalid argument at parameter " << -host_dev_info;
        throw std::runtime_error(oss.str());
    }
    if (host_dev_info > 0)
    {
        std::ostringstream oss;
        oss << "SVD: gesvd did not converge, " << host_dev_info << " superdiagonals did not converge";
        throw std::runtime_error(oss.str());
    }

    return {std::move(U_gpu), std::move(S_gpu), std::move(Vt_gpu)};
}

} // anonymous namespace

template <typename Scalar>
std::tuple<DenseMatrix<Scalar, Device::GPU>, DenseMatrix<Scalar, Device::GPU>, DenseMatrix<Scalar, Device::GPU>>
svd(const DenseMatrix<Scalar, Device::GPU>& A)
{
    return svdGpuImpl(A);
}

// Explicit template instantiations
template std::tuple<DenseMatrix<float, Device::GPU>, DenseMatrix<float, Device::GPU>, DenseMatrix<float, Device::GPU>>
svd(const DenseMatrix<float, Device::GPU>&);
template std::tuple<DenseMatrix<double, Device::GPU>, DenseMatrix<double, Device::GPU>, DenseMatrix<double, Device::GPU>>
svd(const DenseMatrix<double, Device::GPU>&);

} // namespace plamatrix
