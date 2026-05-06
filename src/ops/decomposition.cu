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

namespace
{

template <typename Scalar>
std::tuple<DenseMatrix<Scalar, Device::GPU>, DenseMatrix<Scalar, Device::GPU>>
qrGpuImpl(const DenseMatrix<Scalar, Device::GPU>& A)
{
    Index m = A.rows();
    Index n = A.cols();

    if (m == 0 || n == 0)
    {
        throw std::runtime_error("QR: input matrix has zero dimensions");
    }

    cusolverDnHandle_t handle = getCusolverHandle();

    int m_int = static_cast<int>(m);
    int n_int = static_cast<int>(n);
    int lda   = m_int;
    int min_mn = (m_int < n_int) ? m_int : n_int;

    // Copy A to device (geqrf overwrites A)
    DenseMatrix<Scalar, Device::GPU> R(m, n);
    PLAMATRIX_CHECK_CUDA(
        cudaMemcpy(R.data(), A.data(),
                   static_cast<std::size_t>(m * n) * sizeof(Scalar),
                   cudaMemcpyDeviceToDevice));

    // Allocate tau on GPU
    Scalar* d_tau = nullptr;
    PLAMATRIX_CHECK_CUDA(cudaMalloc(&d_tau, static_cast<std::size_t>(min_mn) * sizeof(Scalar)));

    // Query workspace size for geqrf
    int lwork = 0;
    if constexpr (std::is_same_v<Scalar, float>)
    {
        PLAMATRIX_CHECK_CUSOLVER(
            cusolverDnSgeqrf_bufferSize(handle, m_int, n_int, R.data(), lda, &lwork));
    }
    else
    {
        PLAMATRIX_CHECK_CUSOLVER(
            cusolverDnDgeqrf_bufferSize(handle, m_int, n_int, R.data(), lda, &lwork));
    }

    // Allocate workspace and dev_info
    Scalar* d_work = nullptr;
    int* d_dev_info = nullptr;
    PLAMATRIX_CHECK_CUDA(cudaMalloc(&d_work, static_cast<std::size_t>(lwork) * sizeof(Scalar)));
    PLAMATRIX_CHECK_CUDA(cudaMalloc(&d_dev_info, sizeof(int)));

    // Call geqrf
    if constexpr (std::is_same_v<Scalar, float>)
    {
        PLAMATRIX_CHECK_CUSOLVER(
            cusolverDnSgeqrf(handle, m_int, n_int, R.data(), lda, d_tau,
                             d_work, lwork, d_dev_info));
    }
    else
    {
        PLAMATRIX_CHECK_CUSOLVER(
            cusolverDnDgeqrf(handle, m_int, n_int, R.data(), lda, d_tau,
                             d_work, lwork, d_dev_info));
    }

    // Check dev_info
    int host_dev_info = 0;
    PLAMATRIX_CHECK_CUDA(cudaMemcpy(&host_dev_info, d_dev_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (host_dev_info < 0)
    {
        PLAMATRIX_CHECK_CUDA(cudaFree(d_work));
        PLAMATRIX_CHECK_CUDA(cudaFree(d_dev_info));
        PLAMATRIX_CHECK_CUDA(cudaFree(d_tau));
        std::ostringstream oss;
        oss << "QR: invalid argument at parameter " << -host_dev_info;
        throw std::runtime_error(oss.str());
    }

    // Extract Q: use orgqr
    // Need a copy of R with the upper triangular part for Q generation
    // orgqr takes: (m, n, k, A, lda, tau, work, lwork, devInfo)
    // where k = min(m, n), and A contains the QR factors on input
    // On output, A contains Q (first k columns are orthonormal)

    // Allocate Q matrix: copy of R, then orgqr fills it with Q
    DenseMatrix<Scalar, Device::GPU> Q_full(m, m);
    // Copy the first n columns from R to Q_full (orgqr works on the input matrix)
    // For m > n, we need to pad Q to m x m dimensions
    // Easy approach: create m x m matrix, set to identity, copy R's first n cols
    // Actually, orgqr with (m, m, m, ...) should work fine for generating full m×m Q

    // Zero out Q_full first (set to identity)
    PLAMATRIX_CHECK_CUDA(
        cudaMemset(Q_full.data(), 0, static_cast<std::size_t>(m * m) * sizeof(Scalar)));
    // Set diagonal to 1 (CPU side for simplicity)
    {
        DenseMatrix<Scalar, Device::CPU> Q_cpu(m, m);
        for (Index i = 0; i < m; ++i)
        {
            Q_cpu(i, i) = Scalar(1);
        }
        PLAMATRIX_CHECK_CUDA(
            cudaMemcpy(Q_full.data(), Q_cpu.data(),
                       static_cast<std::size_t>(m * m) * sizeof(Scalar),
                       cudaMemcpyHostToDevice));
    }

    // Copy the upper triangular part and Householder vectors from R to Q_full's first n columns
    for (Index j = 0; j < n; ++j)
    {
        PLAMATRIX_CHECK_CUDA(
            cudaMemcpy(Q_full.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(m),
                       R.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(m),
                       static_cast<std::size_t>(m) * sizeof(Scalar),
                       cudaMemcpyDeviceToDevice));
    }

    // Query workspace size for orgqr
    int lwork_orgqr = 0;
    if constexpr (std::is_same_v<Scalar, float>)
    {
        PLAMATRIX_CHECK_CUSOLVER(
            cusolverDnSorgqr_bufferSize(handle, m_int, m_int, min_mn,
                                        Q_full.data(), m_int, d_tau, &lwork_orgqr));
    }
    else
    {
        PLAMATRIX_CHECK_CUSOLVER(
            cusolverDnDorgqr_bufferSize(handle, m_int, m_int, min_mn,
                                        Q_full.data(), m_int, d_tau, &lwork_orgqr));
    }

    // Allocate orgqr workspace
    Scalar* d_work_orgqr = nullptr;
    PLAMATRIX_CHECK_CUDA(cudaMalloc(&d_work_orgqr, static_cast<std::size_t>(lwork_orgqr) * sizeof(Scalar)));

    // Call orgqr
    if constexpr (std::is_same_v<Scalar, float>)
    {
        PLAMATRIX_CHECK_CUSOLVER(
            cusolverDnSorgqr(handle, m_int, m_int, min_mn,
                             Q_full.data(), m_int, d_tau,
                             d_work_orgqr, lwork_orgqr, d_dev_info));
    }
    else
    {
        PLAMATRIX_CHECK_CUSOLVER(
            cusolverDnDorgqr(handle, m_int, m_int, min_mn,
                             Q_full.data(), m_int, d_tau,
                             d_work_orgqr, lwork_orgqr, d_dev_info));
    }

    // Check orgqr dev_info
    PLAMATRIX_CHECK_CUDA(cudaMemcpy(&host_dev_info, d_dev_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (host_dev_info < 0)
    {
        PLAMATRIX_CHECK_CUDA(cudaFree(d_work));
        PLAMATRIX_CHECK_CUDA(cudaFree(d_dev_info));
        PLAMATRIX_CHECK_CUDA(cudaFree(d_tau));
        PLAMATRIX_CHECK_CUDA(cudaFree(d_work_orgqr));
        std::ostringstream oss;
        oss << "QR orgqr: invalid argument at parameter " << -host_dev_info;
        throw std::runtime_error(oss.str());
    }

    // Zero out sub-diagonal elements of R to get clean upper triangular
    // We do this on CPU for simplicity since R stays on GPU
    // Start with a zero matrix and copy upper triangular part
    DenseMatrix<Scalar, Device::GPU> R_clean(m, n);
    PLAMATRIX_CHECK_CUDA(
        cudaMemset(R_clean.data(), 0, static_cast<std::size_t>(m * n) * sizeof(Scalar)));

    // Copy upper triangular part (column by column)
    for (Index j = 0; j < n; ++j)
    {
        Index copy_len = (j + 1 < m) ? (j + 1) : m;
        PLAMATRIX_CHECK_CUDA(
            cudaMemcpy(R_clean.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(m),
                       R.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(m),
                       static_cast<std::size_t>(copy_len) * sizeof(Scalar),
                       cudaMemcpyDeviceToDevice));
    }

    // Free resources
    PLAMATRIX_CHECK_CUDA(cudaFree(d_work));
    PLAMATRIX_CHECK_CUDA(cudaFree(d_dev_info));
    PLAMATRIX_CHECK_CUDA(cudaFree(d_tau));
    PLAMATRIX_CHECK_CUDA(cudaFree(d_work_orgqr));

    return {std::move(Q_full), std::move(R_clean)};
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> eighGpuImpl(const DenseMatrix<Scalar, Device::GPU>& A)
{
    Index n = A.rows();

    if (n == 0)
    {
        throw std::runtime_error("Eigh: input matrix has zero dimensions");
    }

    if (A.cols() != n)
    {
        throw std::runtime_error("Eigh: input matrix must be square");
    }

    cusolverDnHandle_t handle = getCusolverHandle();

    int n_int = static_cast<int>(n);
    int lda   = n_int;

    // We only need eigenvalues, not eigenvectors
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    // Allocate eigenvalue array on GPU
    DenseMatrix<Scalar, Device::GPU> eigvals_gpu(n, 1);

    // Copy A to GPU work matrix (syevd overwrites A)
    DenseMatrix<Scalar, Device::GPU> A_work(n, n);
    PLAMATRIX_CHECK_CUDA(
        cudaMemcpy(A_work.data(), A.data(),
                   static_cast<std::size_t>(n * n) * sizeof(Scalar),
                   cudaMemcpyDeviceToDevice));

    // Query workspace size
    int lwork = 0;
    if constexpr (std::is_same_v<Scalar, float>)
    {
        PLAMATRIX_CHECK_CUSOLVER(
            cusolverDnSsyevd_bufferSize(handle, jobz, uplo, n_int,
                                        A_work.data(), lda,
                                        eigvals_gpu.data(), &lwork));
    }
    else
    {
        PLAMATRIX_CHECK_CUSOLVER(
            cusolverDnDsyevd_bufferSize(handle, jobz, uplo, n_int,
                                        A_work.data(), lda,
                                        eigvals_gpu.data(), &lwork));
    }

    // Allocate workspace and dev_info
    Scalar* d_work = nullptr;
    int* d_dev_info = nullptr;
    PLAMATRIX_CHECK_CUDA(cudaMalloc(&d_work, static_cast<std::size_t>(lwork) * sizeof(Scalar)));
    PLAMATRIX_CHECK_CUDA(cudaMalloc(&d_dev_info, sizeof(int)));

    // Call syevd
    if constexpr (std::is_same_v<Scalar, float>)
    {
        PLAMATRIX_CHECK_CUSOLVER(
            cusolverDnSsyevd(handle, jobz, uplo, n_int,
                             A_work.data(), lda,
                             eigvals_gpu.data(),
                             d_work, lwork, d_dev_info));
    }
    else
    {
        PLAMATRIX_CHECK_CUSOLVER(
            cusolverDnDsyevd(handle, jobz, uplo, n_int,
                             A_work.data(), lda,
                             eigvals_gpu.data(),
                             d_work, lwork, d_dev_info));
    }

    // Check dev_info
    int host_dev_info = 0;
    PLAMATRIX_CHECK_CUDA(cudaMemcpy(&host_dev_info, d_dev_info, sizeof(int), cudaMemcpyDeviceToHost));

    PLAMATRIX_CHECK_CUDA(cudaFree(d_work));
    PLAMATRIX_CHECK_CUDA(cudaFree(d_dev_info));

    if (host_dev_info < 0)
    {
        std::ostringstream oss;
        oss << "Eigh: invalid argument at parameter " << -host_dev_info;
        throw std::runtime_error(oss.str());
    }
    if (host_dev_info > 0)
    {
        std::ostringstream oss;
        oss << "Eigh: syevd did not converge, " << host_dev_info << " off-diagonals";
        throw std::runtime_error(oss.str());
    }

    // cuSOLVER returns eigenvalues in ascending order; sort to descending
    {
        DenseMatrix<Scalar, Device::CPU> eigvals_cpu(n, 1);
        PLAMATRIX_CHECK_CUDA(
            cudaMemcpy(eigvals_cpu.data(), eigvals_gpu.data(),
                       static_cast<std::size_t>(n) * sizeof(Scalar),
                       cudaMemcpyDeviceToHost));

        // Sort descending in-place
        for (Index i = 0; i < n - 1; ++i)
        {
            Index max_idx = i;
            for (Index j = i + 1; j < n; ++j)
            {
                if (eigvals_cpu(j, 0) > eigvals_cpu(max_idx, 0))
                {
                    max_idx = j;
                }
            }
            if (max_idx != i)
            {
                Scalar tmp = eigvals_cpu(i, 0);
                eigvals_cpu(i, 0) = eigvals_cpu(max_idx, 0);
                eigvals_cpu(max_idx, 0) = tmp;
            }
        }

        PLAMATRIX_CHECK_CUDA(
            cudaMemcpy(eigvals_gpu.data(), eigvals_cpu.data(),
                       static_cast<std::size_t>(n) * sizeof(Scalar),
                       cudaMemcpyHostToDevice));
    }

    return eigvals_gpu;
}

} // anonymous namespace

template <typename Scalar>
std::tuple<DenseMatrix<Scalar, Device::GPU>, DenseMatrix<Scalar, Device::GPU>>
qr(const DenseMatrix<Scalar, Device::GPU>& A)
{
    return qrGpuImpl(A);
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> eigh(const DenseMatrix<Scalar, Device::GPU>& A)
{
    return eighGpuImpl(A);
}

// Explicit template instantiations for qr
template std::tuple<DenseMatrix<float, Device::GPU>, DenseMatrix<float, Device::GPU>>
qr(const DenseMatrix<float, Device::GPU>&);
template std::tuple<DenseMatrix<double, Device::GPU>, DenseMatrix<double, Device::GPU>>
qr(const DenseMatrix<double, Device::GPU>&);

// Explicit template instantiations for eigh
template DenseMatrix<float, Device::GPU> eigh(const DenseMatrix<float, Device::GPU>&);
template DenseMatrix<double, Device::GPU> eigh(const DenseMatrix<double, Device::GPU>&);

} // namespace plamatrix
