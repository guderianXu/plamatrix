#include <sstream>
#include <stdexcept>
#include <limits>
#include <type_traits>

#include <cublas_v2.h>

#include "plamatrix/internal/ops/gemm.h"

namespace plamatrix::internal
{

namespace
{

/// Lazily initialized cuBLAS handle — created on first call, reused thereafter.
cublasHandle_t getCublasHandle()
{
    static thread_local cublasHandle_t handle = nullptr;
    static thread_local int handle_device = -1;
    int current_device = 0;
    PLAMATRIX_CHECK_CUDA(cudaGetDevice(&current_device));
    if (handle != nullptr && handle_device != current_device)
    {
        PLAMATRIX_CHECK_CUBLAS(cublasDestroy(handle));
        handle = nullptr;
    }
    if (handle == nullptr)
    {
        PLAMATRIX_CHECK_CUBLAS(cublasCreate(&handle));
        handle_device = current_device;
    }
    return handle;
}

int checkedCublasInt(Index value, const char* name)
{
    if (value < 0 || value > static_cast<Index>(std::numeric_limits<int>::max()))
    {
        std::ostringstream oss;
        oss << name << " is outside cuBLAS int range: " << value;
        throw std::runtime_error(oss.str());
    }
    return static_cast<int>(value);
}

template <typename Scalar>
void checkGemmDimensions(const DenseStorage<Scalar, Device::GPU>& A,
                         const DenseStorage<Scalar, Device::GPU>& B)
{
    Index m = A.rows();
    Index k = A.cols();
    Index n = B.cols();

    if (k != B.rows())
    {
        std::ostringstream oss;
        oss << "GEMM dimension mismatch: A is " << m << "x" << k
            << ", B is " << B.rows() << "x" << n;
        throw std::runtime_error(oss.str());
    }
}

template <typename Scalar>
void checkGemmOutputDimensions(const DenseStorage<Scalar, Device::GPU>& C,
                               Index rows,
                               Index cols)
{
    if (C.rows() != rows || C.cols() != cols)
    {
        std::ostringstream oss;
        oss << "GEMM output dimension mismatch: output is " << C.rows() << "x" << C.cols()
            << ", expected " << rows << "x" << cols;
        throw std::runtime_error(oss.str());
    }
}

} // anonymous namespace

template <typename Scalar>
void gemmAsync(const DenseStorage<Scalar, Device::GPU>& A,
               const DenseStorage<Scalar, Device::GPU>& B,
               DenseStorage<Scalar, Device::GPU>& C,
               cudaStream_t stream)
{
    Index m = A.rows();
    Index k = A.cols();
    Index n = B.cols();

    checkGemmDimensions(A, B);
    checkGemmOutputDimensions(C, m, n);

    if (m == 0 || n == 0)
    {
        return;
    }

    if (k == 0)
    {
        std::size_t bytes = detail::checkedAllocationBytes<Scalar>(static_cast<std::size_t>(C.size()));
        PLAMATRIX_CHECK_CUDA(cudaMemsetAsync(C.data(), 0, bytes, stream));
        return;
    }

    int m_int = checkedCublasInt(m, "GEMM m");
    int n_int = checkedCublasInt(n, "GEMM n");
    int k_int = checkedCublasInt(k, "GEMM k");
    int lda = m_int;
    int ldb = k_int;
    int ldc = m_int;

    cublasHandle_t handle = getCublasHandle();
    PLAMATRIX_CHECK_CUBLAS(cublasSetStream(handle, stream));

    Scalar alpha = static_cast<Scalar>(1.0);
    Scalar beta  = static_cast<Scalar>(0.0);

    if constexpr (std::is_same_v<Scalar, float>)
    {
        PLAMATRIX_CHECK_CUBLAS(
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        m_int, n_int, k_int,
                        &alpha, A.data(), lda, B.data(), ldb, &beta, C.data(), ldc));
    }
    else
    {
        PLAMATRIX_CHECK_CUBLAS(
            cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        m_int, n_int, k_int,
                        &alpha, A.data(), lda, B.data(), ldb, &beta, C.data(), ldc));
    }
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> gemmAsync(const DenseStorage<Scalar, Device::GPU>& A,
                                           const DenseStorage<Scalar, Device::GPU>& B,
                                           cudaStream_t stream)
{
    checkGemmDimensions(A, B);
    auto C = DenseStorage<Scalar, Device::GPU>::uninitializedAsync(A.rows(), B.cols(), stream);
    gemmAsync(A, B, C, stream);
    return C;
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> gemm(const DenseStorage<Scalar, Device::GPU>& A,
                                       const DenseStorage<Scalar, Device::GPU>& B,
                                       cudaStream_t stream)
{
    auto C = gemmAsync(A, B, stream);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    return C;
}

template <typename Scalar>
void gemm(const DenseStorage<Scalar, Device::GPU>& A,
          const DenseStorage<Scalar, Device::GPU>& B,
          DenseStorage<Scalar, Device::GPU>& C,
          cudaStream_t stream)
{
    gemmAsync(A, B, C, stream);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
}

// Explicit template instantiations
#ifdef PLAMATRIX_USE_FLOAT
template DenseStorage<float, Device::GPU> gemmAsync(const DenseStorage<float, Device::GPU>&,
                                                   const DenseStorage<float, Device::GPU>&,
                                                   cudaStream_t);

template void gemmAsync(const DenseStorage<float, Device::GPU>&,
                        const DenseStorage<float, Device::GPU>&,
                        DenseStorage<float, Device::GPU>&,
                        cudaStream_t);

template DenseStorage<float, Device::GPU> gemm(const DenseStorage<float, Device::GPU>&,
                                                const DenseStorage<float, Device::GPU>&,
                                                cudaStream_t);

template void gemm(const DenseStorage<float, Device::GPU>&,
                   const DenseStorage<float, Device::GPU>&,
                   DenseStorage<float, Device::GPU>&,
                   cudaStream_t);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template DenseStorage<double, Device::GPU> gemmAsync(const DenseStorage<double, Device::GPU>&,
                                                    const DenseStorage<double, Device::GPU>&,
                                                    cudaStream_t);

template void gemmAsync(const DenseStorage<double, Device::GPU>&,
                        const DenseStorage<double, Device::GPU>&,
                        DenseStorage<double, Device::GPU>&,
                        cudaStream_t);

template DenseStorage<double, Device::GPU> gemm(const DenseStorage<double, Device::GPU>&,
                                                 const DenseStorage<double, Device::GPU>&,
                                                 cudaStream_t);

template void gemm(const DenseStorage<double, Device::GPU>&,
                   const DenseStorage<double, Device::GPU>&,
                   DenseStorage<double, Device::GPU>&,
                   cudaStream_t);
#endif

} // namespace plamatrix::internal
