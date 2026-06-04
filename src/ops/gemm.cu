#include <sstream>
#include <stdexcept>
#include <limits>
#include <type_traits>

#include <cublas_v2.h>

#include "plamatrix/ops/gemm.h"

namespace plamatrix
{

namespace
{

/// Lazily initialized cuBLAS handle — created on first call, reused thereafter.
cublasHandle_t getCublasHandle()
{
    static thread_local cublasHandle_t handle = nullptr;
    if (handle == nullptr)
    {
        PLAMATRIX_CHECK_CUBLAS(cublasCreate(&handle));
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

} // anonymous namespace

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> gemm(const DenseMatrix<Scalar, Device::GPU>& A,
                                       const DenseMatrix<Scalar, Device::GPU>& B,
                                       cudaStream_t stream)
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

    int m_int = checkedCublasInt(m, "GEMM m");
    int n_int = checkedCublasInt(n, "GEMM n");
    int k_int = checkedCublasInt(k, "GEMM k");
    int lda = m_int;
    int ldb = k_int;
    int ldc = m_int;

    DenseMatrix<Scalar, Device::GPU> C(m, n);

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

    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    return C;
}

// Explicit template instantiations
template DenseMatrix<float, Device::GPU> gemm(const DenseMatrix<float, Device::GPU>&,
                                                const DenseMatrix<float, Device::GPU>&,
                                                cudaStream_t);
template DenseMatrix<double, Device::GPU> gemm(const DenseMatrix<double, Device::GPU>&,
                                                 const DenseMatrix<double, Device::GPU>&,
                                                 cudaStream_t);

} // namespace plamatrix
