#include <sstream>
#include <stdexcept>
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
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr)
    {
        PLAMATRIX_CHECK_CUBLAS(cublasCreate(&handle));
    }
    return handle;
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

    DenseMatrix<Scalar, Device::GPU> C(m, n);

    cublasHandle_t handle = getCublasHandle();
    if (stream != nullptr)
    {
        PLAMATRIX_CHECK_CUBLAS(cublasSetStream(handle, stream));
    }

    Scalar alpha = static_cast<Scalar>(1.0);
    Scalar beta  = static_cast<Scalar>(0.0);

    int lda = static_cast<int>(m);
    int ldb = static_cast<int>(k);
    int ldc = static_cast<int>(m);

    if constexpr (std::is_same_v<Scalar, float>)
    {
        PLAMATRIX_CHECK_CUBLAS(
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                        &alpha, A.data(), lda, B.data(), ldb, &beta, C.data(), ldc));
    }
    else
    {
        PLAMATRIX_CHECK_CUBLAS(
            cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                        &alpha, A.data(), lda, B.data(), ldb, &beta, C.data(), ldc));
    }

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
