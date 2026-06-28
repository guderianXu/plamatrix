#include <sstream>
#include <stdexcept>

#include "plamatrix/core/gpu_runtime.h"
#include "plamatrix/core/metal_context.h"
#include "plamatrix/ops/gemm.h"

namespace plamatrix
{

namespace
{

template <typename Scalar>
void checkGemmDimensions(const DenseMatrix<Scalar, Device::GPU>& A,
                         const DenseMatrix<Scalar, Device::GPU>& B)
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
void checkGemmOutputDimensions(const DenseMatrix<Scalar, Device::GPU>& C,
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

} // namespace

#ifdef PLAMATRIX_USE_FLOAT
template <>
void gemmAsync(const DenseMatrix<float, Device::GPU>& A,
               const DenseMatrix<float, Device::GPU>& B,
               DenseMatrix<float, Device::GPU>& C,
               cudaStream_t)
{
    Index m = A.rows();
    Index k = A.cols();
    Index n = B.cols();

    checkGemmDimensions(A, B);
    checkGemmOutputDimensions(C, m, n);
    detail::metalGemmFloat(A.data(), B.data(), C.data(), m, n, k);
}
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template <>
void gemmAsync(const DenseMatrix<double, Device::GPU>& A,
               const DenseMatrix<double, Device::GPU>& B,
               DenseMatrix<double, Device::GPU>& C,
               cudaStream_t)
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
        detail::gpuMemset(C.data(), 0, static_cast<std::size_t>(C.size()) * sizeof(double));
        return;
    }
    detail::metalGemmDoubleFallback(A.data(), B.data(), C.data(), m, n, k);
}
#endif

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> gemmAsync(const DenseMatrix<Scalar, Device::GPU>& A,
                                           const DenseMatrix<Scalar, Device::GPU>& B,
                                           cudaStream_t stream)
{
    checkGemmDimensions(A, B);
    DenseMatrix<Scalar, Device::GPU> C(A.rows(), B.cols());
    gemmAsync(A, B, C, stream);
    return C;
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> gemm(const DenseMatrix<Scalar, Device::GPU>& A,
                                      const DenseMatrix<Scalar, Device::GPU>& B,
                                      cudaStream_t stream)
{
    return gemmAsync(A, B, stream);
}

template <typename Scalar>
void gemm(const DenseMatrix<Scalar, Device::GPU>& A,
          const DenseMatrix<Scalar, Device::GPU>& B,
          DenseMatrix<Scalar, Device::GPU>& C,
          cudaStream_t stream)
{
    gemmAsync(A, B, C, stream);
}

#ifdef PLAMATRIX_USE_FLOAT
template DenseMatrix<float, Device::GPU> gemmAsync(const DenseMatrix<float, Device::GPU>&,
                                                   const DenseMatrix<float, Device::GPU>&,
                                                   cudaStream_t);

template DenseMatrix<float, Device::GPU> gemm(const DenseMatrix<float, Device::GPU>&,
                                              const DenseMatrix<float, Device::GPU>&,
                                              cudaStream_t);

template void gemm(const DenseMatrix<float, Device::GPU>&,
                   const DenseMatrix<float, Device::GPU>&,
                   DenseMatrix<float, Device::GPU>&,
                   cudaStream_t);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template DenseMatrix<double, Device::GPU> gemmAsync(const DenseMatrix<double, Device::GPU>&,
                                                    const DenseMatrix<double, Device::GPU>&,
                                                    cudaStream_t);

template DenseMatrix<double, Device::GPU> gemm(const DenseMatrix<double, Device::GPU>&,
                                               const DenseMatrix<double, Device::GPU>&,
                                               cudaStream_t);

template void gemm(const DenseMatrix<double, Device::GPU>&,
                   const DenseMatrix<double, Device::GPU>&,
                   DenseMatrix<double, Device::GPU>&,
                   cudaStream_t);
#endif

} // namespace plamatrix
