#include <sstream>
#include <stdexcept>

#include "plamatrix/core/metal_context.h"
#include "plamatrix/ops/solver.h"

namespace plamatrix
{

namespace
{

template <typename Scalar>
void checkSolveInputs(const DenseMatrix<Scalar, Device::GPU>& A,
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
}

} // namespace

#ifdef PLAMATRIX_USE_FLOAT
template <>
DenseMatrix<float, Device::GPU> solve<float, Device::GPU>(const DenseMatrix<float, Device::GPU>& A,
                                                          const DenseMatrix<float, Device::GPU>& B)
{
    checkSolveInputs(A, B);
    DenseMatrix<float, Device::GPU> X(A.rows(), B.cols());
    detail::metalSolveFloat(A.data(), B.data(), X.data(), A.rows(), B.cols());
    return X;
}
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template <>
DenseMatrix<double, Device::GPU> solve<double, Device::GPU>(const DenseMatrix<double, Device::GPU>& A,
                                                            const DenseMatrix<double, Device::GPU>& B)
{
    checkSolveInputs(A, B);
    auto A_cpu = A.toCpu();
    auto B_cpu = B.toCpu();
    return solve(A_cpu, B_cpu).toGpu();
}
#endif

} // namespace plamatrix
