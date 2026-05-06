#include <sstream>
#include <stdexcept>

#include <omp.h>

#include "plamatrix/ops/gemm.h"

namespace plamatrix
{

template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> gemm(const DenseMatrix<Scalar, Device::CPU>& A,
                                       const DenseMatrix<Scalar, Device::CPU>& B)
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

    DenseMatrix<Scalar, Device::CPU> C(m, n);

    const Scalar* A_data = A.data();
    const Scalar* B_data = B.data();
    Scalar* C_data = C.data();

    #pragma omp parallel for collapse(2)
    for (Index j = 0; j < n; ++j)
    {
        for (Index i = 0; i < m; ++i)
        {
            Scalar sum = Scalar(0);
            for (Index p = 0; p < k; ++p)
            {
                sum += A_data[i + p * m] * B_data[p + j * k];
            }
            C_data[i + j * m] = sum;
        }
    }

    return C;
}

// Explicit template instantiations
template DenseMatrix<float, Device::CPU> gemm(const DenseMatrix<float, Device::CPU>&,
                                                const DenseMatrix<float, Device::CPU>&);
template DenseMatrix<double, Device::CPU> gemm(const DenseMatrix<double, Device::CPU>&,
                                                 const DenseMatrix<double, Device::CPU>&);

} // namespace plamatrix
