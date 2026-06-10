#include <sstream>
#include <stdexcept>

#include <omp.h>

#include "plamatrix/core/parallel.h"
#include "plamatrix/ops/gemm.h"

#ifdef PLAMATRIX_WITH_BLAS
#include "fortran_linalg.h"
#endif

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

#ifdef PLAMATRIX_WITH_BLAS
    int m_int = detail::checkedLapackInt(m, "GEMM m");
    int n_int = detail::checkedLapackInt(n, "GEMM n");
    int k_int = detail::checkedLapackInt(k, "GEMM k");
#endif

    DenseMatrix<Scalar, Device::CPU> C(m, n);
    if (m == 0 || n == 0 || k == 0)
    {
        return C;
    }

    const Scalar* A_data = A.data();
    const Scalar* B_data = B.data();
    Scalar* C_data = C.data();

#ifdef PLAMATRIX_WITH_BLAS
    detail::fortranGemm(m_int, n_int, k_int, A_data, B_data, C_data);
#else
    Index work_items = m * n * k;
    if (detail::shouldUseOpenMp(work_items))
    {
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
    }
    else
    {
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
    }
#endif

    return C;
}

// Explicit template instantiations
#ifdef PLAMATRIX_USE_FLOAT
template DenseMatrix<float, Device::CPU> gemm(const DenseMatrix<float, Device::CPU>&,
                                                const DenseMatrix<float, Device::CPU>&);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template DenseMatrix<double, Device::CPU> gemm(const DenseMatrix<double, Device::CPU>&,
                                                 const DenseMatrix<double, Device::CPU>&);
#endif

} // namespace plamatrix
