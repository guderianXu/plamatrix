#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include "plamatrix/internal/ops/solver.h"

namespace plamatrix::internal
{

    template <typename Scalar>
    DenseStorage<Scalar, Device::CPU> solveCpu(const DenseStorage<Scalar, Device::CPU>& A,
                                               const DenseStorage<Scalar, Device::CPU>& B)
    {
        Index n = A.rows();
        Index nrhs = B.cols();

        // Validate dimensions
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
            oss << "Solve: dimension mismatch. A is " << n << "x" << n << ", B is " << B.rows() << "x" << nrhs;
            throw std::runtime_error(oss.str());
        }

        // Working copies: LU holds the LU factors, X holds the solution / modified RHS
        DenseStorage<Scalar, Device::CPU> LU(n, n);
        DenseStorage<Scalar, Device::CPU> X(n, nrhs);
        for (Index j = 0; j < n; ++j)
        {
            for (Index i = 0; i < n; ++i)
            {
                LU(i, j) = A(i, j);
            }
        }
        for (Index j = 0; j < nrhs; ++j)
        {
            for (Index i = 0; i < n; ++i)
            {
                X(i, j) = B(i, j);
            }
        }

        Scalar epsilon = static_cast<Scalar>(1e-15);

        // LU factorization with partial pivoting (in-place)
        for (Index k = 0; k < n; ++k)
        {
            // Find pivot: row with largest |element| in column k, at or below diagonal
            Index pivot_row = k;
            Scalar max_val = std::abs(LU(k, k));
            for (Index i = k + 1; i < n; ++i)
            {
                Scalar abs_val = std::abs(LU(i, k));
                if (abs_val > max_val)
                {
                    max_val = abs_val;
                    pivot_row = i;
                }
            }

            if (max_val < epsilon)
            {
                std::ostringstream oss;
                oss << "Solve: matrix is singular (zero pivot at column " << k << ")";
                throw std::runtime_error(oss.str());
            }

            // Swap rows in LU and X if pivot is not on diagonal
            if (pivot_row != k)
            {
                for (Index j = k; j < n; ++j)
                {
                    std::swap(LU(pivot_row, j), LU(k, j));
                }
                for (Index j = 0; j < nrhs; ++j)
                {
                    std::swap(X(pivot_row, j), X(k, j));
                }
            }

            // Eliminate below diagonal
            Scalar pivot = LU(k, k);
            for (Index i = k + 1; i < n; ++i)
            {
                LU(i, k) /= pivot;
            }
            for (Index j = k + 1; j < n; ++j)
            {
                const Scalar pivot_value = LU(k, j);
#pragma omp simd
                for (Index i = k + 1; i < n; ++i)
                    LU(i, j) -= LU(i, k) * pivot_value;
            }
            for (Index j = 0; j < nrhs; ++j)
            {
                const Scalar pivot_value = X(k, j);
#pragma omp simd
                for (Index i = k + 1; i < n; ++i)
                    X(i, j) -= LU(i, k) * pivot_value;
            }
        }

        // Back-substitution: solve U * X = (modified RHS) where U is upper triangular
        for (Index j = 0; j < nrhs; ++j)
        {
            for (Index i = n; i-- > 0;)
            {
                X(i, j) /= LU(i, i);
                const Scalar solved = X(i, j);
#pragma omp simd
                for (Index previous = 0; previous < i; ++previous)
                    X(previous, j) -= LU(previous, i) * solved;
            }
        }

        return X;
    }

    template <typename Scalar, Device Dev>
    DenseStorage<Scalar, Dev> solve(const DenseStorage<Scalar, Dev>& A, const DenseStorage<Scalar, Dev>& B)
    {
        // Dispatch to CPU implementation (GPU specialization lives in solver.cu)
        if constexpr (Dev == Device::CPU)
        {
            return solveCpu(A, B);
        }
        else
        {
            // GPU path: should be provided via solver.cu (explicit instantiation)
            // This static_assert ensures we don't accidentally try to use CPU fallback on GPU
            static_assert(Dev == Device::CPU, "GPU solve requires CUDA backend (solver.cu)");
            return DenseStorage<Scalar, Dev>(0, 0); // unreachable
        }
    }

// Explicit template instantiations for CPU
#ifdef PLAMATRIX_USE_FLOAT
    template DenseStorage<float, Device::CPU> solve(const DenseStorage<float, Device::CPU>&,
                                                    const DenseStorage<float, Device::CPU>&);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
    template DenseStorage<double, Device::CPU> solve(const DenseStorage<double, Device::CPU>&,
                                                     const DenseStorage<double, Device::CPU>&);
#endif

} // namespace plamatrix::internal
