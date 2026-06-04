#pragma once

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "plamatrix/core/types.h"

extern "C"
{
void sgemm_(const char* transa, const char* transb,
            const int* m, const int* n, const int* k,
            const float* alpha, const float* a, const int* lda,
            const float* b, const int* ldb,
            const float* beta, float* c, const int* ldc);
void dgemm_(const char* transa, const char* transb,
            const int* m, const int* n, const int* k,
            const double* alpha, const double* a, const int* lda,
            const double* b, const int* ldb,
            const double* beta, double* c, const int* ldc);

void sgesvd_(const char* jobu, const char* jobvt,
             const int* m, const int* n,
             float* a, const int* lda,
             float* s, float* u, const int* ldu,
             float* vt, const int* ldvt,
             float* work, const int* lwork, int* info);
void dgesvd_(const char* jobu, const char* jobvt,
             const int* m, const int* n,
             double* a, const int* lda,
             double* s, double* u, const int* ldu,
             double* vt, const int* ldvt,
             double* work, const int* lwork, int* info);

void ssyev_(const char* jobz, const char* uplo,
            const int* n, float* a, const int* lda,
            float* w, float* work, const int* lwork, int* info);
void dsyev_(const char* jobz, const char* uplo,
            const int* n, double* a, const int* lda,
            double* w, double* work, const int* lwork, int* info);
}

namespace plamatrix
{
namespace detail
{

inline int checkedLapackInt(Index value, const char* name)
{
    if (value < 0 || value > static_cast<Index>(std::numeric_limits<int>::max()))
    {
        std::ostringstream oss;
        oss << name << " is outside LAPACK int range: " << value;
        throw std::runtime_error(oss.str());
    }
    return static_cast<int>(value);
}

template <typename Scalar>
void fortranGemm(int m,
                 int n,
                 int k,
                 const Scalar* A,
                 const Scalar* B,
                 Scalar* C)
{
    const char trans = 'N';
    const Scalar alpha = static_cast<Scalar>(1);
    const Scalar beta = static_cast<Scalar>(0);
    if constexpr (std::is_same_v<Scalar, float>)
    {
        sgemm_(&trans, &trans, &m, &n, &k, &alpha, A, &m, B, &k, &beta, C, &m);
    }
    else
    {
        dgemm_(&trans, &trans, &m, &n, &k, &alpha, A, &m, B, &k, &beta, C, &m);
    }
}

template <typename Scalar>
void fortranGesvd(int m,
                  int n,
                  Scalar* A,
                  Scalar* S,
                  Scalar* U,
                  Scalar* Vt)
{
    const char job_all = 'A';
    const int lda = m;
    const int ldu = m;
    const int ldvt = n;
    int info = 0;
    int lwork = -1;
    Scalar work_query = Scalar(0);

    if constexpr (std::is_same_v<Scalar, float>)
    {
        sgesvd_(&job_all, &job_all, &m, &n, A, &lda, S, U, &ldu, Vt, &ldvt, &work_query, &lwork, &info);
    }
    else
    {
        dgesvd_(&job_all, &job_all, &m, &n, A, &lda, S, U, &ldu, Vt, &ldvt, &work_query, &lwork, &info);
    }
    if (info != 0)
    {
        throw std::runtime_error("SVD: LAPACK workspace query failed");
    }

    lwork = std::max(1, static_cast<int>(work_query));
    std::vector<Scalar> work(static_cast<std::size_t>(lwork));

    if constexpr (std::is_same_v<Scalar, float>)
    {
        sgesvd_(&job_all, &job_all, &m, &n, A, &lda, S, U, &ldu, Vt, &ldvt, work.data(), &lwork, &info);
    }
    else
    {
        dgesvd_(&job_all, &job_all, &m, &n, A, &lda, S, U, &ldu, Vt, &ldvt, work.data(), &lwork, &info);
    }

    if (info < 0)
    {
        std::ostringstream oss;
        oss << "SVD: LAPACK gesvd invalid argument " << -info;
        throw std::runtime_error(oss.str());
    }
    if (info > 0)
    {
        std::ostringstream oss;
        oss << "SVD: LAPACK gesvd did not converge, " << info << " superdiagonals did not converge";
        throw std::runtime_error(oss.str());
    }
}

template <typename Scalar>
void fortranSyev(int n, Scalar* A, Scalar* eigenvalues)
{
    const char job_values_only = 'N';
    const char upper = 'U';
    const int lda = n;
    int info = 0;
    int lwork = -1;
    Scalar work_query = Scalar(0);

    if constexpr (std::is_same_v<Scalar, float>)
    {
        ssyev_(&job_values_only, &upper, &n, A, &lda, eigenvalues, &work_query, &lwork, &info);
    }
    else
    {
        dsyev_(&job_values_only, &upper, &n, A, &lda, eigenvalues, &work_query, &lwork, &info);
    }
    if (info != 0)
    {
        throw std::runtime_error("Eigh: LAPACK workspace query failed");
    }

    lwork = std::max(1, static_cast<int>(work_query));
    std::vector<Scalar> work(static_cast<std::size_t>(lwork));

    if constexpr (std::is_same_v<Scalar, float>)
    {
        ssyev_(&job_values_only, &upper, &n, A, &lda, eigenvalues, work.data(), &lwork, &info);
    }
    else
    {
        dsyev_(&job_values_only, &upper, &n, A, &lda, eigenvalues, work.data(), &lwork, &info);
    }

    if (info < 0)
    {
        std::ostringstream oss;
        oss << "Eigh: LAPACK syev invalid argument " << -info;
        throw std::runtime_error(oss.str());
    }
    if (info > 0)
    {
        std::ostringstream oss;
        oss << "Eigh: LAPACK syev did not converge, " << info << " off-diagonal elements remain";
        throw std::runtime_error(oss.str());
    }
}

} // namespace detail
} // namespace plamatrix
