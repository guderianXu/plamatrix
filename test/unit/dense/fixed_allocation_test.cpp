#include <atomic>
#include <cstdlib>
#include <new>

#ifdef _MSC_VER
#include <malloc.h>
#endif

#include <gtest/gtest.h>

#include "plamatrix/plamatrix.h"

namespace
{
    std::atomic<bool> count_allocations{false};
    std::atomic<std::size_t> allocation_count{0};

    void recordAllocation() noexcept
    {
        if (count_allocations.load(std::memory_order_relaxed))
            allocation_count.fetch_add(1, std::memory_order_relaxed);
    }

    void* allocate(std::size_t size)
    {
        recordAllocation();
        if (void* result = std::malloc(size == 0 ? 1 : size))
            return result;
        throw std::bad_alloc();
    }

    void* allocateAligned(std::size_t size, std::size_t alignment)
    {
        recordAllocation();
#ifdef _MSC_VER
        if (void* result = _aligned_malloc(size == 0 ? 1 : size, alignment))
            return result;
#else
        void* result = nullptr;
        if (posix_memalign(&result, alignment, size == 0 ? 1 : size) == 0)
            return result;
#endif
        throw std::bad_alloc();
    }

    void deallocateAligned(void* pointer) noexcept
    {
#ifdef _MSC_VER
        _aligned_free(pointer);
#else
        std::free(pointer);
#endif
    }
} // namespace

void* operator new(std::size_t size)
{
    return allocate(size);
}
void* operator new[](std::size_t size)
{
    return allocate(size);
}
void* operator new(std::size_t size, std::align_val_t alignment)
{
    return allocateAligned(size, static_cast<std::size_t>(alignment));
}
void* operator new[](std::size_t size, std::align_val_t alignment)
{
    return allocateAligned(size, static_cast<std::size_t>(alignment));
}
void operator delete(void* pointer) noexcept
{
    std::free(pointer);
}
void operator delete[](void* pointer) noexcept
{
    std::free(pointer);
}
void operator delete(void* pointer, std::size_t) noexcept
{
    std::free(pointer);
}
void operator delete[](void* pointer, std::size_t) noexcept
{
    std::free(pointer);
}
void operator delete(void* pointer, std::align_val_t) noexcept
{
    deallocateAligned(pointer);
}
void operator delete[](void* pointer, std::align_val_t) noexcept
{
    deallocateAligned(pointer);
}
void operator delete(void* pointer, std::size_t, std::align_val_t) noexcept
{
    deallocateAligned(pointer);
}
void operator delete[](void* pointer, std::size_t, std::align_val_t) noexcept
{
    deallocateAligned(pointer);
}

TEST(FixedAllocation, CommonThreeByThreeDecompositionsUseNoHeap)
{
    plamatrix::Matrix3d general;
    general << 4.0, 1.0, -2.0, 1.5, 3.0, 0.5, -1.0, 2.0, 5.0;
    plamatrix::Matrix3d positive;
    positive << 5.0, 1.0, 0.5, 1.0, 4.0, 0.25, 0.5, 0.25, 3.0;
    const plamatrix::Vector3d expected(1.0, -2.0, 0.5);
    const plamatrix::Vector3d general_rhs = general * expected;
    const plamatrix::Vector3d positive_rhs = positive * expected;

    // Warm up process-wide runtime state before observing allocations from the numeric objects.
    static_cast<void>(general.partialPivLu());

    allocation_count.store(0, std::memory_order_relaxed);
    count_allocations.store(true, std::memory_order_relaxed);
    const auto partial_lu = general.partialPivLu();
    const auto after_partial_lu = allocation_count.load(std::memory_order_relaxed);
    const auto full_lu = general.fullPivLu();
    const auto after_full_lu = allocation_count.load(std::memory_order_relaxed);
    const auto llt = positive.llt();
    const auto after_llt = allocation_count.load(std::memory_order_relaxed);
    const auto ldlt = positive.ldlt();
    const auto after_ldlt = allocation_count.load(std::memory_order_relaxed);
    const auto qr = general.householderQr();
    const auto after_qr = allocation_count.load(std::memory_order_relaxed);
    const auto pivoted_qr = general.colPivHouseholderQr();
    const auto after_pivoted_qr = allocation_count.load(std::memory_order_relaxed);
    const auto svd = general.jacobiSvd<plamatrix::ComputeFullU | plamatrix::ComputeFullV>();
    const auto after_svd = allocation_count.load(std::memory_order_relaxed);
    const plamatrix::SelfAdjointEigenSolver<plamatrix::Matrix3d> eigen(positive);
    const auto after_eigen = allocation_count.load(std::memory_order_relaxed);
    const auto partial_solution = partial_lu.solve(general_rhs);
    const auto after_partial_solve = allocation_count.load(std::memory_order_relaxed);
    const auto full_solution = full_lu.solve(general_rhs);
    const auto after_full_solve = allocation_count.load(std::memory_order_relaxed);
    const auto llt_solution = llt.solve(positive_rhs);
    const auto after_llt_solve = allocation_count.load(std::memory_order_relaxed);
    const auto ldlt_solution = ldlt.solve(positive_rhs);
    const auto after_ldlt_solve = allocation_count.load(std::memory_order_relaxed);
    const auto qr_solution = qr.solve(general_rhs);
    const auto after_qr_solve = allocation_count.load(std::memory_order_relaxed);
    const auto pivoted_qr_solution = pivoted_qr.solve(general_rhs);
    const auto after_pivoted_qr_solve = allocation_count.load(std::memory_order_relaxed);
    const auto svd_solution = svd.solve(general_rhs);
    const auto after_svd_solve = allocation_count.load(std::memory_order_relaxed);
    count_allocations.store(false, std::memory_order_relaxed);

    EXPECT_EQ(llt.info(), plamatrix::Success);
    EXPECT_EQ(qr.info(), plamatrix::Success);
    EXPECT_EQ(pivoted_qr.info(), plamatrix::Success);
    EXPECT_EQ(svd.info(), plamatrix::Success);
    EXPECT_EQ(eigen.info(), plamatrix::Success);
    EXPECT_TRUE(partial_solution.isApprox(expected, 1.0e-12));
    EXPECT_TRUE(full_solution.isApprox(expected, 1.0e-12));
    EXPECT_TRUE(llt_solution.isApprox(expected, 1.0e-12));
    EXPECT_TRUE(ldlt_solution.isApprox(expected, 1.0e-12));
    EXPECT_TRUE(qr_solution.isApprox(expected, 1.0e-12));
    EXPECT_TRUE(pivoted_qr_solution.isApprox(expected, 1.0e-12));
    EXPECT_TRUE(svd_solution.isApprox(expected, 1.0e-10));
    EXPECT_EQ(allocation_count.load(std::memory_order_relaxed), 0U)
        << "partial=" << after_partial_lu << " full=" << after_full_lu << " llt=" << after_llt << " ldlt=" << after_ldlt
        << " qr=" << after_qr << " pivoted_qr=" << after_pivoted_qr << " svd=" << after_svd << " eigen=" << after_eigen
        << " partial_solve=" << after_partial_solve << " full_solve=" << after_full_solve
        << " llt_solve=" << after_llt_solve << " ldlt_solve=" << after_ldlt_solve << " qr_solve=" << after_qr_solve
        << " pivoted_qr_solve=" << after_pivoted_qr_solve << " svd_solve=" << after_svd_solve;
}

namespace
{
    template <int Size> void verifyPhotogrammetryBlockUsesNoHeap()
    {
        using Matrix = plamatrix::Matrix<double, Size, Size>;
        using Vector = plamatrix::Matrix<double, Size, 1>;

        Matrix general = Matrix::Zero();
        Matrix positive = Matrix::Zero();
        Vector expected;
        for (plamatrix::Index row = 0; row < Size; ++row)
        {
            expected(row) = 0.25 * static_cast<double>(row + 1);
            general(row, row) = static_cast<double>(Size + 2);
            positive(row, row) = 4.0;
            if (row + 1 < Size)
            {
                general(row, row + 1) = 0.5;
                general(row + 1, row) = -0.25;
                positive(row, row + 1) = 0.5;
                positive(row + 1, row) = 0.5;
            }
        }
        const Vector general_rhs = general * expected;
        const Vector positive_rhs = positive * expected;

        static_cast<void>(general.partialPivLu());
        allocation_count.store(0, std::memory_order_relaxed);
        count_allocations.store(true, std::memory_order_relaxed);
        const auto partial_lu = general.partialPivLu();
        const auto full_lu = general.fullPivLu();
        const auto llt = positive.llt();
        const auto ldlt = positive.ldlt();
        const auto qr = general.householderQr();
        const auto pivoted_qr = general.colPivHouseholderQr();
        const auto svd = general.template jacobiSvd<plamatrix::ComputeFullU | plamatrix::ComputeFullV>();
        const plamatrix::SelfAdjointEigenSolver<Matrix> eigen(positive);
        const auto partial_solution = partial_lu.solve(general_rhs);
        const auto full_solution = full_lu.solve(general_rhs);
        const auto llt_solution = llt.solve(positive_rhs);
        const auto ldlt_solution = ldlt.solve(positive_rhs);
        const auto qr_solution = qr.solve(general_rhs);
        const auto pivoted_qr_solution = pivoted_qr.solve(general_rhs);
        const auto svd_solution = svd.solve(general_rhs);
        count_allocations.store(false, std::memory_order_relaxed);

        EXPECT_EQ(allocation_count.load(std::memory_order_relaxed), 0U) << "fixed block size=" << Size;
        EXPECT_EQ(llt.info(), plamatrix::Success);
        EXPECT_EQ(qr.info(), plamatrix::Success);
        EXPECT_EQ(pivoted_qr.info(), plamatrix::Success);
        EXPECT_EQ(svd.info(), plamatrix::Success);
        EXPECT_EQ(eigen.info(), plamatrix::Success);
        EXPECT_TRUE(partial_solution.isApprox(expected, 1.0e-11));
        EXPECT_TRUE(full_solution.isApprox(expected, 1.0e-11));
        EXPECT_TRUE(llt_solution.isApprox(expected, 1.0e-11));
        EXPECT_TRUE(ldlt_solution.isApprox(expected, 1.0e-11));
        EXPECT_TRUE(qr_solution.isApprox(expected, 1.0e-11));
        EXPECT_TRUE(pivoted_qr_solution.isApprox(expected, 1.0e-11));
        EXPECT_TRUE(svd_solution.isApprox(expected, 1.0e-9));
    }
} // namespace

TEST(FixedAllocation, PhotogrammetryBlockSizesUseNoHeap)
{
    verifyPhotogrammetryBlockUsesNoHeap<6>();
    verifyPhotogrammetryBlockUsesNoHeap<9>();
    verifyPhotogrammetryBlockUsesNoHeap<15>();
}

TEST(FixedAllocation, MixedFixedAndDynamicDotDoesNotMaterializeAnOperand)
{
    const plamatrix::Vector3d fixed(1.0, 2.0, 3.0);
    plamatrix::VectorXd dynamic(3);
    dynamic << 4.0, 5.0, 6.0;

    allocation_count.store(0, std::memory_order_relaxed);
    count_allocations.store(true, std::memory_order_relaxed);
    const double fixed_dynamic = fixed.dot(dynamic);
    const double dynamic_fixed = dynamic.dot(fixed);
    count_allocations.store(false, std::memory_order_relaxed);

    EXPECT_DOUBLE_EQ(fixed_dynamic, 32.0);
    EXPECT_DOUBLE_EQ(dynamic_fixed, 32.0);
    EXPECT_EQ(allocation_count.load(std::memory_order_relaxed), 0U);
}
