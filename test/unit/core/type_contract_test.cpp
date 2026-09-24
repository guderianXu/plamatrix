#include <type_traits>

#include <gtest/gtest.h>

#include "plamatrix/internal/core/api.h"
#include "plamatrix/internal/core/layout.h"
#include "plamatrix/internal/core/memory_resource.h"
#include "plamatrix/internal/core/memory_space.h"
#include "plamatrix/internal/core/scalar_traits.h"
#include "plamatrix/internal/core/shape.h"
#include "plamatrix/internal/core/device.h"
#include "plamatrix/internal/dense/matrix_view.h"

namespace plamatrix::internal
{

static_assert(PLAMATRIX_VERSION_MAJOR == 1);
static_assert(isSupportedScalar_v<float>);
static_assert(isSupportedScalar_v<double>);
static_assert(isSupportedScalar_v<std::int32_t>);
static_assert(isSupportedScalar_v<std::int64_t>);
static_assert(!isSupportedScalar_v<long double>);
static_assert(std::is_same_v<ScalarTraits<float>::AccumulationType, float>);
static_assert(std::is_same_v<ScalarTraits<float>::IndexType, Index>);

TEST(TypeContract, ShapeChecksOverflowAndElementCount)
{
    const Shape2 shape(3, 4);
    EXPECT_EQ(shape.rows, 3);
    EXPECT_EQ(shape.cols, 4);
    EXPECT_EQ(shape.elementCount(), 12U);
    EXPECT_THROW(Shape2(-1, 2), std::invalid_argument);
}

TEST(TypeContract, LayoutAndMemorySpaceAreExplicit)
{
    EXPECT_EQ(Layout::ColumnMajor, Layout::ColumnMajor);
    EXPECT_NE(Layout::ColumnMajor, Layout::RowMajor);
    EXPECT_NE(MemorySpace::Host, MemorySpace::Cuda);
}

TEST(TypeContract, MemoryResourceIsAnExplicitSeam)
{
    EXPECT_TRUE(std::is_abstract_v<MemoryResource>);
}

TEST(TypeContract, MutableViewCanBeMadeConstExplicitly)
{
    float values[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    auto mutable_view = makeColumnMajorView<float, Device::CPU>(values, 2, 2);
    auto const_view = mutable_view.asConst();
    static_assert(decltype(const_view)::isConst());
    EXPECT_EQ(const_view(1, 1), 4.0F);
}

} // namespace plamatrix::internal
