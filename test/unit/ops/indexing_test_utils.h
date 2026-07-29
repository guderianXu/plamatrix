#pragma once

#include <initializer_list>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include "plamatrix/plamatrix.h"
#include "../../support/cuda_test_utils.h"

namespace plamatrix
{
namespace indexing_test_detail
{

template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> makeMatrix(
    Index rows,
    Index cols,
    std::initializer_list<Scalar> values)
{
    DenseMatrix<Scalar, Device::CPU> matrix(rows, cols);
    if (matrix.size() != static_cast<Index>(values.size()))
    {
        throw std::invalid_argument("makeMatrix: initializer length must match matrix size");
    }

    Index offset = 0;
    for (Scalar value : values)
    {
        matrix.data()[offset++] = value;
    }
    return matrix;
}

template <typename Scalar>
void expectMatrix(
    const DenseMatrix<Scalar, Device::CPU>& actual,
    Index rows,
    Index cols,
    std::initializer_list<Scalar> expected)
{
    ASSERT_EQ(actual.rows(), rows);
    ASSERT_EQ(actual.cols(), cols);
    ASSERT_EQ(actual.size(), static_cast<Index>(expected.size()));

    Index offset = 0;
    for (Scalar value : expected)
    {
        EXPECT_EQ(actual.data()[offset++], value);
    }
}

#if defined(PLAMATRIX_USE_FLOAT) && defined(PLAMATRIX_USE_DOUBLE)
using ScalarTypes = ::testing::Types<float, double>;
#elif defined(PLAMATRIX_USE_FLOAT)
using ScalarTypes = ::testing::Types<float>;
#elif defined(PLAMATRIX_USE_DOUBLE)
using ScalarTypes = ::testing::Types<double>;
#else
#error "Indexing tests require PLAMATRIX_USE_FLOAT or PLAMATRIX_USE_DOUBLE"
#endif

template <typename Scalar>
class IndexingTest : public ::testing::Test
{
};

#ifdef PLAMATRIX_WITH_CUDA

template <typename Scalar>
void expectGpuMatrix(
    const DenseMatrix<Scalar, Device::GPU>& actual,
    Index rows,
    Index cols,
    std::initializer_list<Scalar> expected)
{
    expectMatrix(actual.toCpu(), rows, cols, expected);
}

template <typename Function>
void expectLogicErrorContaining(
    Function&& function,
    std::initializer_list<const char*> fragments)
{
    try
    {
        function();
        FAIL() << "Expected std::logic_error";
    }
    catch (const std::logic_error& error)
    {
        const std::string message = error.what();
        for (const char* fragment : fragments)
        {
            EXPECT_NE(message.find(fragment), std::string::npos) << message;
        }
    }
}

#endif

} // namespace indexing_test_detail
} // namespace plamatrix
