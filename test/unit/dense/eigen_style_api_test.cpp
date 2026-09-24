#include <type_traits>
#include <limits>
#include <sstream>

#include <gtest/gtest.h>

#ifdef PLAMATRIX_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

#include "plamatrix/plamatrix.h"
#include "plamatrix/internal/backend.h"

namespace plamatrix::internal
{

    TEST(EigenStyleMatrix, FixedAndDynamicTypesUseNamedFactories)
    {
        static_assert(Matrix3d::RowsAtCompileTime == 3);
        static_assert(Matrix3d::ColsAtCompileTime == 3);
        static_assert(Vector3d::ColsAtCompileTime == 1);
        static_assert(VectorXd::RowsAtCompileTime == Dynamic);

        Vector3d point(1.0, 2.0, 3.0);
        Matrix3d rotation = Matrix3d::Identity();
        Vector3d rotated = rotation * point;
        MatrixXd measurements = MatrixXd::Zero(100, 3);
        VectorXd weights = VectorXd::Ones(100);

        EXPECT_DOUBLE_EQ(rotated(0), 1.0);
        EXPECT_DOUBLE_EQ(rotated(1), 2.0);
        EXPECT_DOUBLE_EQ(rotated(2), 3.0);
        EXPECT_EQ(measurements.rows(), 100);
        EXPECT_EQ(measurements.cols(), 3);
        EXPECT_DOUBLE_EQ(measurements(99, 2), 0.0);
        EXPECT_DOUBLE_EQ(weights(99), 1.0);

        MatrixXd dynamic_rotation = rotation;
        Vector3d dynamic_result = dynamic_rotation * point;
        EXPECT_DOUBLE_EQ(dynamic_result(2), 3.0);
        EXPECT_THROW(static_cast<void>(Matrix3d(MatrixXd::Zero(2, 2))), Error);
        Matrix3d mixed_sum = rotation + MatrixXd::Identity(3, 3);
        EXPECT_DOUBLE_EQ(mixed_sum(0, 0), 2.0);

        measurements.resize(4, 3);
        EXPECT_EQ(measurements.rows(), 4);
        measurements.block(1, 1, 2, 2)(0, 0) = 7.0;
        EXPECT_DOUBLE_EQ(measurements.row(1)(0, 1), 7.0);
        EXPECT_DOUBLE_EQ(measurements.col(1)(1, 0), 7.0);
        weights.resize(4);
        EXPECT_EQ(weights.size(), 4);
    }

    TEST(EigenStyleMatrix, CommaInitializerFillsRowsAcrossColumnMajorStorage)
    {
        Matrix3d fixed;
        fixed << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0;
        EXPECT_DOUBLE_EQ(fixed(0, 1), 2.0);
        EXPECT_DOUBLE_EQ(fixed(1, 0), 4.0);
        EXPECT_DOUBLE_EQ(static_cast<const Matrix3d&>(fixed).data()[1], 4.0);

        MatrixXd dynamic(2, 3);
        dynamic << 10.0, 11.0, 12.0, 13.0, 14.0, 15.0;
        EXPECT_DOUBLE_EQ(dynamic(0, 2), 12.0);
        EXPECT_DOUBLE_EQ(dynamic(1, 0), 13.0);

        VectorXd vector(3);
        vector << 2.0, 4.0, 6.0;
        EXPECT_DOUBLE_EQ(vector(2), 6.0);

        std::ostringstream printed;
        printed << fixed;
        EXPECT_EQ(printed.str(), "1 2 3\n4 5 6\n7 8 9");
    }

    TEST(EigenStyleMatrix, CommaInitializerRejectsWrongCoefficientCount)
    {
        MatrixXd matrix(2, 2);
        const auto too_few = [&] { matrix << 1.0, 2.0, 3.0; };
        const auto too_many = [&] { matrix << 1.0, 2.0, 3.0, 4.0, 5.0; };
        EXPECT_THROW(too_few(), Error);
        EXPECT_THROW(too_many(), Error);

        MatrixXd empty;
        EXPECT_THROW(static_cast<void>(empty << 1.0), Error);
    }

    TEST(EigenStyleMatrix, OperatorsHaveLinearAlgebraSemantics)
    {
        MatrixXd left(2, 2);
        left(0, 0) = 1.0;
        left(1, 0) = 3.0;
        left(0, 1) = 2.0;
        left(1, 1) = 4.0;
        MatrixXd right = MatrixXd::Identity(2, 2);

        MatrixXd product = left * right;
        MatrixXd sum = product + right;
        MatrixXd difference = sum - right;
        MatrixXd coefficients = left.cwiseProduct(left);
        MatrixXd transposed = left.transpose();

        EXPECT_DOUBLE_EQ(product(1, 1), 4.0);
        EXPECT_DOUBLE_EQ(sum(0, 0), 2.0);
        EXPECT_DOUBLE_EQ(difference(1, 0), 3.0);
        EXPECT_DOUBLE_EQ(coefficients(1, 0), 9.0);
        EXPECT_DOUBLE_EQ(transposed(0, 1), 3.0);
        EXPECT_DOUBLE_EQ((2.0 * left)(0, 1), 4.0);
        EXPECT_DOUBLE_EQ((left / 2.0)(1, 1), 2.0);

        MatrixXd accumulated = left;
        accumulated += right;
        accumulated -= right;
        accumulated *= 2.0;
        accumulated /= 2.0;
        EXPECT_DOUBLE_EQ(accumulated(1, 1), 4.0);

        Vector3d vector(1.0, 2.0, 2.0);
        EXPECT_DOUBLE_EQ(vector.dot(vector), 9.0);
        EXPECT_DOUBLE_EQ(vector.norm(), 3.0);
        EXPECT_DOUBLE_EQ(vector.squaredNorm(), 9.0);
        EXPECT_NEAR(vector.normalized().norm(), 1.0, 1e-12);
        EXPECT_THROW(Vector3d::Zero().normalized(), Error);
        EXPECT_TRUE(vector.allFinite());
        Vector3d right_axis(0.0, 1.0, 0.0);
        Vector3d cross = Vector3d(1.0, 0.0, 0.0).cross(right_axis);
        EXPECT_DOUBLE_EQ(cross(2), 1.0);
    }

    TEST(EigenStyleMatrix, IntegerScalarDivisionUsesCoefficientQuotients)
    {
        Matrix<int, 3, 1> values(5, 8, -9);
        const auto quotients = values / 2;
        EXPECT_EQ(quotients(0), 2);
        EXPECT_EQ(quotients(1), 4);
        EXPECT_EQ(quotients(2), -4);
    }

    TEST(EigenStyleMatrix, CopyIsDeepAndShapesAreChecked)
    {
        MatrixXd original = MatrixXd::Ones(2, 2);
        MatrixXd copy = original;
        copy(0, 0) = 5.0;

        EXPECT_DOUBLE_EQ(original(0, 0), 1.0);
        EXPECT_DOUBLE_EQ(copy(0, 0), 5.0);
        EXPECT_THROW(static_cast<void>(Matrix3d(2, 2)), Error);
        EXPECT_THROW(static_cast<void>(MatrixXd::Zero(-1, 2)), Error);
        EXPECT_THROW(static_cast<void>(original * MatrixXd::Zero(3, 1)), Error);
        Vector3d non_finite(1.0, std::numeric_limits<double>::quiet_NaN(), 3.0);
        EXPECT_FALSE(non_finite.allFinite());
    }

    TEST(EigenStyleMatrix, BlockAndVectorSegmentsWriteThroughAndHandleOverlap)
    {
        MatrixXd matrix = MatrixXd::Zero(4, 4);
        Matrix2d block;
        block << 1.0, 2.0, 3.0, 4.0;
        matrix.block<2, 2>(1, 1) = block;
        EXPECT_DOUBLE_EQ(matrix(2, 2), 4.0);

        matrix.block(2, 1, 2, 2) = matrix.block(1, 1, 2, 2);
        EXPECT_DOUBLE_EQ(matrix(2, 1), 1.0);
        EXPECT_DOUBLE_EQ(matrix(3, 2), 4.0);

        VectorXd vector(5);
        vector << 1.0, 2.0, 3.0, 4.0, 5.0;
        vector.segment(1, 3) = vector.head(3);
        EXPECT_DOUBLE_EQ(vector(3), 3.0);
        vector.tail<2>() += Vector2d(10.0, 20.0);
        EXPECT_DOUBLE_EQ(vector(3), 13.0);
        EXPECT_DOUBLE_EQ(vector(4), 25.0);
        EXPECT_THROW(matrix.block(0, 0, 2, 2) = MatrixXd::Zero(3, 3), std::invalid_argument);
        EXPECT_THROW(vector.segment(4, 2), std::out_of_range);

        MatrixXd large = MatrixXd::Zero(6, 6);
        for (Index col = 0; col < 6; ++col)
        {
            for (Index row = 0; row < 6; ++row)
            {
                large(row, col) = static_cast<double>(row + 10 * col);
            }
        }
        large.block(1, 1, 5, 5) = large.block(0, 0, 5, 5);
        EXPECT_DOUBLE_EQ(large(5, 5), 44.0);
        EXPECT_DOUBLE_EQ(large(1, 1), 0.0);

        plamatrix::Matrix<double, 1, plamatrix::Dynamic> row_vector(1, 4);
        row_vector << 1.0, 2.0, 3.0, 4.0;
        row_vector.tail<2>() -= row_vector.head<2>();
        EXPECT_DOUBLE_EQ(row_vector.segment(2, 2)(0), 2.0);
        EXPECT_DOUBLE_EQ(row_vector.segment(2, 2)(1), 2.0);
    }

    TEST(EigenStyleMatrix, PartialPivotLuSolvesAndReusesFactorization)
    {
        Matrix3d coefficients;
        coefficients << 0.0, 2.0, 1.0, 1.0, 1.0, 0.0, 2.0, 0.0, 1.0;
        const auto factor = coefficients.partialPivLu();
        const Vector3d expected(1.0, 2.0, 3.0);
        const Vector3d right = coefficients * expected;
        const Vector3d solved = factor.solve(right);
        EXPECT_EQ(plamatrix::internal::MatrixAccess::executionInfo(solved).backend, Backend::Cpu);
        EXPECT_EQ(plamatrix::internal::MatrixAccess::executionInfo(solved).reason, "CPU partial-pivot LU solve");
        for (Index row = 0; row < 3; ++row)
        {
            EXPECT_NEAR(solved(row), expected(row), 1e-12);
        }

        MatrixXd right_multi(3, 2);
        right_multi.col(0) = right;
        right_multi.col(1) = 2.0 * right;
        const MatrixXd multi_solution = factor.solve(right_multi);
        EXPECT_NEAR(multi_solution(0, 1), 2.0, 1e-12);
        EXPECT_NEAR(multi_solution(2, 1), 6.0, 1e-12);
        EXPECT_THROW(static_cast<void>(factor.solve(VectorXd::Zero(2))), Error);
        EXPECT_THROW(static_cast<void>(MatrixXd::Zero(2, 3).partialPivLu()), Error);
        EXPECT_THROW(static_cast<void>(MatrixXd::Zero(2, 2).partialPivLu()), Error);
        Matrix2f float_coefficients;
        float_coefficients << 4.0f, 1.0f, 1.0f, 3.0f;
        const auto float_solution = float_coefficients.partialPivLu().solve(Vector2f(6.0f, 7.0f));
        EXPECT_NEAR(float_solution(0), 1.0f, 1e-6f);
        EXPECT_NEAR(float_solution(1), 2.0f, 1e-6f);
        {
            ScopedExecutionPolicy gpu_required(ExecutionPolicy::GpuRequired);
            EXPECT_THROW(static_cast<void>(coefficients.partialPivLu()), Error);
            EXPECT_THROW(static_cast<void>(factor.solve(right)), Error);
        }
    }

    TEST(EigenStyleMatrix, AutoKeepsSmallOperationsOnCpu)
    {
        MatrixXd left = MatrixXd::Ones(2, 2);
        MatrixXd right = MatrixXd::Identity(2, 2);
        MatrixXd result = left * right;

        EXPECT_EQ(plamatrix::internal::MatrixAccess::executionInfo(result).backend, Backend::Cpu);
        EXPECT_FALSE(plamatrix::internal::MatrixAccess::isDeviceResident(result));
        EXPECT_DOUBLE_EQ(result(1, 0), 1.0);
    }

    TEST(EigenStyleMatrix, ScopedPolicyRestoresPreviousSettings)
    {
        const auto previous = currentExecutionSettings();
        {
            ScopedExecutionPolicy cpu_only(ExecutionPolicy::CpuOnly);
            EXPECT_EQ(currentExecutionSettings().policy, ExecutionPolicy::CpuOnly);
            MatrixXd result = MatrixXd::Ones(2, 2) * MatrixXd::Identity(2, 2);
            EXPECT_EQ(plamatrix::internal::MatrixAccess::executionInfo(result).backend, Backend::Cpu);
        }
        EXPECT_EQ(currentExecutionSettings().policy, previous.policy);
        EXPECT_EQ(currentExecutionSettings().preferredGpu, previous.preferredGpu);
    }

    TEST(EigenStyleMatrix, AutoSelectsCudaForLargeProductWhenAvailable)
    {
#ifdef PLAMATRIX_WITH_CUDA
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        {
            GTEST_SKIP() << "CUDA device unavailable";
        }

        MatrixXf left = MatrixXf::Ones(512, 512);
        MatrixXf right = MatrixXf::Ones(512, 512);
        MatrixXf product = left * right;
        EXPECT_EQ(plamatrix::internal::MatrixAccess::executionInfo(product).backend, Backend::Cuda);
        EXPECT_TRUE(plamatrix::internal::MatrixAccess::isDeviceResident(product));
        EXPECT_EQ(plamatrix::internal::MatrixAccess::executionInfo(product).bytesUploaded,
                  2U * 512U * 512U * sizeof(float));
        const MatrixXf& result = product;
        EXPECT_FLOAT_EQ(result(0, 0), 512.0f);
        EXPECT_FLOAT_EQ(result(511, 511), 512.0f);
#else
        GTEST_SKIP() << "PlaMatrix was built without CUDA";
#endif
    }

    TEST(EigenStyleMatrix, AutoUsesFastestPortableBackendPriority)
    {
        const auto decision = plamatrix::detail::chooseDenseBackend<float>(
            plamatrix::detail::DenseOperation::Product, 128.0L * 1024.0L * 1024.0L, false);
        if (detail::GpuOps<float>::available(Backend::Cuda))
        {
            EXPECT_EQ(decision.backend, Backend::Cuda);
        }
        else if (detail::GpuOps<float>::available(Backend::OpenCl))
        {
            EXPECT_EQ(decision.backend, Backend::OpenCl);
        }
        else if (detail::GpuOps<float>::available(Backend::Vulkan))
        {
            EXPECT_EQ(decision.backend, Backend::Vulkan);
        }
        else
        {
            EXPECT_EQ(decision.backend, Backend::Cpu);
        }
    }

    TEST(EigenStyleMatrix, PreferredVulkanExecutesOrReportsUnavailable)
    {
        MatrixXf left = MatrixXf::Ones(2, 2);
        MatrixXf right = MatrixXf::Identity(2, 2);
        {
            ScopedExecutionPolicy prefer_vulkan(ExecutionPolicy::GpuPreferred, Backend::Vulkan);
            auto result = left * right;
            const auto backend = plamatrix::internal::MatrixAccess::executionInfo(result).backend;
            EXPECT_TRUE(backend == Backend::Vulkan || backend == Backend::Cpu);
            EXPECT_FLOAT_EQ(result(1, 0), 1.0f);
        }

        if (detail::GpuOps<float>::available(Backend::Vulkan))
        {
            ScopedExecutionPolicy require_vulkan(ExecutionPolicy::GpuRequired, Backend::Vulkan);
            const MatrixXf result = left * right;
            EXPECT_EQ(plamatrix::internal::MatrixAccess::executionInfo(result).backend, Backend::Vulkan);
            EXPECT_FLOAT_EQ(result(0, 0), 1.0f);
        }
        else
        {
            ScopedExecutionPolicy require_vulkan(ExecutionPolicy::GpuRequired, Backend::Vulkan);
            EXPECT_THROW(static_cast<void>((left * right).eval()), Error);
        }
    }

    TEST(EigenStyleMatrix, PreferredOpenClExecutesOrReportsUnavailable)
    {
        MatrixXd left = MatrixXd::Ones(2, 3);
        MatrixXd right = MatrixXd::Ones(3, 2);
        if (!detail::GpuOps<double>::available(Backend::OpenCl))
        {
            ScopedExecutionPolicy require_opencl(ExecutionPolicy::GpuRequired, Backend::OpenCl);
            EXPECT_THROW(static_cast<void>((left * right).eval()), Error);
            return;
        }

        ScopedExecutionPolicy require_opencl(ExecutionPolicy::GpuRequired, Backend::OpenCl);
        const MatrixXd result = left * right;
        EXPECT_EQ(plamatrix::internal::MatrixAccess::executionInfo(result).backend, Backend::OpenCl);
        EXPECT_DOUBLE_EQ(result(0, 0), 3.0);
        EXPECT_DOUBLE_EQ(result(1, 1), 3.0);
    }

    TEST(EigenStyleMatrix, PortableGemmHandlesPartialTiles)
    {
        MatrixXf left(17, 19);
        MatrixXf right(19, 23);
        for (Index column = 0; column < left.cols(); ++column)
        {
            for (Index row = 0; row < left.rows(); ++row)
            {
                left(row, column) = static_cast<float>((3 * row - 2 * column) % 11) / 7.0f;
            }
        }
        for (Index column = 0; column < right.cols(); ++column)
        {
            for (Index row = 0; row < right.rows(); ++row)
            {
                right(row, column) = static_cast<float>((row + 5 * column) % 13) / 9.0f;
            }
        }

        MatrixXf expected;
        {
            ScopedExecutionPolicy cpu_only(ExecutionPolicy::CpuOnly);
            expected = left * right;
        }
        for (const Backend backend : {Backend::Vulkan, Backend::OpenCl})
        {
            if (!detail::GpuOps<float>::available(backend))
            {
                continue;
            }
            ScopedExecutionPolicy required(ExecutionPolicy::GpuRequired, backend);
            const MatrixXf result = left * right;
            EXPECT_EQ(MatrixAccess::executionInfo(result).backend, backend);
            EXPECT_TRUE(result.isApprox(expected, 2.0e-5f));
        }
    }

    TEST(EigenStyleMatrix, OpenClDoubleGemmHandlesPartialTiles)
    {
        if (!detail::GpuOps<double>::available(Backend::OpenCl))
        {
            GTEST_SKIP() << "OpenCL double precision unavailable";
        }

        MatrixXd left(17, 19);
        MatrixXd right(19, 23);
        for (Index column = 0; column < left.cols(); ++column)
        {
            for (Index row = 0; row < left.rows(); ++row)
            {
                left(row, column) = static_cast<double>((2 * row + column) % 17) / 11.0;
            }
        }
        for (Index column = 0; column < right.cols(); ++column)
        {
            for (Index row = 0; row < right.rows(); ++row)
            {
                right(row, column) = static_cast<double>((row - 3 * column) % 19) / 13.0;
            }
        }

        MatrixXd expected;
        {
            ScopedExecutionPolicy cpu_only(ExecutionPolicy::CpuOnly);
            expected = left * right;
        }
        ScopedExecutionPolicy required(ExecutionPolicy::GpuRequired, Backend::OpenCl);
        const MatrixXd result = left * right;
        EXPECT_EQ(MatrixAccess::executionInfo(result).backend, Backend::OpenCl);
        EXPECT_TRUE(result.isApprox(expected, 1.0e-12));
    }

    TEST(EigenStyleMatrix, PortableBackendsKeepDenseOperationChainsResident)
    {
        for (const Backend backend : {Backend::Vulkan, Backend::OpenCl})
        {
            if (!detail::GpuOps<float>::available(backend))
            {
                continue;
            }
            ScopedExecutionPolicy required(ExecutionPolicy::GpuRequired, backend);
            MatrixXf left(2, 2);
            left << 1.0f, 2.0f, 3.0f, 4.0f;
            MatrixXf right = MatrixXf::Identity(2, 2);
            MatrixXf product = left * right;
            MatrixXf sum = product + MatrixXf::Ones(2, 2);
            MatrixXf scaled = sum * 2.0f;
            MatrixXf elementwise = scaled.cwiseProduct(MatrixXf::Constant(2, 2, 0.5f));
            EXPECT_EQ(MatrixAccess::executionInfo(elementwise).backend, backend);
            EXPECT_TRUE(MatrixAccess::isDeviceResident(elementwise));
            MatrixXf transposed = elementwise.transpose();

            const MatrixXf& result = transposed;
            EXPECT_FLOAT_EQ(result(0, 0), 2.0f);
            EXPECT_FLOAT_EQ(result(0, 1), 4.0f);
            EXPECT_FLOAT_EQ(result(1, 0), 3.0f);
            EXPECT_FLOAT_EQ(result(1, 1), 5.0f);
        }
    }

    TEST(EigenStyleMatrix, PortableBackendsComputeResidentReductions)
    {
        VectorXf seed(513);
        VectorXf other_seed(513);
        for (Index index = 0; index < seed.size(); ++index)
        {
            seed(index) = static_cast<float>((index % 11) - 5) / 7.0F;
            other_seed(index) = static_cast<float>((index % 7) + 1) / 9.0F;
        }

        float expected_sum = 0.0F;
        float expected_dot = 0.0F;
        float expected_squared_norm = 0.0F;
        {
            ScopedExecutionPolicy cpu_only(ExecutionPolicy::CpuOnly);
            expected_sum = seed.sum();
            expected_dot = seed.dot(other_seed);
            expected_squared_norm = seed.squaredNorm();
        }

        for (const Backend backend : {Backend::Cuda, Backend::OpenCl, Backend::Vulkan})
        {
            if (!detail::GpuOps<float>::available(backend))
            {
                continue;
            }
            VectorXf values = seed;
            VectorXf other = other_seed;
            ScopedExecutionPolicy required(ExecutionPolicy::GpuRequired, backend);
            EXPECT_NEAR(values.sum(), expected_sum, 2.0e-4F);
            EXPECT_TRUE(MatrixAccess::isDeviceResident(values));
            EXPECT_NEAR(values.dot(other), expected_dot, 2.0e-4F);
            EXPECT_TRUE(MatrixAccess::isDeviceResident(other));
            EXPECT_NEAR(values.squaredNorm(), expected_squared_norm, 2.0e-4F);
        }
    }

    TEST(EigenStyleMatrix, UnsupportedScalarReportsOperationError)
    {
        Matrix<int, Dynamic, Dynamic> left = Matrix<int, Dynamic, Dynamic>::Ones(2, 2);
        Matrix<int, Dynamic, Dynamic> right = Matrix<int, Dynamic, Dynamic>::Identity(2, 2);
        ScopedExecutionPolicy require_gpu(ExecutionPolicy::GpuRequired);
        try
        {
            static_cast<void>((left * right).eval());
            FAIL() << "Integer matrix multiplication should not use CUDA";
        }
        catch (const Error& error)
        {
            EXPECT_EQ(error.code(), ErrorCode::UnsupportedOperation);
        }
    }

    TEST(EigenStyleMatrix, GpuPreferredUsesResidentResultsWhenAvailable)
    {
        MatrixXd left = MatrixXd::Ones(2, 2);
        MatrixXd right = MatrixXd::Identity(2, 2);
        ScopedExecutionPolicy prefer_gpu(ExecutionPolicy::GpuPreferred);
        MatrixXd product = left * right;
        MatrixXd sum = product + right;

        if (plamatrix::internal::MatrixAccess::executionInfo(sum).backend != Backend::Cuda)
        {
            EXPECT_EQ(plamatrix::internal::MatrixAccess::executionInfo(sum).backend, Backend::Cpu);
            EXPECT_THROW(
                {
                    ScopedExecutionPolicy require_gpu(ExecutionPolicy::GpuRequired);
                    static_cast<void>((left * right).eval());
                },
                Error);
            return;
        }

        EXPECT_TRUE(plamatrix::internal::MatrixAccess::isDeviceResident(product));
        EXPECT_TRUE(plamatrix::internal::MatrixAccess::isDeviceResident(sum));
        EXPECT_TRUE(plamatrix::internal::MatrixAccess::executionInfo(sum).reusedDeviceInput);
        EXPECT_EQ(plamatrix::internal::MatrixAccess::executionInfo(sum).bytesUploaded, 0U);
        {
            ScopedExecutionPolicy cpu_only(ExecutionPolicy::CpuOnly);
            MatrixXd cpu_result = product + right;
            EXPECT_EQ(plamatrix::internal::MatrixAccess::executionInfo(cpu_result).backend, Backend::Cpu);
            EXPECT_EQ(plamatrix::internal::MatrixAccess::executionInfo(cpu_result).bytesDownloaded,
                      4U * sizeof(double));
        }
        const MatrixXd& const_sum = sum;
        EXPECT_DOUBLE_EQ(const_sum(0, 0), 2.0);
        EXPECT_TRUE(plamatrix::internal::MatrixAccess::isDeviceResident(sum));
        EXPECT_EQ(plamatrix::internal::MatrixAccess::executionInfo(sum).bytesDownloaded, 4U * sizeof(double));
        sum(0, 0) = 9.0;
        EXPECT_FALSE(plamatrix::internal::MatrixAccess::isDeviceResident(sum));
    }

} // namespace plamatrix::internal
