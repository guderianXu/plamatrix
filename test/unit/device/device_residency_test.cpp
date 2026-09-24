#include <gtest/gtest.h>

#include "plamatrix/internal/core/error.h"
#include "plamatrix/internal/core/execution_context.h"
#include "plamatrix/internal/device/device_csr_matrix.h"
#include "plamatrix/internal/device/device_matrix.h"
#include "plamatrix/internal/device/device_vector.h"
#include "plamatrix/internal/sparse/csr_storage.h"
#include "plamatrix/internal/sparse/iterative_solver.h"

#ifdef PLAMATRIX_WITH_OPENCL
#include "plamatrix/internal/opencl/native.h"
#include "plamatrix/internal/opencl/runtime.h"
#endif

#ifdef PLAMATRIX_WITH_VULKAN
#include "plamatrix/internal/vulkan/native.h"
#endif

#ifdef PLAMATRIX_WITH_CUDA
#include <cuda_runtime_api.h>
#include "plamatrix/internal/cuda/native.h"
#endif

namespace plamatrix::internal
{

    namespace
    {
        void verifyOwnedContextLifetime(Backend backend)
        {
            std::weak_ptr<ExecutionContext> first_owner;
            std::weak_ptr<ExecutionContext> second_owner;
            {
                auto result = [&]()
                {
                    auto context = ExecutionContext::createShared({backend, 0});
                    first_owner = context;
                    auto source = ResidentMatrix<float>::copyFrom(Matrix3f::Identity(), context);
                    return source.clone();
                }();
                EXPECT_FALSE(first_owner.expired());
                EXPECT_TRUE(result.toHostMatrix().isApprox(Matrix3f::Identity()));
                auto next = [&]()
                {
                    auto context = ExecutionContext::createShared({backend, 0});
                    second_owner = context;
                    return ResidentMatrix<float>::copyFrom(Matrix3f::Ones(), context);
                }();
                result = std::move(next);
                EXPECT_TRUE(first_owner.expired());
                EXPECT_FALSE(second_owner.expired());
                EXPECT_TRUE(result.toHostMatrix().isApprox(Matrix3f::Ones()));
                auto moved = std::move(result);
                EXPECT_TRUE(moved.toHostMatrix().isApprox(Matrix3f::Ones()));
            }
            EXPECT_TRUE(second_owner.expired());
        }
    } // namespace

    TEST(ResidentMatrix, OwnedContextSurvivesProducerCloneAndCrossContextMove)
    {
        verifyOwnedContextLifetime(Backend::Cpu);
        EXPECT_THROW((ResidentMatrix<float>(1, 1, std::shared_ptr<ExecutionContext>{})), Error);
    }

    TEST(ResidentMatrix, BindsCpuContextAndShape)
    {
        auto context = ExecutionContext::create();
        ResidentMatrix<float> matrix(3, 2, context);

        EXPECT_EQ(matrix.rows(), 3);
        EXPECT_EQ(matrix.cols(), 2);
        EXPECT_EQ(matrix.memorySpace(), MemorySpace::Host);
        EXPECT_EQ(&matrix.context(), &context);
        EXPECT_NO_THROW(matrix.validateContext(context));
    }

    TEST(ResidentMatrix, RejectsDifferentContext)
    {
        auto first = ExecutionContext::create();
        auto second = ExecutionContext::create();
        ResidentMatrix<float> matrix(2, 2, first);

        EXPECT_THROW(matrix.validateContext(second), Error);
    }

    TEST(ResidentMatrix, ExplicitHostRoundTrip)
    {
        auto context = ExecutionContext::create();
        ResidentMatrix<float> matrix(2, 2, context);
        const float source[] = {1.0F, 2.0F, 3.0F, 4.0F};
        float destination[4] = {};

        matrix.copyFromHost(source, 4);
        matrix.copyToHost(destination, 4);

        EXPECT_EQ(destination[0], 1.0F);
        EXPECT_EQ(destination[3], 4.0F);
    }

    TEST(ResidentMatrix, CopiesEigenStyleHostMatrix)
    {
        auto context = ExecutionContext::create();
        Matrix3f source = Matrix3f::Identity();
        auto resident = ResidentMatrix<float>::copyFrom(source, context);
        MatrixXf round_trip = resident.toHostMatrix();

        EXPECT_EQ(round_trip.rows(), 3);
        EXPECT_EQ(round_trip.cols(), 3);
        EXPECT_FLOAT_EQ(round_trip(0, 0), 1.0F);
        EXPECT_FLOAT_EQ(round_trip(1, 0), 0.0F);
        EXPECT_FLOAT_EQ(round_trip(2, 2), 1.0F);
    }

    TEST(ResidentMatrix, SharedContextSupportsIndependentCloneAndCheckedViews)
    {
        const auto context = ExecutionContext::createShared();
        auto source = ResidentMatrix<double>::copyFrom(MatrixXd::Identity(2, 2), *context);
        auto clone = source.clone();
        EXPECT_EQ(&clone.context(), context.get());
        EXPECT_NE(source.data(), clone.data());
        clone.view<Device::CPU>()(0, 1) = 7.0;
        EXPECT_DOUBLE_EQ(source.view<Device::CPU>()(0, 1), 0.0);
        const auto& constant = clone;
        EXPECT_DOUBLE_EQ(constant.view<Device::CPU>()(0, 1), 7.0);
        EXPECT_THROW(clone.view<Device::GPU>(), Error);
        clone.copyFromResident(source);
        EXPECT_TRUE(clone.toHostMatrix().isApprox(MatrixXd::Identity(2, 2)));
        EXPECT_NO_THROW(clone.copyFromResident(clone));

        ResidentMatrix<double> wrong_shape(1, 4, *context);
        EXPECT_THROW(wrong_shape.copyFromResident(source), Error);
        const auto other_context = ExecutionContext::createShared();
        ResidentMatrix<double> wrong_context(2, 2, *other_context);
        EXPECT_THROW(wrong_context.copyFromResident(source), Error);
        ResidentMatrix<double> empty(0, 3, *context);
        const auto empty_clone = empty.clone();
        EXPECT_EQ(empty_clone.cols(), 3);
        EXPECT_EQ(empty_clone.size(), 0U);
    }

#ifdef PLAMATRIX_WITH_CUDA
    TEST(ResidentMatrix, CudaOwnedContextSurvivesProducerCloneAndCrossContextMove)
    {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        verifyOwnedContextLifetime(Backend::Cuda);
    }

    TEST(ResidentMatrix, CudaCloneOrdersAfterContextStreamWrites)
    {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        const auto context = ExecutionContext::createShared({Backend::Cuda, 0});
        auto source = ResidentMatrix<float>::copyFrom(MatrixXf::Ones(1024, 3), *context);
        ASSERT_EQ(
            cudaMemsetAsync(source.data(), 0, source.size() * sizeof(float), cuda::NativeAccess::stream(*context)),
            cudaSuccess);
        auto clone = source.clone();
        EXPECT_NE(source.data(), clone.data());
        EXPECT_EQ(clone.toHostMatrix().squaredNorm(), 0.0F);
        EXPECT_EQ(source.toHostMatrix().squaredNorm(), 0.0F);
        EXPECT_EQ(clone.view<Device::GPU>().data(), clone.data());
        EXPECT_THROW(clone.view<Device::CPU>(), Error);
        const auto& constant = clone;
        EXPECT_EQ(constant.view<Device::GPU>().rows(), 1024);
    }

    TEST(ResidentMatrix, CopiesEigenStyleHostMatrixThroughCuda)
    {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        auto context = ExecutionContext::create({Backend::Cuda, 0});
        Matrix3f source = Matrix3f::Identity();
        auto resident = ResidentMatrix<float>::copyFrom(source, context);
        auto round_trip = resident.toHostMatrix();

        EXPECT_EQ(resident.memorySpace(), MemorySpace::Cuda);
        EXPECT_FLOAT_EQ(round_trip(0, 0), 1.0F);
        EXPECT_FLOAT_EQ(round_trip(2, 2), 1.0F);
    }
#endif

    TEST(ResidentVector, ExplicitHostRoundTrip)
    {
        auto context = ExecutionContext::create();
        ResidentVector<double> vector(3, context);
        const double source[] = {2.0, 4.0, 8.0};
        double destination[3] = {};

        vector.copyFromHost(source, 3);
        vector.copyToHost(destination, 3);

        EXPECT_EQ(destination[1], 4.0);
        EXPECT_EQ(destination[2], 8.0);
    }

    TEST(ResidentCsrMatrix, ExplicitHostCopyPreservesStructure)
    {
        auto context = ExecutionContext::create();
        CsrStorage<float, Device::CPU> host(2, 3, 2);
        host.values()[0] = 2.0F;
        host.values()[1] = 4.0F;
        host.colIndices()[0] = 0;
        host.colIndices()[1] = 2;
        host.rowOffsets()[0] = 0;
        host.rowOffsets()[1] = 1;
        host.rowOffsets()[2] = 2;

        const auto resident = ResidentCsrMatrix<float>::copyFrom(host, context);
        EXPECT_EQ(resident.rows(), 2);
        EXPECT_EQ(resident.cols(), 3);
        EXPECT_EQ(resident.nnz(), 2);
        EXPECT_EQ(resident.values()[1], 4.0F);
        EXPECT_EQ(resident.colIndices()[1], 2);
        EXPECT_EQ(resident.rowOffsets()[2], 2);
        EXPECT_EQ(resident.structureRevision(), 0U);
    }

    TEST(ResidentCsrMatrix, CopiesEigenStyleSparseStorageAndVectors)
    {
        auto context = ExecutionContext::create();
        SparseMatrix<double, RowMajor, Index> host(2, 3);
        const std::vector<Triplet<double, Index>> entries{{1, 2, 3.0}, {0, 1, 2.0}, {0, 1, 5.0}};
        host.setFromTriplets(entries.begin(), entries.end());
        const auto resident = ResidentCsrMatrix<double>::copyFrom(host, context);
        EXPECT_EQ(resident.nnz(), 2);
        EXPECT_EQ(resident.rowOffsets()[0], 0);
        EXPECT_EQ(resident.rowOffsets()[2], 2);
        EXPECT_EQ(resident.colIndices()[0], 1);
        EXPECT_DOUBLE_EQ(resident.values()[0], 7.0);
        EXPECT_EQ(resident.structureOptions().sortedness, CsrSortedness::Sorted);
        EXPECT_EQ(resident.structureOptions().duplicatePolicy, CsrDuplicatePolicy::Reject);

        const Vector3d vector(1.0, 2.0, 3.0);
        const auto resident_vector = ResidentVector<double>::copyFrom(vector, context);
        EXPECT_TRUE(resident_vector.toHostVector().isApprox(vector));
        EXPECT_THROW(ResidentVector<double>::copyFrom(Matrix2d::Identity(), context), Error);
        const auto empty = ResidentVector<double>::copyFrom(VectorXd(0), context);
        EXPECT_EQ(empty.toHostVector().size(), 0);
    }

#ifdef PLAMATRIX_WITH_CUDA
    TEST(ResidentCsrMatrix, EigenStyleInputsSolveThroughResidentCudaPcg)
    {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        auto context = ExecutionContext::create({Backend::Cuda, 0});
        SparseMatrix<double> host(2, 2);
        host.insert(0, 0) = 2.0;
        host.insert(1, 1) = 4.0;
        const auto resident = ResidentCsrMatrix<double>::copyFrom(host, context);
        const auto rhs = ResidentVector<double>::copyFrom(Vector2d(6.0, 8.0), context);
        auto solution = ResidentVector<double>::copyFrom(Vector2d::Zero(), context);
        SolveOptions options;
        options.relativeTolerance = 1e-12;
        options.absoluteTolerance = 0.0;
        const auto report = pcg(resident, rhs, solution, options, context);
        EXPECT_TRUE(report.converged);
        EXPECT_TRUE(solution.toHostVector().isApprox(Vector2d(3.0, 2.0), 1e-12));
    }
#endif

    TEST(ResidentCsrMatrix, StructureMutationInvalidatesAnalysis)
    {
        auto context = ExecutionContext::create();
        CsrStorage<float, Device::CPU> host(1, 1, 0);
        auto resident = ResidentCsrMatrix<float>::copyFrom(host, context);

        resident.markStructureModified();
        EXPECT_EQ(resident.structureRevision(), 1U);
        EXPECT_FALSE(resident.analysisValid());
    }

#ifdef PLAMATRIX_WITH_OPENCL

    TEST(ResidentOpenClStorage, EigenStyleDoubleInputsRespectDevicePrecision)
    {
        const auto devices = opencl::enumerateOpenClGpuDevices();
        if (devices.empty() || !devices.front().available || !devices.front().compilerAvailable)
        {
            GTEST_SKIP() << "No usable OpenCL GPU is available";
        }
        auto context = ExecutionContext::create({Backend::OpenCl, devices.front().index});
        SparseMatrix<double> host(2, 2);
        host.insert(0, 0) = 2.0;
        host.insert(1, 1) = 4.0;
        const auto matrix = ResidentCsrMatrix<double>::copyFrom(host, context);
        const auto rhs = ResidentVector<double>::copyFrom(Vector2d(6.0, 8.0), context);
        auto solution = ResidentVector<double>::copyFrom(Vector2d::Zero(), context);
        SolveOptions options;
        options.relativeTolerance = 1e-12;
        if (!context.capabilities().float64)
        {
            EXPECT_THROW(pcg(matrix, rhs, solution, options, context), Error);
            return;
        }
        const auto report = pcg(matrix, rhs, solution, options, context);
        EXPECT_TRUE(report.converged);
        EXPECT_TRUE(solution.toHostVector().isApprox(Vector2d(3.0, 2.0), 1e-12));
    }

    TEST(ResidentOpenClStorage, ExplicitTransfersUseOwnContext)
    {
        const auto devices = opencl::enumerateOpenClGpuDevices();
        if (devices.empty() || !devices.front().available || !devices.front().compilerAvailable)
        {
            GTEST_SKIP() << "No usable OpenCL GPU is available";
        }
        auto context = ExecutionContext::create({Backend::OpenCl, devices.front().index});
        ResidentVector<float> vector(3, context);
        const float source[] = {3.0F, 5.0F, 7.0F};
        float destination[3] = {};

        vector.copyFromHost(source, 3);
        vector.copyToHost(destination, 3);

        EXPECT_EQ(vector.memorySpace(), MemorySpace::OpenCl);
        EXPECT_NE(opencl::NativeAccess::buffer(vector), nullptr);
        EXPECT_THROW(static_cast<void>(vector.data()), Error);
        EXPECT_EQ(destination[0], 3.0F);
        EXPECT_EQ(destination[2], 7.0F);
        const MatrixXf host = MatrixXf::Ones(2, 2);
        auto resident_matrix = ResidentMatrix<float>::copyFrom(host, context);
        const auto round_trip = resident_matrix.toHostMatrix();
        EXPECT_FLOAT_EQ(round_trip(1, 1), 1.0F);
        const auto clone = resident_matrix.clone();
        EXPECT_TRUE(clone.toHostMatrix().isApprox(host));
        EXPECT_NE(opencl::NativeAccess::buffer(clone), opencl::NativeAccess::buffer(resident_matrix));
        EXPECT_THROW(clone.view<Device::CPU>(), Error);
        EXPECT_THROW(clone.view<Device::GPU>(), Error);
    }

    TEST(ResidentOpenClStorage, CsrCopyPreservesAllArrays)
    {
        const auto devices = opencl::enumerateOpenClGpuDevices();
        if (devices.empty() || !devices.front().available || !devices.front().compilerAvailable)
        {
            GTEST_SKIP() << "No usable OpenCL GPU is available";
        }
        auto context = ExecutionContext::create({Backend::OpenCl, devices.front().index});
        CsrStorage<float, Device::CPU> host(2, 3, 2);
        host.values()[0] = 2.0F;
        host.values()[1] = 4.0F;
        host.colIndices()[0] = 0;
        host.colIndices()[1] = 2;
        host.rowOffsets()[0] = 0;
        host.rowOffsets()[1] = 1;
        host.rowOffsets()[2] = 2;

        const auto resident = ResidentCsrMatrix<float>::copyFrom(host, context);
        float values[2] = {};
        Index columns[2] = {};
        Index offsets[3] = {};
        resident.copyToHost(values, columns, offsets);

        EXPECT_NE(opencl::NativeAccess::values(resident), nullptr);
        EXPECT_EQ(values[1], 4.0F);
        EXPECT_EQ(columns[1], 2);
        EXPECT_EQ(offsets[2], 2);
    }

    TEST(ResidentOpenClStorage, SolverRunsOnResidentDeviceStorage)
    {
        const auto devices = opencl::enumerateOpenClGpuDevices();
        if (devices.empty() || !devices.front().available || !devices.front().compilerAvailable)
        {
            GTEST_SKIP() << "No usable OpenCL GPU is available";
        }
        auto context = ExecutionContext::create({Backend::OpenCl, devices.front().index});
        CsrStorage<float, Device::CPU> host(1, 1, 1);
        host.values()[0] = 2.0F;
        host.colIndices()[0] = 0;
        host.rowOffsets()[0] = 0;
        host.rowOffsets()[1] = 1;
        const auto matrix = ResidentCsrMatrix<float>::copyFrom(host, context);
        ResidentVector<float> rhs(1, context);
        ResidentVector<float> solution(1, context);
        const float rhs_value[] = {4.0F};
        const float zero[] = {0.0F};
        rhs.copyFromHost(rhs_value, 1);
        solution.copyFromHost(zero, 1);

        SolveOptions options;
        options.maxIterations = 4;
        options.absoluteTolerance = 1.0e-6;
        options.requireConvergence = true;
        options.recordResidualHistory = true;
        const auto report = pcg(matrix, rhs, solution, options, context);
        float result = 0.0F;
        solution.copyToHost(&result, 1);

        EXPECT_TRUE(report.converged);
        EXPECT_EQ(report.iterations, 1);
        EXPECT_NEAR(result, 2.0F, 1.0e-6F);
        EXPECT_EQ(report.residualHistory.size(), 2U);
    }

    TEST(ResidentOpenClStorage, RepeatedSolveReusesUploadedCsr)
    {
        const auto devices = opencl::enumerateOpenClGpuDevices();
        if (devices.empty() || !devices.front().available || !devices.front().compilerAvailable)
        {
            GTEST_SKIP() << "No usable OpenCL GPU is available";
        }
        auto context = ExecutionContext::create({Backend::OpenCl, devices.front().index});
        CsrStorage<float, Device::CPU> host(2, 2, 4);
        host.rowOffsets()[0] = 0;
        host.rowOffsets()[1] = 2;
        host.rowOffsets()[2] = 4;
        host.colIndices()[0] = 0;
        host.colIndices()[1] = 1;
        host.colIndices()[2] = 0;
        host.colIndices()[3] = 1;
        host.values()[0] = 4.0F;
        host.values()[1] = 1.0F;
        host.values()[2] = 1.0F;
        host.values()[3] = 3.0F;
        const auto matrix = ResidentCsrMatrix<float>::copyFrom(host, context);
        ResidentVector<float> rhs(2, context);
        ResidentVector<float> solution(2, context);
        const float rhs_values[] = {1.0F, 2.0F};
        const float zero[] = {0.0F, 0.0F};
        rhs.copyFromHost(rhs_values, 2);
        host.values()[0] = 100.0F;

        SolveOptions options;
        options.maxIterations = 10;
        options.absoluteTolerance = 1.0e-6;
        options.requireConvergence = true;
        for (int solve_index = 0; solve_index < 2; ++solve_index)
        {
            solution.copyFromHost(zero, 2);
            const auto report = pcg(matrix, rhs, solution, options, context);
            float result[2] = {};
            solution.copyToHost(result, 2);
            EXPECT_TRUE(report.converged);
            EXPECT_NEAR(result[0], 1.0F / 11.0F, 1.0e-5F);
            EXPECT_NEAR(result[1], 7.0F / 11.0F, 1.0e-5F);
        }
    }

#endif

#ifdef PLAMATRIX_WITH_VULKAN

    TEST(ResidentVulkanStorage, ExplicitTransfersUseOwnContext)
    {
        const auto devices = vulkan::enumerateVulkanDevices();
        auto selected = devices.end();
        for (auto candidate = devices.begin(); candidate != devices.end(); ++candidate)
        {
            if (candidate->hasComputeQueue)
            {
                selected = candidate;
                break;
            }
        }
        if (selected == devices.end())
        {
            GTEST_SKIP() << "No Vulkan compute device is available";
        }
        auto context = ExecutionContext::create({Backend::Vulkan, selected->index});
        ResidentVector<float> vector(3, context);
        const float source[] = {3.0F, 5.0F, 7.0F};
        float destination[3] = {};

        vector.copyFromHost(source, 3);
        vector.copyToHost(destination, 3);

        EXPECT_EQ(vector.memorySpace(), MemorySpace::Vulkan);
        ASSERT_NE(vulkan::NativeAccess::buffer(vector), nullptr);
        EXPECT_NE(vulkan::NativeAccess::buffer(vector)->handle(), VK_NULL_HANDLE);
        EXPECT_THROW(static_cast<void>(vector.data()), Error);
        EXPECT_EQ(destination[0], 3.0F);
        EXPECT_EQ(destination[2], 7.0F);
        const MatrixXf host = MatrixXf::Ones(2, 2);
        auto resident_matrix = ResidentMatrix<float>::copyFrom(host, context);
        const auto round_trip = resident_matrix.toHostMatrix();
        EXPECT_FLOAT_EQ(round_trip(1, 1), 1.0F);
        const auto clone = resident_matrix.clone();
        EXPECT_TRUE(clone.toHostMatrix().isApprox(host));
        EXPECT_NE(vulkan::NativeAccess::buffer(clone)->handle(),
                  vulkan::NativeAccess::buffer(resident_matrix)->handle());
        EXPECT_THROW(clone.view<Device::CPU>(), Error);
        EXPECT_THROW(clone.view<Device::GPU>(), Error);
    }

    TEST(ResidentVulkanStorage, CsrCopyPreservesAllArrays)
    {
        const auto devices = vulkan::enumerateVulkanDevices();
        auto selected = devices.end();
        for (auto candidate = devices.begin(); candidate != devices.end(); ++candidate)
        {
            if (candidate->hasComputeQueue)
            {
                selected = candidate;
                break;
            }
        }
        if (selected == devices.end())
        {
            GTEST_SKIP() << "No Vulkan compute device is available";
        }
        auto context = ExecutionContext::create({Backend::Vulkan, selected->index});
        CsrStorage<float, Device::CPU> host(2, 3, 2);
        host.values()[0] = 2.0F;
        host.values()[1] = 4.0F;
        host.colIndices()[0] = 0;
        host.colIndices()[1] = 2;
        host.rowOffsets()[0] = 0;
        host.rowOffsets()[1] = 1;
        host.rowOffsets()[2] = 2;

        const auto resident = ResidentCsrMatrix<float>::copyFrom(host, context);
        float values[2] = {};
        Index columns[2] = {};
        Index offsets[3] = {};
        resident.copyToHost(values, columns, offsets);

        ASSERT_NE(vulkan::NativeAccess::values(resident), nullptr);
        EXPECT_NE(vulkan::NativeAccess::values(resident)->handle(), VK_NULL_HANDLE);
        EXPECT_EQ(values[1], 4.0F);
        EXPECT_EQ(columns[1], 2);
        EXPECT_EQ(offsets[2], 2);
    }

    TEST(ResidentVulkanStorage, RepeatedPcgUsesUploadedCsr)
    {
        const auto devices = vulkan::enumerateVulkanDevices();
        auto selected = devices.end();
        for (auto candidate = devices.begin(); candidate != devices.end(); ++candidate)
        {
            if (candidate->hasComputeQueue)
            {
                selected = candidate;
                break;
            }
        }
        if (selected == devices.end())
        {
            GTEST_SKIP() << "No Vulkan compute device is available";
        }
        auto context = ExecutionContext::create({Backend::Vulkan, selected->index});
        CsrStorage<float, Device::CPU> host(2, 2, 4);
        host.rowOffsets()[0] = 0;
        host.rowOffsets()[1] = 2;
        host.rowOffsets()[2] = 4;
        host.colIndices()[0] = 0;
        host.colIndices()[1] = 1;
        host.colIndices()[2] = 0;
        host.colIndices()[3] = 1;
        host.values()[0] = 4.0F;
        host.values()[1] = 1.0F;
        host.values()[2] = 1.0F;
        host.values()[3] = 3.0F;
        const auto matrix = ResidentCsrMatrix<float>::copyFrom(host, context);
        ResidentVector<float> rhs(2, context);
        ResidentVector<float> solution(2, context);
        const float rhs_values[] = {1.0F, 2.0F};
        const float zero[] = {0.0F, 0.0F};
        rhs.copyFromHost(rhs_values, 2);
        host.values()[0] = 100.0F;

        SolveOptions options;
        options.maxIterations = 10;
        options.absoluteTolerance = 1.0e-5;
        options.requireConvergence = true;
        options.recordResidualHistory = true;
        for (int solve_index = 0; solve_index < 2; ++solve_index)
        {
            solution.copyFromHost(zero, 2);
            const auto report = pcg(matrix, rhs, solution, options, context);
            float result[2] = {};
            solution.copyToHost(result, 2);
            EXPECT_TRUE(report.converged);
            EXPECT_EQ(report.iterations, 2);
            EXPECT_NEAR(result[0], 1.0F / 11.0F, 1.0e-5F);
            EXPECT_NEAR(result[1], 7.0F / 11.0F, 1.0e-5F);
            EXPECT_EQ(report.residualHistory.size(), 3U);
        }
    }

#endif

#ifdef PLAMATRIX_WITH_CUDA

    TEST(ResidentCudaStorage, ExplicitTransfersUseDeviceMemory)
    {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        {
            GTEST_SKIP() << "No CUDA device is available";
        }

        auto context = ExecutionContext::create({Backend::Cuda, 0});
        ResidentMatrix<float> matrix(2, 2, context);
        const float source[] = {5.0F, 6.0F, 7.0F, 8.0F};
        float destination[4] = {};

        matrix.copyFromHost(source, 4);
        matrix.copyToHost(destination, 4);

        EXPECT_EQ(matrix.memorySpace(), MemorySpace::Cuda);
        EXPECT_EQ(destination[0], 5.0F);
        EXPECT_EQ(destination[3], 8.0F);
    }

    TEST(ResidentCudaStorage, CsrCopyPreservesAllArrays)
    {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        {
            GTEST_SKIP() << "No CUDA device is available";
        }

        auto context = ExecutionContext::create({Backend::Cuda, 0});
        CsrStorage<float, Device::CPU> host(2, 3, 2);
        host.values()[0] = 2.0F;
        host.values()[1] = 4.0F;
        host.colIndices()[0] = 0;
        host.colIndices()[1] = 2;
        host.rowOffsets()[0] = 0;
        host.rowOffsets()[1] = 1;
        host.rowOffsets()[2] = 2;

        const auto resident = ResidentCsrMatrix<float>::copyFrom(host, context);
        float values[2] = {};
        Index columns[2] = {};
        Index offsets[3] = {};
        resident.copyToHost(values, columns, offsets);

        EXPECT_EQ(values[1], 4.0F);
        EXPECT_EQ(columns[1], 2);
        EXPECT_EQ(offsets[2], 2);
    }

    TEST(ResidentCudaStorage, SolverRunsOnResidentDeviceStorage)
    {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        {
            GTEST_SKIP() << "No CUDA device is available";
        }

        auto context = ExecutionContext::create({Backend::Cuda, 0});
        CsrStorage<float, Device::CPU> host(1, 1, 1);
        host.values()[0] = 1.0F;
        host.colIndices()[0] = 0;
        host.rowOffsets()[0] = 0;
        host.rowOffsets()[1] = 1;
        auto matrix = ResidentCsrMatrix<float>::copyFrom(host, context);
        ResidentVector<float> rhs(1, context);
        ResidentVector<float> solution(1, context);
        const float rhs_value[] = {1.0F};
        const float initial_solution[] = {0.0F};
        rhs.copyFromHost(rhs_value, 1);
        solution.copyFromHost(initial_solution, 1);

        SolveOptions options;
        options.maxIterations = 4;
        options.absoluteTolerance = 1.0e-6;
        options.requireConvergence = true;
        const auto report = pcg(matrix, rhs, solution, options, context);
        float result = 0.0F;
        solution.copyToHost(&result, 1);

        EXPECT_TRUE(report.converged);
        EXPECT_EQ(report.iterations, 1);
        EXPECT_NEAR(result, 1.0F, 1.0e-6F);
    }

#endif

} // namespace plamatrix::internal
