#include <type_traits>
#include <limits>

#include <gtest/gtest.h>

#include "plamatrix/internal/core/backend.h"
#include "plamatrix/internal/core/error.h"
#include "plamatrix/internal/core/event.h"
#include "plamatrix/internal/core/execution_context.h"

#ifdef PLAMATRIX_WITH_OPENCL
#include "plamatrix/internal/opencl/native.h"
#include "plamatrix/internal/opencl/runtime.h"
#endif

#ifdef PLAMATRIX_WITH_CUDA
#include "plamatrix/internal/cuda/native.h"
#endif

#ifdef PLAMATRIX_WITH_VULKAN
#include "plamatrix/internal/vulkan/native.h"
#endif

namespace plamatrix::internal
{

    TEST(ExecutionContext, CreatesCpuContextByDefault)
    {
        static_assert(!std::is_move_constructible_v<ExecutionContext>);
        static_assert(!std::is_copy_constructible_v<ExecutionContext>);
        auto context = ExecutionContext::create();
        EXPECT_EQ(context.backend(), Backend::Cpu);
        EXPECT_EQ(context.device().backend, Backend::Cpu);
        EXPECT_EQ(context.device().index, 0U);
        EXPECT_NO_THROW(context.synchronize());
    }

#ifdef PLAMATRIX_WITH_CUDA
    TEST(ExecutionContext, ExposesCudaStreamToNativeImplementations)
    {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        auto context = ExecutionContext::create({Backend::Cuda, 0});
        EXPECT_NE(cuda::NativeAccess::stream(context), nullptr);
        auto cpu_context = ExecutionContext::create();
        EXPECT_THROW(static_cast<void>(cuda::NativeAccess::stream(cpu_context)), Error);
    }
#endif

    TEST(ExecutionContext, ReportsUnavailableOptionalBackend)
    {
        EXPECT_THROW(
            {
                try
                {
                    static_cast<void>(
                        ExecutionContext::create(DeviceId{Backend::Vulkan, std::numeric_limits<std::size_t>::max()}));
                }
                catch (const Error& error)
                {
                    EXPECT_EQ(error.code(), ErrorCode::BackendUnavailable);
                    EXPECT_EQ(error.backend(), Backend::Vulkan);
                    throw;
                }
            },
            Error);
    }

#ifdef PLAMATRIX_WITH_OPENCL
    TEST(ExecutionContext, ExplicitOpenClContextOwnsDeviceAndQueue)
    {
        const auto devices = opencl::enumerateOpenClGpuDevices();
        if (devices.empty())
        {
            GTEST_SKIP() << "No OpenCL GPU device is available";
        }
        const auto candidate = devices.front();
        if (!candidate.available || !candidate.compilerAvailable)
        {
            GTEST_SKIP() << "The first OpenCL GPU device is not usable";
        }
        auto first = ExecutionContext::create({Backend::OpenCl, candidate.index});
        auto second = ExecutionContext::create({Backend::OpenCl, candidate.index});
        EXPECT_EQ(first.device().index, candidate.index);
        EXPECT_EQ(first.backend(), Backend::OpenCl);
        EXPECT_TRUE(first.capabilities().deviceMemory);
        EXPECT_NE(opencl::NativeAccess::context(first), nullptr);
        EXPECT_NE(opencl::NativeAccess::queue(first), nullptr);
        EXPECT_NE(opencl::NativeAccess::context(first), opencl::NativeAccess::context(second));
        EXPECT_NO_THROW(first.synchronize());
    }
#endif

#ifdef PLAMATRIX_WITH_VULKAN
    TEST(ExecutionContext, ExplicitVulkanContextOwnsSelectedDevice)
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
        auto first = ExecutionContext::create({Backend::Vulkan, selected->index});
        auto second = ExecutionContext::create({Backend::Vulkan, selected->index});
        EXPECT_EQ(first.device().index, selected->index);
        EXPECT_TRUE(first.capabilities().deviceMemory);
        EXPECT_EQ(vulkan::NativeAccess::runtime(first).deviceName(), selected->name);
        EXPECT_NE(vulkan::NativeAccess::runtime(first).device(), vulkan::NativeAccess::runtime(second).device());
        EXPECT_NO_THROW(first.synchronize());
    }
#endif

    TEST(ExecutionContext, RejectsNonZeroCpuDeviceIndex)
    {
        EXPECT_THROW(static_cast<void>(ExecutionContext::create(DeviceId{Backend::Cpu, 1})), Error);
    }

    TEST(Event, CompletedEventIsMoveOnlyAndWaitIsIdempotent)
    {
        static_assert(std::is_move_constructible_v<Event>);
        static_assert(!std::is_copy_constructible_v<Event>);

        auto event = Event::completed();
        EXPECT_TRUE(event.valid());
        EXPECT_TRUE(event.ready());
        EXPECT_NO_THROW(event.wait());
        EXPECT_NO_THROW(event.wait());

        auto moved = std::move(event);
        EXPECT_FALSE(event.valid());
        EXPECT_TRUE(moved.ready());
    }

} // namespace plamatrix::internal
