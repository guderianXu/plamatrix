#include <gtest/gtest.h>

#include "plamatrix/opencl/runtime.h"

#include <stdexcept>

#ifdef PLAMATRIX_WITH_OPENCL
#include "plamatrix/opencl/execution.h"

#include <CL/cl.h>

#include <algorithm>
#include <cstddef>
#include <future>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#endif

namespace
{

#ifdef PLAMATRIX_WITH_OPENCL

bool hasUsableOpenClDevice()
{
    return plamatrix::opencl::hasUsableOpenClDevice();
}

int runSingleValueKernel(
    plamatrix::opencl::OpenClRuntime& runtime,
    cl_program program)
{
    plamatrix::opencl::CommandQueue queue(runtime.createQueue());
    plamatrix::opencl::DeviceBuffer output_buffer(
        runtime.context(), CL_MEM_WRITE_ONLY, sizeof(cl_int));
    plamatrix::opencl::CompiledKernel kernel(program, "cacheProbe");
    plamatrix::opencl::kernelBufferArg(kernel, 0, output_buffer);

    const std::size_t global_size = 1;
    plamatrix::opencl::checkOpenCl(
        clEnqueueNDRangeKernel(
            queue.get(), kernel.get(), 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr),
        "clEnqueueNDRangeKernel(cacheProbe)");

    std::vector<cl_int> output(1, 0);
    plamatrix::opencl::readVector(queue.get(), output_buffer, output);
    return output.front();
}

#endif

} // namespace

#ifdef PLAMATRIX_WITH_OPENCL

TEST(OpenClRuntime, EnumerationContainsSelectedDevice)
{
    if (!hasUsableOpenClDevice())
    {
        GTEST_SKIP() << "No usable OpenCL GPU with an online compiler";
    }

    const auto devices = plamatrix::opencl::enumerateOpenClGpuDevices();
    ASSERT_FALSE(devices.empty());
    EXPECT_NO_THROW(plamatrix::opencl::requireUsableOpenClDevice());

    const int selected_index = plamatrix::opencl::selectedOpenClDeviceIndex();
    ASSERT_GE(selected_index, 0);
    const auto selected = std::find_if(devices.begin(), devices.end(), [&](const auto& device)
    {
        return device.index == static_cast<std::size_t>(selected_index);
    });
    ASSERT_NE(selected, devices.end());
    EXPECT_TRUE(selected->available);
    EXPECT_TRUE(selected->compilerAvailable);
    EXPECT_EQ(plamatrix::opencl::selectedOpenClDeviceName(), selected->name);
}

TEST(OpenClRuntime, UploadDownloadAndKernelExecution)
{
    if (!hasUsableOpenClDevice())
    {
        GTEST_SKIP() << "No usable OpenCL GPU with an online compiler";
    }

    auto& runtime = plamatrix::opencl::OpenClRuntime::instance();
    plamatrix::opencl::CommandQueue queue(runtime.createQueue());
    const std::vector<cl_int> input{1, -2, 7, 21};
    std::vector<cl_int> output(input.size(), 0);
    auto input_buffer = plamatrix::opencl::inputVector(runtime, input);
    auto output_buffer = plamatrix::opencl::inOutVector(runtime, output);

    const std::string source = R"CLC(
        __kernel void addDelta(
            __global const int* input,
            __global int* output,
            const int delta)
        {
            const size_t index = get_global_id(0);
            output[index] = input[index] + delta;
        }
    )CLC";
    cl_program program = runtime.program("plamatrix_test_add_delta", source);
    plamatrix::opencl::CompiledKernel kernel(program, "addDelta");
    const cl_int delta = 5;
    plamatrix::opencl::kernelBufferArg(kernel, 0, input_buffer);
    plamatrix::opencl::kernelBufferArg(kernel, 1, output_buffer);
    plamatrix::opencl::kernelArg(kernel, 2, delta);

    const std::size_t global_size = input.size();
    plamatrix::opencl::checkOpenCl(
        clEnqueueNDRangeKernel(
            queue.get(), kernel.get(), 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr),
        "clEnqueueNDRangeKernel(addDelta)");
    plamatrix::opencl::readVector(queue.get(), output_buffer, output);

    EXPECT_EQ(output, (std::vector<cl_int>{6, 3, 12, 26}));
}

TEST(OpenClRuntime, ProgramCacheSeparatesCallerSourceAndOptions)
{
    if (!hasUsableOpenClDevice())
    {
        GTEST_SKIP() << "No usable OpenCL GPU with an online compiler";
    }

    auto& runtime = plamatrix::opencl::OpenClRuntime::instance();
    const std::string source_one = R"CLC(
        __kernel void cacheProbe(__global int* output)
        {
            output[0] = CACHE_VALUE;
        }
    )CLC";
    const std::string source_two = R"CLC(
        __kernel void cacheProbe(__global int* output)
        {
            output[0] = CACHE_VALUE + 100;
        }
    )CLC";

    cl_program first = runtime.program("shared_key", source_one, "-DCACHE_VALUE=1");
    cl_program first_again = runtime.program("shared_key", source_one, "-DCACHE_VALUE=1");
    cl_program changed_source = runtime.program("shared_key", source_two, "-DCACHE_VALUE=1");
    cl_program changed_options = runtime.program("shared_key", source_one, "-DCACHE_VALUE=2");
    cl_program changed_caller = runtime.program("other_key", source_one, "-DCACHE_VALUE=1");

    EXPECT_EQ(first, first_again);
    EXPECT_NE(first, changed_source);
    EXPECT_NE(first, changed_options);
    EXPECT_NE(first, changed_caller);
    EXPECT_EQ(runSingleValueKernel(runtime, first), 1);
    EXPECT_EQ(runSingleValueKernel(runtime, changed_source), 101);
    EXPECT_EQ(runSingleValueKernel(runtime, changed_options), 2);
    EXPECT_EQ(runSingleValueKernel(runtime, changed_caller), 1);
}

TEST(OpenClRuntime, ProgramCacheIsSharedAcrossThreads)
{
    if (!hasUsableOpenClDevice())
    {
        GTEST_SKIP() << "No usable OpenCL GPU with an online compiler";
    }

    auto& runtime = plamatrix::opencl::OpenClRuntime::instance();
    const std::string source = R"CLC(
        __kernel void threadedCacheProbe(__global int* output)
        {
            output[0] = 17;
        }
    )CLC";
    cl_program expected = runtime.program("threaded_cache", source);

    std::vector<std::future<cl_program>> futures;
    for (int index = 0; index < 8; ++index)
    {
        futures.emplace_back(std::async(std::launch::async, [&runtime, &source]()
        {
            return runtime.program("threaded_cache", source);
        }));
    }
    for (auto& future : futures)
    {
        EXPECT_EQ(future.get(), expected);
    }
}

TEST(OpenClExecution, ResourceOwnersMoveAndValidateReadBounds)
{
    if (!hasUsableOpenClDevice())
    {
        GTEST_SKIP() << "No usable OpenCL GPU with an online compiler";
    }

    auto& runtime = plamatrix::opencl::OpenClRuntime::instance();
    plamatrix::opencl::CommandQueue original_queue(runtime.createQueue());
    const cl_command_queue queue_handle = original_queue.get();
    plamatrix::opencl::CommandQueue moved_queue(std::move(original_queue));
    EXPECT_EQ(original_queue.get(), nullptr);
    EXPECT_EQ(moved_queue.get(), queue_handle);

    plamatrix::opencl::DeviceBuffer original_buffer(
        runtime.context(), CL_MEM_READ_WRITE, sizeof(cl_int));
    const cl_mem buffer_handle = original_buffer.get();
    plamatrix::opencl::DeviceBuffer moved_buffer(std::move(original_buffer));
    EXPECT_EQ(original_buffer.get(), nullptr);
    EXPECT_EQ(original_buffer.size(), 0U);
    EXPECT_EQ(moved_buffer.get(), buffer_handle);
    EXPECT_EQ(moved_buffer.size(), sizeof(cl_int));

    std::vector<cl_int> too_large(2, 0);
    EXPECT_THROW(
        plamatrix::opencl::readVector(moved_queue.get(), moved_buffer, too_large),
        std::out_of_range);
    EXPECT_EQ(plamatrix::opencl::byteSize<cl_int>(3), 3U * sizeof(cl_int));
    EXPECT_THROW(
        plamatrix::opencl::byteSize<cl_int>(std::numeric_limits<std::size_t>::max()),
        std::overflow_error);
}

TEST(OpenClExecution, BuildOptionsUseCallerMacro)
{
    EXPECT_TRUE(plamatrix::opencl::realBuildOptions<float>("PROJECT_REAL_DOUBLE").empty());
    EXPECT_EQ(
        plamatrix::opencl::realBuildOptions<double>("PROJECT_REAL_DOUBLE"),
        "-DPROJECT_REAL_DOUBLE=1");
}

#else

TEST(OpenClRuntime, DisabledBuildUsesPublicStubs)
{
    EXPECT_TRUE(plamatrix::opencl::enumerateOpenClGpuDevices().empty());
    EXPECT_FALSE(plamatrix::opencl::hasUsableOpenClDevice());
    EXPECT_THROW(plamatrix::opencl::requireUsableOpenClDevice(), std::runtime_error);
    EXPECT_TRUE(plamatrix::opencl::selectedOpenClDeviceName().empty());
    EXPECT_EQ(plamatrix::opencl::selectedOpenClDeviceIndex(), -1);
}

#endif
