#include <gtest/gtest.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#ifdef PLAMATRIX_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace {

bool hasUsableCudaRuntime()
{
#ifdef PLAMATRIX_WITH_CUDA
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount <= 0)
    {
        cudaGetLastError();
        return false;
    }

    err = cudaFree(nullptr);
    if (err != cudaSuccess)
    {
        cudaGetLastError();
        return false;
    }

    return true;
#else
    return false;
#endif
}

bool isGtestListMode(int argc, char **argv)
{
    constexpr const char *kListFlag = "--gtest_list_tests";
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);
        if (arg == kListFlag || arg.rfind(std::string(kListFlag) + "=", 0) == 0)
        {
            return true;
        }
    }
    return false;
}

bool wildcardMatch(const char *pattern, const char *text)
{
    const char *star = nullptr;
    const char *starText = nullptr;
    while (*text != '\0')
    {
        if (*pattern == '?' || *pattern == *text)
        {
            ++pattern;
            ++text;
        }
        else if (*pattern == '*')
        {
            star = pattern++;
            starText = text;
        }
        else if (star != nullptr)
        {
            pattern = star + 1;
            text = ++starText;
        }
        else
        {
            return false;
        }
    }
    while (*pattern == '*')
    {
        ++pattern;
    }
    return *pattern == '\0';
}

const std::vector<std::string> &gpuFilterPatterns()
{
    static const std::vector<std::string> kPatterns = {
        "*Gpu*",
        "*CpuGpu*",
        "*CpuVsGpu*",
        "NumPyReference.svd_CompareWithScipy",
        "NumPyReference.eigh_CompareWithScipy",
    };
    return kPatterns;
}

std::string gpuNegativeFilters()
{
    std::string filters;
    for (const std::string &pattern : gpuFilterPatterns())
    {
        if (!filters.empty())
        {
            filters += ":";
        }
        filters += pattern;
    }
    return filters;
}

bool isSingleGpuOnlyFilter(const std::string &filter)
{
    if (filter.empty() || filter == "*" || filter.find(':') != std::string::npos
        || filter.find('-') != std::string::npos)
    {
        return false;
    }

    for (const std::string &pattern : gpuFilterPatterns())
    {
        if (wildcardMatch(pattern.c_str(), filter.c_str()))
        {
            return true;
        }
    }
    return false;
}

std::string addNegativeFilters(const std::string &filter, const std::string &negativeFilters)
{
    const std::string base = filter.empty() ? "*" : filter;
    const std::size_t split = base.find('-');
    if (split == std::string::npos)
    {
        return base + ":-" + negativeFilters;
    }

    const std::string positives = base.substr(0, split);
    const std::string negatives = base.substr(split + 1);
    if (negatives.empty())
    {
        return positives + "-" + negativeFilters;
    }
    return positives + "-" + negatives + ":" + negativeFilters;
}

} // namespace

int main(int argc, char **argv)
{
    const bool listMode = isGtestListMode(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);

    if (!listMode && !hasUsableCudaRuntime())
    {
        const std::string originalFilter = ::testing::GTEST_FLAG(filter);
        if (isSingleGpuOnlyFilter(originalFilter))
        {
            std::cout << "[plamatrix_tests] CUDA runtime unavailable; skipping GPU test: "
                      << originalFilter << std::endl;
            return 77;
        }

        ::testing::GTEST_FLAG(filter) = addNegativeFilters(originalFilter, gpuNegativeFilters());
        std::cout << "[plamatrix_tests] CUDA runtime unavailable; filtering GPU tests: "
                  << ::testing::GTEST_FLAG(filter) << std::endl;
    }

    return RUN_ALL_TESTS();
}
