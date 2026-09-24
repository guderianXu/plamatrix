#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace plamatrix::internal
{

inline namespace v1
{

struct TraceEntry
{
    std::string operation;
    double milliseconds = 0.0;
};

class ExecutionTrace
{
public:
    void record(std::string operation, double milliseconds)
    {
        _entries.push_back({std::move(operation), milliseconds});
    }

    std::size_t operationCount() const noexcept
    {
        return _entries.size();
    }

    const std::vector<TraceEntry>& entries() const noexcept
    {
        return _entries;
    }

private:
    std::vector<TraceEntry> _entries;
};

struct BackendDiagnostics
{
    double gpuMilliseconds = 0.0;
    std::size_t commandSubmissions = 0;
    std::size_t barrierCount = 0;
    bool deterministic = false;
};

} // namespace v1

} // namespace plamatrix::internal
