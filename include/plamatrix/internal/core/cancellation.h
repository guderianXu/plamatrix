#pragma once

#include <atomic>

namespace plamatrix::internal
{

inline namespace v1
{

class CancellationToken
{
public:
    void cancel() noexcept
    {
        _cancelled.store(true, std::memory_order_release);
    }

    bool isCancellationRequested() const noexcept
    {
        return _cancelled.load(std::memory_order_acquire);
    }

private:
    std::atomic<bool> _cancelled{false};
};

} // namespace v1

} // namespace plamatrix::internal
