#pragma once

#include <memory>
#include <utility>

#include "plamatrix/internal/core/api.h"

namespace plamatrix::internal
{

inline namespace v1
{

class PLAMATRIX_API Event
{
public:
    Event() noexcept = default;
    ~Event() noexcept = default;
    Event(Event&&) noexcept = default;
    Event& operator=(Event&&) noexcept = default;
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    static Event completed();

    bool valid() const noexcept;
    bool ready() const noexcept;
    void wait();

private:
    struct State;
    explicit Event(std::shared_ptr<State> state) noexcept
        : _state(std::move(state))
    {
    }

    std::shared_ptr<State> _state;
};

} // namespace v1

} // namespace plamatrix::internal
