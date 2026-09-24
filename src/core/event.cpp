#include "plamatrix/internal/core/event.h"

#include <exception>

namespace plamatrix::internal
{

inline namespace v1
{

struct Event::State
{
    bool ready = true;
    std::exception_ptr error;
};

Event Event::completed()
{
    return Event(std::make_shared<State>());
}

bool Event::valid() const noexcept
{
    return static_cast<bool>(_state);
}

bool Event::ready() const noexcept
{
    return _state != nullptr && _state->ready;
}

void Event::wait()
{
    if (!_state)
    {
        return;
    }
    if (_state->error)
    {
        std::rethrow_exception(_state->error);
    }
}

} // namespace v1

} // namespace plamatrix::internal
