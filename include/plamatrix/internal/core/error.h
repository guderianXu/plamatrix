#pragma once

#include <stdexcept>
#include <string>
#include <utility>

#include "plamatrix/internal/core/api.h"
#include "plamatrix/internal/core/backend.h"

namespace plamatrix::internal
{

inline namespace v1
{

enum class ErrorCode
{
    InvalidArgument,
    UnsupportedBackend,
    BackendUnavailable,
    UnsupportedOperation,
    InvalidState,
    BackendFailure,
    NumericalFailure
};

class PLAMATRIX_API Error : public std::runtime_error
{
public:
    Error(ErrorCode code, std::string message, Backend backend = Backend::Cpu, int native_code = 0)
        : std::runtime_error(std::move(message))
        , _code(code)
        , _backend(backend)
        , _nativeCode(native_code)
    {
    }

    ErrorCode code() const noexcept
    {
        return _code;
    }

    Backend backend() const noexcept
    {
        return _backend;
    }

    int nativeCode() const noexcept
    {
        return _nativeCode;
    }

private:
    ErrorCode _code;
    Backend _backend;
    int _nativeCode;
};

} // namespace v1

} // namespace plamatrix::internal
