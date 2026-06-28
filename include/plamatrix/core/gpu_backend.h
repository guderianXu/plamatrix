#pragma once

namespace plamatrix
{

enum class GpuBackend
{
    None,
    Cuda,
    Metal
};

/// Return the GPU backend selected at configure time.
/// @return Selected backend enum value.
GpuBackend gpuBackend();

/// Return the selected backend name: "cuda", "metal", or "none".
/// @return Stable string literal for the configured GPU backend.
const char* gpuBackendName();

} // namespace plamatrix
