#include "gemm_microkernel.h"

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

namespace plamatrix::detail
{

bool cpuSupportsAvx2Fma() noexcept
{
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int registers[4]{};
    __cpuid(registers, 1);
    const bool os_xsave = (registers[2] & (1 << 27)) != 0;
    const bool avx = (registers[2] & (1 << 28)) != 0;
    const bool fma = (registers[2] & (1 << 12)) != 0;
    if (!os_xsave || !avx || !fma || (_xgetbv(0) & 0x6) != 0x6)
    {
        return false;
    }
    __cpuidex(registers, 7, 0);
    return (registers[1] & (1 << 5)) != 0;
#elif (defined(__GNUC__) || defined(__clang__)) && \
      (defined(__x86_64__) || defined(__i386__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

} // namespace plamatrix::detail
