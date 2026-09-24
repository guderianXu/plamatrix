#pragma once

#if defined(_WIN32) && defined(PLAMATRIX_BUILD_SHARED)
#if defined(PLAMATRIX_BUILDING_LIBRARY)
#define PLAMATRIX_API __declspec(dllexport)
#else
#define PLAMATRIX_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define PLAMATRIX_API __attribute__((visibility("default")))
#else
#define PLAMATRIX_API
#endif

#define PLAMATRIX_VERSION_MAJOR 1
#define PLAMATRIX_VERSION_MINOR 0
#define PLAMATRIX_VERSION_PATCH 0
