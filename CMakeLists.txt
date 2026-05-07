cmake_minimum_required(VERSION 3.18)

# ---- Project definition (CXX always; CUDA added conditionally) ----
project(plamatrix
    VERSION 0.1.0
    DESCRIPTION "High-performance matrix library for point cloud processing"
    LANGUAGES CXX
)

# ---- C++ standard ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Options ----
option(PLAMATRIX_USE_FLOAT  "Pre-compile float32 instantiations" ON)
option(PLAMATRIX_USE_DOUBLE "Pre-compile float64 instantiations" ON)
option(BUILD_TESTS          "Build unit tests" OFF)
option(BUILD_BENCHMARKS     "Build benchmark suite" OFF)
option(BUILD_DOCS           "Build documentation" OFF)

# ---- Detect CUDA ----
find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND)
    set(PLAMATRIX_CUDA_DEFAULT ON)
else()
    set(PLAMATRIX_CUDA_DEFAULT OFF)
endif()

option(PLAMATRIX_WITH_CUDA "Build with CUDA GPU acceleration" ${PLAMATRIX_CUDA_DEFAULT})

if(PLAMATRIX_WITH_CUDA)
    if(NOT CUDAToolkit_FOUND)
        message(FATAL_ERROR "PLAMATRIX_WITH_CUDA=ON but CUDA Toolkit not found. "
                            "Install CUDA Toolkit or set -DPLAMATRIX_WITH_CUDA=OFF for CPU-only build.")
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(PLAMATRIX_CUDA_ARCH "native" CACHE STRING "CUDA architecture target")
    set(CMAKE_CUDA_ARCHITECTURES ${PLAMATRIX_CUDA_ARCH})
endif()

# ---- OpenMP (always required) ----
find_package(OpenMP REQUIRED)

# ---- Library target ----
add_library(plamatrix)
add_library(plamatrix::plamatrix ALIAS plamatrix)

target_include_directories(plamatrix
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# ---- Link libraries ----
if(PLAMATRIX_WITH_CUDA)
    target_link_libraries(plamatrix
        PUBLIC
            CUDA::cublas
            CUDA::cusolver
            CUDA::cudart
            CUDA::cuda_driver
            OpenMP::OpenMP_CXX
    )
    target_compile_definitions(plamatrix PUBLIC PLAMATRIX_WITH_CUDA=1)
else()
    target_link_libraries(plamatrix
        PUBLIC
            OpenMP::OpenMP_CXX
    )
    target_compile_definitions(plamatrix PUBLIC PLAMATRIX_NO_CUDA=1)
endif()

target_compile_definitions(plamatrix
    PUBLIC
        $<$<BOOL:${PLAMATRIX_USE_FLOAT}>:PLAMATRIX_USE_FLOAT=1>
        $<$<BOOL:${PLAMATRIX_USE_DOUBLE}>:PLAMATRIX_USE_DOUBLE=1>
)

# ---- Source files ----
add_subdirectory(src)

# ---- Tests ----
if(BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    add_subdirectory(test)
endif()

# ---- Benchmarks ----
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()

# ---- Install ----
include(GNUInstallDirs)
install(TARGETS plamatrix
    EXPORT plamatrixTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/plamatrix
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(EXPORT plamatrixTargets
    NAMESPACE plamatrix::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/plamatrix
)
include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/plamatrixConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/plamatrixConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/plamatrix
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/plamatrixConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/plamatrix
)

# ---- Status ----
message(STATUS "")
message(STATUS "PlaMatrix Configuration:")
message(STATUS "  CUDA support:    ${PLAMATRIX_WITH_CUDA}")
if(PLAMATRIX_WITH_CUDA)
    message(STATUS "  CUDA arch:       ${PLAMATRIX_CUDA_ARCH}")
endif()
message(STATUS "  Float32 support: ${PLAMATRIX_USE_FLOAT}")
message(STATUS "  Float64 support: ${PLAMATRIX_USE_DOUBLE}")
message(STATUS "  Build tests:     ${BUILD_TESTS}")
message(STATUS "  Build benchmarks:${BUILD_BENCHMARKS}")
message(STATUS "")
