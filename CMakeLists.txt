cmake_minimum_required(VERSION 3.18)
project(plamatrix
    VERSION 0.1.0
    DESCRIPTION "High-performance matrix library for point cloud processing"
    LANGUAGES CXX CUDA
)

# ---- C++ standard ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---- Options ----
option(PLAMATRIX_USE_FLOAT  "Pre-compile float32 instantiations" ON)
option(PLAMATRIX_USE_DOUBLE "Pre-compile float64 instantiations" ON)
option(BUILD_TESTS          "Build unit tests" OFF)
option(BUILD_BENCHMARKS     "Build benchmark suite" OFF)
option(BUILD_DOCS           "Build documentation" OFF)

set(PLAMATRIX_CUDA_ARCH "native" CACHE STRING "CUDA architecture target")

# ---- CUDA settings ----
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES ${PLAMATRIX_CUDA_ARCH})

# ---- OpenMP ----
find_package(OpenMP REQUIRED)

# ---- Library target ----
add_library(plamatrix)
add_library(plamatrix::plamatrix ALIAS plamatrix)

target_include_directories(plamatrix
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(plamatrix
    PUBLIC
        CUDA::cublas
        CUDA::cusolver
        CUDA::cudart
        OpenMP::OpenMP_CXX
)

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
