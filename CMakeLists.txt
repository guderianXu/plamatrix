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

set(_PLAMATRIX_BUILD_TESTS_USER_DEFINED OFF)
if(DEFINED PLAMATRIX_BUILD_TESTS)
    set(_PLAMATRIX_BUILD_TESTS_USER_DEFINED ON)
endif()
option(PLAMATRIX_BUILD_TESTS "Build PlaMatrix unit tests" OFF)
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR
        AND DEFINED BUILD_TESTS
        AND NOT _PLAMATRIX_BUILD_TESTS_USER_DEFINED)
    set(PLAMATRIX_BUILD_TESTS ${BUILD_TESTS} CACHE BOOL "Build PlaMatrix unit tests" FORCE)
endif()

set(_PLAMATRIX_BUILD_BENCHMARKS_USER_DEFINED OFF)
if(DEFINED PLAMATRIX_BUILD_BENCHMARKS)
    set(_PLAMATRIX_BUILD_BENCHMARKS_USER_DEFINED ON)
endif()
option(PLAMATRIX_BUILD_BENCHMARKS "Build PlaMatrix benchmark suite" OFF)
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    option(BUILD_BENCHMARKS "Build benchmark suite" OFF)
    if(DEFINED BUILD_BENCHMARKS AND NOT _PLAMATRIX_BUILD_BENCHMARKS_USER_DEFINED)
        set(PLAMATRIX_BUILD_BENCHMARKS ${BUILD_BENCHMARKS}
            CACHE BOOL "Build PlaMatrix benchmark suite" FORCE)
    endif()
endif()

set(_PLAMATRIX_BUILD_DOCS_USER_DEFINED OFF)
if(DEFINED PLAMATRIX_BUILD_DOCS)
    set(_PLAMATRIX_BUILD_DOCS_USER_DEFINED ON)
endif()
option(PLAMATRIX_BUILD_DOCS "Build PlaMatrix documentation" OFF)
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    option(BUILD_DOCS "Build documentation" OFF)
    if(DEFINED BUILD_DOCS AND NOT _PLAMATRIX_BUILD_DOCS_USER_DEFINED)
        set(PLAMATRIX_BUILD_DOCS ${BUILD_DOCS}
            CACHE BOOL "Build PlaMatrix documentation" FORCE)
    endif()
endif()

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
    if(NOT DEFINED PLAMATRIX_CUDA_ARCHITECTURES)
        if(DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_CUDA_ARCHITECTURES)
            set(PLAMATRIX_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
                CACHE STRING "CUDA architectures for PlaMatrix")
        else()
            set(PLAMATRIX_CUDA_ARCHITECTURES "75;86;89"
                CACHE STRING "CUDA architectures for PlaMatrix")
        endif()
    endif()
endif()

# ---- OpenMP (always required) ----
find_package(OpenMP REQUIRED)

# ---- Library target ----
add_library(plamatrix)
add_library(plamatrix::plamatrix ALIAS plamatrix)

if(PLAMATRIX_WITH_CUDA)
    set_target_properties(plamatrix PROPERTIES
        CUDA_ARCHITECTURES "${PLAMATRIX_CUDA_ARCHITECTURES}"
    )
endif()

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
if(PLAMATRIX_BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    add_subdirectory(test)
endif()

# ---- Benchmarks ----
if(PLAMATRIX_BUILD_BENCHMARKS)
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
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/plamatrixConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/plamatrixConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/plamatrixConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/plamatrix
)

# ---- Status ----
message(STATUS "")
message(STATUS "PlaMatrix Configuration:")
message(STATUS "  CUDA support:    ${PLAMATRIX_WITH_CUDA}")
if(PLAMATRIX_WITH_CUDA)
    message(STATUS "  CUDA arch:       ${PLAMATRIX_CUDA_ARCHITECTURES}")
endif()
message(STATUS "  Float32 support: ${PLAMATRIX_USE_FLOAT}")
message(STATUS "  Float64 support: ${PLAMATRIX_USE_DOUBLE}")
message(STATUS "  Build tests:     ${PLAMATRIX_BUILD_TESTS}")
message(STATUS "  Build benchmarks:${PLAMATRIX_BUILD_BENCHMARKS}")
message(STATUS "  Build docs:      ${PLAMATRIX_BUILD_DOCS}")
message(STATUS "")
