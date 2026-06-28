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
option(PLAMATRIX_WITH_SYSTEM_LINALG "Use system BLAS/LAPACK acceleration when available" ON)

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

# ---- GPU backend selection ----
set(PLAMATRIX_GPU_BACKEND "AUTO" CACHE STRING "GPU backend: AUTO, CUDA, METAL, or NONE")
set_property(CACHE PLAMATRIX_GPU_BACKEND PROPERTY STRINGS AUTO CUDA METAL NONE)
string(TOUPPER "${PLAMATRIX_GPU_BACKEND}" PLAMATRIX_GPU_BACKEND)

set(_PLAMATRIX_LEGACY_CUDA_USER_DEFINED OFF)
if(DEFINED PLAMATRIX_WITH_CUDA)
    set(_PLAMATRIX_LEGACY_CUDA_USER_DEFINED ON)
endif()
option(PLAMATRIX_WITH_CUDA "Build with CUDA GPU acceleration" OFF)
option(PLAMATRIX_WITH_METAL "Build with macOS Metal GPU acceleration" ON)

find_package(CUDAToolkit QUIET)

if(_PLAMATRIX_LEGACY_CUDA_USER_DEFINED)
    if(PLAMATRIX_WITH_CUDA)
        set(PLAMATRIX_GPU_BACKEND "CUDA" CACHE STRING "GPU backend: AUTO, CUDA, METAL, or NONE" FORCE)
    elseif(PLAMATRIX_GPU_BACKEND STREQUAL "AUTO")
        set(_PLAMATRIX_DISABLE_AUTO_CUDA ON)
    endif()
endif()

set(PLAMATRIX_WITH_CUDA OFF)
set(PLAMATRIX_WITH_METAL_RESOLVED OFF)
set(PLAMATRIX_NO_GPU OFF)

if(PLAMATRIX_GPU_BACKEND STREQUAL "AUTO")
    if(APPLE AND PLAMATRIX_WITH_METAL)
        set(PLAMATRIX_WITH_METAL_RESOLVED ON)
        set(PLAMATRIX_GPU_BACKEND_RESOLVED "METAL")
    elseif(CUDAToolkit_FOUND AND NOT _PLAMATRIX_DISABLE_AUTO_CUDA)
        set(PLAMATRIX_WITH_CUDA ON)
        set(PLAMATRIX_GPU_BACKEND_RESOLVED "CUDA")
    else()
        set(PLAMATRIX_NO_GPU ON)
        set(PLAMATRIX_GPU_BACKEND_RESOLVED "NONE")
    endif()
elseif(PLAMATRIX_GPU_BACKEND STREQUAL "CUDA")
    if(NOT CUDAToolkit_FOUND)
        message(FATAL_ERROR "PLAMATRIX_GPU_BACKEND=CUDA but CUDA Toolkit was not found.")
    endif()
    set(PLAMATRIX_WITH_CUDA ON)
    set(PLAMATRIX_GPU_BACKEND_RESOLVED "CUDA")
elseif(PLAMATRIX_GPU_BACKEND STREQUAL "METAL")
    if(NOT APPLE)
        message(FATAL_ERROR "PLAMATRIX_GPU_BACKEND=METAL is only supported on macOS.")
    endif()
    set(PLAMATRIX_WITH_METAL_RESOLVED ON)
    set(PLAMATRIX_GPU_BACKEND_RESOLVED "METAL")
elseif(PLAMATRIX_GPU_BACKEND STREQUAL "NONE")
    set(PLAMATRIX_NO_GPU ON)
    set(PLAMATRIX_GPU_BACKEND_RESOLVED "NONE")
else()
    message(FATAL_ERROR "Unknown PLAMATRIX_GPU_BACKEND=${PLAMATRIX_GPU_BACKEND}. Use AUTO, CUDA, METAL, or NONE.")
endif()

if(PLAMATRIX_WITH_CUDA)
    set(_PLAMATRIX_CMAKE_CUDA_ARCHITECTURES_USER_DEFINED OFF)
    if(DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
        set(_PLAMATRIX_CMAKE_CUDA_ARCHITECTURES_USER_DEFINED ON)
        set(_PLAMATRIX_REQUESTED_CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    endif()

    if(NOT CUDAToolkit_FOUND)
        message(FATAL_ERROR "PLAMATRIX_WITH_CUDA=ON but CUDA Toolkit not found. "
                            "Install CUDA Toolkit or set -DPLAMATRIX_WITH_CUDA=OFF for CPU-only build.")
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT DEFINED PLAMATRIX_CUDA_ARCHITECTURES)
        if(_PLAMATRIX_CMAKE_CUDA_ARCHITECTURES_USER_DEFINED)
            set(PLAMATRIX_CUDA_ARCHITECTURES "${_PLAMATRIX_REQUESTED_CMAKE_CUDA_ARCHITECTURES}"
                CACHE STRING "CUDA architectures for PlaMatrix")
        else()
            set(PLAMATRIX_CUDA_ARCHITECTURES "75;86;89"
                CACHE STRING "CUDA architectures for PlaMatrix")
        endif()
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "${PLAMATRIX_CUDA_ARCHITECTURES}"
        CACHE STRING "CUDA architectures for PlaMatrix" FORCE)

    set(PLAMATRIX_CUDA_RUNTIME_DIRS "")
    if(CUDAToolkit_BIN_DIR)
        get_filename_component(_PLAMATRIX_CUDA_TOOLKIT_ROOT "${CUDAToolkit_BIN_DIR}" DIRECTORY)
        foreach(_dir
                "${_PLAMATRIX_CUDA_TOOLKIT_ROOT}/lib"
                "${_PLAMATRIX_CUDA_TOOLKIT_ROOT}/lib64")
            if(IS_DIRECTORY "${_dir}")
                list(APPEND PLAMATRIX_CUDA_RUNTIME_DIRS "${_dir}")
            endif()
        endforeach()
    endif()
    foreach(_target CUDA::cusolver CUDA::cublas CUDA::cudart)
        if(TARGET ${_target})
            foreach(_prop IMPORTED_LOCATION IMPORTED_LOCATION_RELEASE)
                get_target_property(_loc ${_target} ${_prop})
                if(_loc)
                    get_filename_component(_dir "${_loc}" DIRECTORY)
                    if(IS_DIRECTORY "${_dir}")
                        list(APPEND PLAMATRIX_CUDA_RUNTIME_DIRS "${_dir}")
                    endif()
                endif()
            endforeach()
        endif()
    endforeach()
    if(PLAMATRIX_CUDA_RUNTIME_DIRS)
        list(REMOVE_DUPLICATES PLAMATRIX_CUDA_RUNTIME_DIRS)
        list(APPEND CMAKE_BUILD_RPATH ${PLAMATRIX_CUDA_RUNTIME_DIRS})
        list(REMOVE_DUPLICATES CMAKE_BUILD_RPATH)
    endif()
endif()

if(PLAMATRIX_WITH_METAL_RESOLVED)
    enable_language(OBJCXX)
    find_library(PLAMATRIX_FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(PLAMATRIX_METAL_FRAMEWORK Metal REQUIRED)
    find_library(PLAMATRIX_MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
    find_library(PLAMATRIX_ACCELERATE_FRAMEWORK Accelerate REQUIRED)
endif()

# ---- OpenMP (optional) ----
set(PLAMATRIX_WITH_OPENMP "AUTO" CACHE STRING "OpenMP support: AUTO, ON, or OFF")
set_property(CACHE PLAMATRIX_WITH_OPENMP PROPERTY STRINGS AUTO ON OFF)
string(TOUPPER "${PLAMATRIX_WITH_OPENMP}" PLAMATRIX_WITH_OPENMP)

set(PLAMATRIX_OPENMP_ENABLED OFF)
if(PLAMATRIX_WITH_OPENMP STREQUAL "ON")
    find_package(OpenMP REQUIRED)
    set(PLAMATRIX_OPENMP_ENABLED ON)
elseif(PLAMATRIX_WITH_OPENMP STREQUAL "AUTO")
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        set(PLAMATRIX_OPENMP_ENABLED ON)
    else()
        message(STATUS "OpenMP not found; CPU fallback loops will run serially")
    endif()
elseif(NOT PLAMATRIX_WITH_OPENMP STREQUAL "OFF")
    message(FATAL_ERROR "Unknown PLAMATRIX_WITH_OPENMP=${PLAMATRIX_WITH_OPENMP}. Use AUTO, ON, or OFF.")
endif()

# ---- Optional CPU BLAS/LAPACK acceleration ----
set(PLAMATRIX_LINALG_PROVIDER "fallback")
set(PLAMATRIX_LINALG_LIBRARIES "")
set(PLAMATRIX_WITH_FOUND_SYSTEM_LINALG OFF)
if(PLAMATRIX_WITH_SYSTEM_LINALG)
    find_package(BLAS QUIET)
    find_package(LAPACK QUIET)

    if(TARGET BLAS::BLAS AND TARGET LAPACK::LAPACK)
        set(PLAMATRIX_LINALG_PROVIDER "system_blas_lapack")
        set(PLAMATRIX_WITH_FOUND_SYSTEM_LINALG ON)
        set(PLAMATRIX_LINALG_LIBRARIES
            LAPACK::LAPACK
            BLAS::BLAS
        )
    else()
        find_library(PLAMATRIX_MKL_RT_LIBRARY
            NAMES mkl_rt
            HINTS "$ENV{CONDA_PREFIX}/lib" "$ENV{HOME}/anaconda3/lib" "$ENV{HOME}/miniconda3/lib"
        )
        if(PLAMATRIX_MKL_RT_LIBRARY)
            set(PLAMATRIX_LINALG_PROVIDER "mkl_rt")
            set(PLAMATRIX_LINALG_LIBRARIES "${PLAMATRIX_MKL_RT_LIBRARY}")
        else()
            find_library(PLAMATRIX_BLAS_LIBRARY
                NAMES blas openblas libblas.so.3 libopenblas.so.0
                PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/blas
            )
            find_library(PLAMATRIX_LAPACK_LIBRARY
                NAMES lapack liblapack.so.3
                PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/lapack
            )
            if(PLAMATRIX_BLAS_LIBRARY AND PLAMATRIX_LAPACK_LIBRARY)
                set(PLAMATRIX_LINALG_PROVIDER "blas_lapack")
                set(PLAMATRIX_LINALG_LIBRARIES
                    "${PLAMATRIX_LAPACK_LIBRARY}"
                    "${PLAMATRIX_BLAS_LIBRARY}"
                )
            else()
                message(STATUS "System BLAS/LAPACK not found; using PlaMatrix fallback linalg")
            endif()
        endif()
    endif()
endif()

# ---- Library target ----
add_library(plamatrix)
add_library(plamatrix::plamatrix ALIAS plamatrix)

target_compile_features(plamatrix PUBLIC cxx_std_17)

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
if(PLAMATRIX_OPENMP_ENABLED)
    target_link_libraries(plamatrix PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(plamatrix PUBLIC PLAMATRIX_WITH_OPENMP=1)
endif()

if(PLAMATRIX_WITH_CUDA)
    target_link_libraries(plamatrix
        PUBLIC
            CUDA::cublas
            CUDA::cusolver
            CUDA::cudart
            CUDA::cuda_driver
    )
    target_compile_definitions(plamatrix PUBLIC PLAMATRIX_WITH_CUDA=1)
elseif(PLAMATRIX_WITH_METAL_RESOLVED)
    target_link_libraries(plamatrix
        PUBLIC
            ${PLAMATRIX_FOUNDATION_FRAMEWORK}
            ${PLAMATRIX_METAL_FRAMEWORK}
            ${PLAMATRIX_MPS_FRAMEWORK}
            ${PLAMATRIX_ACCELERATE_FRAMEWORK}
    )
    target_compile_definitions(plamatrix PUBLIC PLAMATRIX_WITH_METAL=1)
else()
    target_compile_definitions(plamatrix PUBLIC PLAMATRIX_NO_GPU=1 PLAMATRIX_NO_CUDA=1)
endif()

target_compile_definitions(plamatrix
    PUBLIC
        $<$<BOOL:${PLAMATRIX_USE_FLOAT}>:PLAMATRIX_USE_FLOAT=1>
        $<$<BOOL:${PLAMATRIX_USE_DOUBLE}>:PLAMATRIX_USE_DOUBLE=1>
)

if(PLAMATRIX_LINALG_LIBRARIES)
    target_link_libraries(plamatrix PUBLIC ${PLAMATRIX_LINALG_LIBRARIES})
    target_compile_definitions(plamatrix PRIVATE
        PLAMATRIX_WITH_BLAS=1
        PLAMATRIX_WITH_LAPACK=1
    )
endif()

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
message(STATUS "  GPU backend:     ${PLAMATRIX_GPU_BACKEND_RESOLVED}")
message(STATUS "  CUDA support:    ${PLAMATRIX_WITH_CUDA}")
if(PLAMATRIX_WITH_CUDA)
    message(STATUS "  CUDA arch:       ${PLAMATRIX_CUDA_ARCHITECTURES}")
endif()
message(STATUS "  Metal support:   ${PLAMATRIX_WITH_METAL_RESOLVED}")
message(STATUS "  OpenMP support:  ${PLAMATRIX_OPENMP_ENABLED}")
message(STATUS "  Float32 support: ${PLAMATRIX_USE_FLOAT}")
message(STATUS "  Float64 support: ${PLAMATRIX_USE_DOUBLE}")
message(STATUS "  CPU linalg:      ${PLAMATRIX_LINALG_PROVIDER}")
message(STATUS "  Build tests:     ${PLAMATRIX_BUILD_TESTS}")
message(STATUS "  Build benchmarks:${PLAMATRIX_BUILD_BENCHMARKS}")
message(STATUS "  Build docs:      ${PLAMATRIX_BUILD_DOCS}")
message(STATUS "")
