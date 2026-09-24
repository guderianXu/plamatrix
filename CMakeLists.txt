cmake_minimum_required(VERSION 3.18)

# ---- Project definition (CXX always; CUDA added conditionally) ----
project(plamatrix
    VERSION 1.0.0
    DESCRIPTION "High-performance matrix library for point cloud processing"
    LANGUAGES CXX
)

# ---- C++ standard ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Options ----
option(PLAMATRIX_USE_FLOAT  "Pre-compile float32 instantiations" ON)
option(PLAMATRIX_USE_DOUBLE "Pre-compile float64 instantiations" ON)
set(_PLAMATRIX_WITH_OPENCL_USER_DEFINED OFF)
if(DEFINED PLAMATRIX_WITH_OPENCL)
    set(_PLAMATRIX_WITH_OPENCL_USER_DEFINED ON)
endif()
option(PLAMATRIX_WITH_OPENCL "Build the OpenCL runtime and execution backend" ON)

set(_PLAMATRIX_WITH_VULKAN_USER_DEFINED OFF)
if(DEFINED PLAMATRIX_WITH_VULKAN)
    set(_PLAMATRIX_WITH_VULKAN_USER_DEFINED ON)
endif()
option(PLAMATRIX_WITH_VULKAN "Build the Vulkan Compute backend" OFF)

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

option(PLAMATRIX_WITH_EIGEN_REFERENCE
    "Build optional Eigen differential tests and comparison benchmark" OFF)
option(PLAMATRIX_FETCH_EIGEN_REFERENCE
    "Download the pinned Eigen 5 reference when it is not installed" OFF)
set(PLAMATRIX_EIGEN3_INCLUDE_DIR "" CACHE PATH
    "Eigen3 include directory used by optional reference validation")
if(PLAMATRIX_FETCH_EIGEN_REFERENCE)
    set(PLAMATRIX_WITH_EIGEN_REFERENCE ON CACHE BOOL
        "Build optional Eigen differential tests and comparison benchmark" FORCE)
endif()
if(PLAMATRIX_WITH_EIGEN_REFERENCE)
    if(PLAMATRIX_FETCH_EIGEN_REFERENCE AND TARGET Eigen3::Eigen)
        message(FATAL_ERROR
            "PLAMATRIX_FETCH_EIGEN_REFERENCE cannot replace an existing Eigen3::Eigen target. "
            "Configure PlaMatrix as a top-level project or disable fetching and provide Eigen 5.0.1.")
    endif()
    if(PLAMATRIX_FETCH_EIGEN_REFERENCE)
        include(FetchContent)
        FetchContent_Declare(plamatrix_eigen_reference_source
            URL https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz
            URL_HASH SHA256=e9c326dc8c05cd1e044c71f30f1b2e34a6161a3b6ecf445d56b53ff1669e3dec
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
        FetchContent_GetProperties(plamatrix_eigen_reference_source)
        if(NOT plamatrix_eigen_reference_source_POPULATED)
            FetchContent_Populate(plamatrix_eigen_reference_source)
        endif()
        add_library(plamatrix_eigen_reference INTERFACE)
        add_library(Eigen3::Eigen ALIAS plamatrix_eigen_reference)
        target_include_directories(plamatrix_eigen_reference SYSTEM INTERFACE
            ${plamatrix_eigen_reference_source_SOURCE_DIR})
    endif()
    if(NOT TARGET Eigen3::Eigen)
        find_package(Eigen3 5.0 CONFIG QUIET)
    endif()
    if(NOT TARGET Eigen3::Eigen)
        if(PLAMATRIX_EIGEN3_INCLUDE_DIR)
            set(_PLAMATRIX_EIGEN3_INCLUDE_DIR ${PLAMATRIX_EIGEN3_INCLUDE_DIR})
        else()
            find_path(_PLAMATRIX_EIGEN3_INCLUDE_DIR Eigen/Core PATH_SUFFIXES eigen3)
        endif()
        if(NOT EXISTS "${_PLAMATRIX_EIGEN3_INCLUDE_DIR}/Eigen/Core")
            message(FATAL_ERROR
                "PLAMATRIX_WITH_EIGEN_REFERENCE=ON requires Eigen3. Set Eigen3_DIR or "
                "PLAMATRIX_EIGEN3_INCLUDE_DIR to an Eigen include directory.")
        endif()
        add_library(plamatrix_eigen_reference INTERFACE)
        add_library(Eigen3::Eigen ALIAS plamatrix_eigen_reference)
        target_include_directories(plamatrix_eigen_reference SYSTEM INTERFACE
            ${_PLAMATRIX_EIGEN3_INCLUDE_DIR})
    endif()

    include(CheckCXXSourceCompiles)
    get_target_property(_PLAMATRIX_EIGEN_REQUIRED_INCLUDES Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
    set(_PLAMATRIX_SAVED_CMAKE_REQUIRED_INCLUDES "${CMAKE_REQUIRED_INCLUDES}")
    list(APPEND CMAKE_REQUIRED_INCLUDES ${_PLAMATRIX_EIGEN_REQUIRED_INCLUDES})
    unset(PLAMATRIX_EIGEN5_REFERENCE_API_AVAILABLE CACHE)
    check_cxx_source_compiles([=[
        #include <Eigen/Core>
        #include <Eigen/Geometry>
        #if EIGEN_MAJOR_VERSION < 5
        #error PlaMatrix differential tests require Eigen 5
        #endif
        int main()
        {
            Eigen::Matrix3d matrix = Eigen::Matrix3d::Identity();
            auto reshaped = matrix.reshaped();
            auto angles = matrix.canonicalEulerAngles(2, 1, 0);
            return static_cast<int>(reshaped.size() + angles.size());
        }
    ]=] PLAMATRIX_EIGEN5_REFERENCE_API_AVAILABLE)
    set(CMAKE_REQUIRED_INCLUDES "${_PLAMATRIX_SAVED_CMAKE_REQUIRED_INCLUDES}")
    if(NOT PLAMATRIX_EIGEN5_REFERENCE_API_AVAILABLE)
        message(FATAL_ERROR
            "PLAMATRIX_WITH_EIGEN_REFERENCE=ON requires Eigen 5.0 or newer with reshaped() and "
            "canonicalEulerAngles(). Set Eigen3_DIR or PLAMATRIX_EIGEN3_INCLUDE_DIR accordingly.")
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
set(_PLAMATRIX_WITH_CUDA_USER_DEFINED OFF)
if(DEFINED PLAMATRIX_WITH_CUDA)
    set(_PLAMATRIX_WITH_CUDA_USER_DEFINED ON)
endif()
option(PLAMATRIX_WITH_CUDA "Build with CUDA GPU acceleration" OFF)

include(CheckLanguage)
if(PLAMATRIX_WITH_CUDA OR NOT _PLAMATRIX_WITH_CUDA_USER_DEFINED)
    check_language(CUDA)
    if(NOT _PLAMATRIX_WITH_CUDA_USER_DEFINED AND CMAKE_CUDA_COMPILER)
        set(PLAMATRIX_WITH_CUDA ON CACHE BOOL "Build with CUDA GPU acceleration" FORCE)
    endif()
endif()

if(PLAMATRIX_WITH_CUDA)
    set(_PLAMATRIX_CMAKE_CUDA_ARCHITECTURES_USER_DEFINED OFF)
    if(DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
        set(_PLAMATRIX_CMAKE_CUDA_ARCHITECTURES_USER_DEFINED ON)
        set(_PLAMATRIX_REQUESTED_CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    endif()

    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "PLAMATRIX_WITH_CUDA=ON but no CUDA compiler was found. "
                            "Set CMAKE_CUDA_COMPILER to nvcc or disable CUDA.")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    string(REGEX MATCH "^[0-9]+\\.[0-9]+" _PLAMATRIX_CUDA_COMPILER_SERIES
        "${CMAKE_CUDA_COMPILER_VERSION}")
    string(REGEX MATCH "^[0-9]+\\.[0-9]+" _PLAMATRIX_CUDA_TOOLKIT_SERIES
        "${CUDAToolkit_VERSION}")
    if(_PLAMATRIX_CUDA_COMPILER_SERIES AND _PLAMATRIX_CUDA_TOOLKIT_SERIES
            AND NOT _PLAMATRIX_CUDA_COMPILER_SERIES STREQUAL _PLAMATRIX_CUDA_TOOLKIT_SERIES)
        message(FATAL_ERROR
            "CUDA compiler ${CMAKE_CUDA_COMPILER} reports ${CMAKE_CUDA_COMPILER_VERSION}, "
            "but FindCUDAToolkit selected ${CUDAToolkit_VERSION} at "
            "${CUDAToolkit_LIBRARY_ROOT}. Select one coherent CUDA installation.")
    endif()
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
    foreach(_target CUDA::cusolver CUDA::cusparse CUDA::cublas CUDA::cudart)
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

# ---- Optional OpenCL execution backend ----
if(PLAMATRIX_WITH_OPENCL)
    find_package(OpenCL 1.2 QUIET)
    if(NOT OpenCL_FOUND)
        if(_PLAMATRIX_WITH_OPENCL_USER_DEFINED)
            message(FATAL_ERROR "PLAMATRIX_WITH_OPENCL=ON requires an OpenCL 1.2 SDK/loader")
        endif()
        message(STATUS "PlaMatrix OpenCL disabled because an OpenCL 1.2 SDK/loader was not found")
        set(PLAMATRIX_WITH_OPENCL OFF CACHE BOOL
            "Build the OpenCL runtime and execution backend" FORCE)
    endif()
endif()

# ---- Optional Vulkan Compute backend ----
set(PLAMATRIX_VULKAN_COOPERATIVE_MATRIX_SHADER OFF)
if(PLAMATRIX_WITH_VULKAN)
    find_package(Vulkan 1.1 QUIET)
    find_program(PLAMATRIX_GLSLANG_VALIDATOR glslangValidator)
    if(NOT Vulkan_FOUND OR NOT PLAMATRIX_GLSLANG_VALIDATOR)
        if(_PLAMATRIX_WITH_VULKAN_USER_DEFINED)
            message(FATAL_ERROR "PLAMATRIX_WITH_VULKAN=ON requires Vulkan 1.1 headers/loader and glslangValidator")
        endif()
        message(STATUS "PlaMatrix Vulkan disabled because Vulkan 1.1 or glslangValidator was not found")
        set(PLAMATRIX_WITH_VULKAN OFF CACHE BOOL "Build the Vulkan Compute backend" FORCE)
    endif()
    if(PLAMATRIX_WITH_VULKAN)
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles")
        execute_process(
            COMMAND ${PLAMATRIX_GLSLANG_VALIDATOR} -V --target-env vulkan1.1
                "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders/batched_gemm_cooperative.comp"
                -o "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/plamatrix-cooperative-matrix-probe.spv"
            RESULT_VARIABLE _PLAMATRIX_COOPERATIVE_MATRIX_SHADER_RESULT
            OUTPUT_QUIET
            ERROR_QUIET)
        file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/plamatrix-cooperative-matrix-probe.spv")
        if(_PLAMATRIX_COOPERATIVE_MATRIX_SHADER_RESULT EQUAL 0)
            set(PLAMATRIX_VULKAN_COOPERATIVE_MATRIX_SHADER ON)
        else()
            message(STATUS "PlaMatrix Vulkan cooperative matrices disabled because glslangValidator lacks support")
        endif()
    endif()
endif()

# ---- OpenMP (always required for C++ CPU paths) ----
find_package(OpenMP REQUIRED COMPONENTS CXX)

# ---- Library target ----
add_library(plamatrix)
add_library(plamatrix::plamatrix ALIAS plamatrix)

target_compile_features(plamatrix PUBLIC cxx_std_17)

target_compile_definitions(plamatrix PRIVATE PLAMATRIX_BUILDING_LIBRARY=1)

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
            CUDA::cusparse
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

if(PLAMATRIX_WITH_OPENCL)
    target_link_libraries(plamatrix PUBLIC OpenCL::OpenCL)
    target_compile_definitions(plamatrix
        PUBLIC
            PLAMATRIX_WITH_OPENCL=1
            CL_TARGET_OPENCL_VERSION=120
    )
else()
    target_compile_definitions(plamatrix PUBLIC PLAMATRIX_NO_OPENCL=1)
endif()

if(PLAMATRIX_WITH_VULKAN)
    target_link_libraries(plamatrix PUBLIC Vulkan::Vulkan)
    target_compile_definitions(plamatrix PUBLIC PLAMATRIX_WITH_VULKAN=1)
    set(PLAMATRIX_VULKAN_SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/vulkan-shaders")
    target_compile_definitions(plamatrix PUBLIC
        PLAMATRIX_VULKAN_SHADER_DIR=\"${PLAMATRIX_VULKAN_SHADER_DIR}\")
    if(PLAMATRIX_VULKAN_COOPERATIVE_MATRIX_SHADER)
        target_compile_definitions(plamatrix PRIVATE PLAMATRIX_VULKAN_COOPERATIVE_MATRIX_SHADER=1)
    endif()
else()
    target_compile_definitions(plamatrix PUBLIC PLAMATRIX_NO_VULKAN=1)
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
message(STATUS "  OpenCL support:  ${PLAMATRIX_WITH_OPENCL}")
message(STATUS "  Vulkan support:  ${PLAMATRIX_WITH_VULKAN}")
message(STATUS "  Float32 support: ${PLAMATRIX_USE_FLOAT}")
message(STATUS "  Float64 support: ${PLAMATRIX_USE_DOUBLE}")
message(STATUS "  CPU linalg:      native")
message(STATUS "  Build tests:     ${PLAMATRIX_BUILD_TESTS}")
message(STATUS "  Build benchmarks:${PLAMATRIX_BUILD_BENCHMARKS}")
message(STATUS "  Eigen reference: ${PLAMATRIX_WITH_EIGEN_REFERENCE}")
message(STATUS "  Fetch Eigen reference: ${PLAMATRIX_FETCH_EIGEN_REFERENCE}")
message(STATUS "  Build docs:      ${PLAMATRIX_BUILD_DOCS}")
message(STATUS "")
