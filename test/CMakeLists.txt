add_executable(plamatrix_tests
    test_main.cpp
    unit/core/allocator_test.cpp
    unit/core/device_matrix_test.cpp
    unit/core/parallel_test.cpp
    unit/dense/dense_matrix_test.cpp
    unit/dense/dense_ops_test.cpp
    unit/dense/elementwise_test.cpp
    unit/ops/gemm_test.cpp
    unit/ops/decomposition_test.cpp
    unit/ops/solver_test.cpp
    unit/ops/point_cloud_test.cpp
    integration/cpu_gpu_consistency_test.cpp
    integration/no_cuda_behavior_test.cpp
    integration/point_cloud_workflow_test.cpp
    integration/numpy_reference_test.cpp
)
target_link_libraries(plamatrix_tests PRIVATE plamatrix GTest::gtest)
target_include_directories(plamatrix_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(plamatrix_tests PRIVATE
    TEST_REFERENCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/reference"
)

include(GoogleTest)
gtest_discover_tests(plamatrix_tests
    TEST_PREFIX plamatrix.
    DISCOVERY_MODE PRE_TEST
    PROPERTIES SKIP_RETURN_CODE 77
)

set(PLAMATRIX_SPARSE_TEST_SOURCES
    unit/sparse/csr_matrix_test.cpp
    unit/sparse/iterative_solver_test.cpp
    unit/sparse/sparse_ops_test.cpp
)
if(PLAMATRIX_USE_FLOAT)
    list(APPEND PLAMATRIX_SPARSE_TEST_SOURCES unit/sparse/coo_matrix_test.cpp)
endif()

add_executable(plamatrix_sparse_tests ${PLAMATRIX_SPARSE_TEST_SOURCES})
target_link_libraries(plamatrix_sparse_tests PRIVATE plamatrix GTest::gtest_main)
if(PLAMATRIX_WITH_CUDA)
    target_compile_definitions(plamatrix PRIVATE PLAMATRIX_ITERATIVE_SOLVER_TEST_HOOKS=1)
    target_compile_definitions(
        plamatrix_sparse_tests PRIVATE PLAMATRIX_ITERATIVE_SOLVER_TEST_HOOKS=1)
    target_include_directories(
        plamatrix_sparse_tests PRIVATE ${PROJECT_SOURCE_DIR}/src/sparse)
endif()

gtest_discover_tests(plamatrix_sparse_tests
    TEST_PREFIX plamatrix.
    DISCOVERY_MODE PRE_TEST
)

add_dependencies(plamatrix_tests plamatrix_sparse_tests)

add_executable(plamatrix_reduction_tests
    unit/ops/reduction_test.cpp
)
target_link_libraries(plamatrix_reduction_tests PRIVATE plamatrix GTest::gtest_main)

gtest_discover_tests(plamatrix_reduction_tests
    TEST_PREFIX plamatrix.
    DISCOVERY_MODE PRE_TEST
)

add_dependencies(plamatrix_tests plamatrix_reduction_tests)

add_executable(plamatrix_indexing_tests
    unit/ops/indexing_cuda_rows_test.cpp
    unit/ops/indexing_cuda_test.cpp
    unit/ops/indexing_no_cuda_test.cpp
    unit/ops/indexing_test.cpp
    unit/ops/indexing_test_utils.h
    unit/ops/indexing_workspace_test.cpp
)
target_link_libraries(plamatrix_indexing_tests PRIVATE plamatrix GTest::gtest_main)

gtest_discover_tests(plamatrix_indexing_tests
    TEST_PREFIX plamatrix.
    DISCOVERY_MODE PRE_TEST
)

add_dependencies(plamatrix_tests plamatrix_indexing_tests)

add_executable(plamatrix_small_matrix_tests
    unit/ops/small_matrix_async_test.cpp
    unit/ops/small_matrix_cuda_test.cpp
    unit/ops/small_matrix_extreme_test.cpp
    unit/ops/small_matrix_workspace_test.cpp
    unit/ops/small_matrix_test.cpp
)
target_link_libraries(plamatrix_small_matrix_tests PRIVATE plamatrix GTest::gtest_main)
if(PLAMATRIX_WITH_CUDA)
    target_compile_definitions(plamatrix PRIVATE PLAMATRIX_SMALL_MATRIX_TEST_HOOKS=1)
    target_compile_definitions(
        plamatrix_small_matrix_tests PRIVATE PLAMATRIX_SMALL_MATRIX_TEST_HOOKS=1)
    target_include_directories(
        plamatrix_small_matrix_tests PRIVATE ${PROJECT_SOURCE_DIR}/src/ops)
endif()

gtest_discover_tests(plamatrix_small_matrix_tests
    TEST_PREFIX plamatrix.
    DISCOVERY_MODE PRE_TEST
)

add_dependencies(plamatrix_tests plamatrix_small_matrix_tests)

add_executable(plamatrix_statistics_tests
    unit/ops/statistics_test.cpp
)
target_link_libraries(plamatrix_statistics_tests PRIVATE plamatrix GTest::gtest_main)

gtest_discover_tests(plamatrix_statistics_tests
    TEST_PREFIX plamatrix.
    DISCOVERY_MODE PRE_TEST
)

add_dependencies(plamatrix_tests plamatrix_statistics_tests)

add_executable(plamatrix_vector_tests
    unit/ops/vector_test.cpp
)
target_link_libraries(plamatrix_vector_tests PRIVATE plamatrix GTest::gtest_main)

gtest_discover_tests(plamatrix_vector_tests
    TEST_PREFIX plamatrix.
    DISCOVERY_MODE PRE_TEST
)
