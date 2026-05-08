add_executable(plamatrix_tests
    unit/core/allocator_test.cpp
    unit/core/device_matrix_test.cpp
    unit/dense/dense_matrix_test.cpp
    unit/dense/dense_ops_test.cpp
    unit/ops/gemm_test.cpp
    unit/ops/decomposition_test.cpp
    unit/ops/solver_test.cpp
    unit/ops/point_cloud_test.cpp
    unit/sparse/coo_matrix_test.cpp
    unit/sparse/csr_matrix_test.cpp
    integration/cpu_gpu_consistency_test.cpp
    integration/point_cloud_workflow_test.cpp
)
target_link_libraries(plamatrix_tests PRIVATE plamatrix GTest::gtest GTest::gtest_main)
target_include_directories(plamatrix_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
