add_executable(plamatrix_tests
    unit/core/allocator_test.cpp
    unit/core/device_matrix_test.cpp
    unit/dense/dense_matrix_test.cpp
)
target_link_libraries(plamatrix_tests PRIVATE plamatrix GTest::gtest GTest::gtest_main)
target_include_directories(plamatrix_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
