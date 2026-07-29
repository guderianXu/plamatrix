if(PLAMATRIX_WITH_CUDA)
    add_executable(plamatrix_benchmark
        main.cpp
        benchmark_cases.cpp
        benchmark_cases.cu
        release1_cases.cpp
        release1_cases.cu
        release3_cases.cpp
        release3_cases.cu
        report_writer.cpp
    )
    target_include_directories(plamatrix_benchmark PRIVATE ${PROJECT_SOURCE_DIR})
    target_link_libraries(plamatrix_benchmark PRIVATE plamatrix)
else()
    add_executable(plamatrix_benchmark
        main.cpp
        benchmark_cases.cpp
        release1_cases.cpp
        release3_cases.cpp
        report_writer.cpp
    )
    target_include_directories(plamatrix_benchmark PRIVATE ${PROJECT_SOURCE_DIR})
    target_link_libraries(plamatrix_benchmark PRIVATE plamatrix)
    target_compile_definitions(plamatrix_benchmark PRIVATE PLAMATRIX_NO_CUDA=1)
endif()

if(PLAMATRIX_BUILD_TESTS)
    add_executable(plamatrix_benchmark_report_test
        report_writer_test.cpp
        report_writer.cpp
    )
    target_include_directories(plamatrix_benchmark_report_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_link_libraries(plamatrix_benchmark_report_test PRIVATE plamatrix GTest::gtest_main)
    add_test(NAME plamatrix_benchmark_report_schema COMMAND plamatrix_benchmark_report_test)

    foreach(case_name IN ITEMS
            elementwise reduction compact eigh3x3_batch
            coo_to_csr spmv spmm cg pcg)
        add_test(
            NAME plamatrix_benchmark_registers_${case_name}
            COMMAND plamatrix_benchmark --mode cpu --size tiny --case ${case_name}
                --output ${CMAKE_CURRENT_BINARY_DIR}/${case_name}_smoke.md
        )
        set_tests_properties(plamatrix_benchmark_registers_${case_name} PROPERTIES
            PASS_REGULAR_EXPRESSION "${case_name} cpu_serial_total_ms=[0-9]+(\\.[0-9]+)?"
        )
    endforeach()
endif()
