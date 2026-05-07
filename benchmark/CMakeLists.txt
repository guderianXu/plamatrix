if(PLAMATRIX_WITH_CUDA)
    add_executable(plamatrix_benchmark
        main.cpp
        benchmark_cases.cpp
        benchmark_cases.cu
        report_writer.cpp
    )
    target_include_directories(plamatrix_benchmark PRIVATE ${PROJECT_SOURCE_DIR})
    target_link_libraries(plamatrix_benchmark PRIVATE plamatrix)
else()
    add_executable(plamatrix_benchmark
        main.cpp
        benchmark_cases.cpp
        report_writer.cpp
    )
    target_include_directories(plamatrix_benchmark PRIVATE ${PROJECT_SOURCE_DIR})
    target_link_libraries(plamatrix_benchmark PRIVATE plamatrix)
    target_compile_definitions(plamatrix_benchmark PRIVATE PLAMATRIX_NO_CUDA=1)
endif()
