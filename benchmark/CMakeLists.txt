if(PLAMATRIX_WITH_CUDA)
    add_executable(plamatrix_benchmark
        main.cpp
        benchmark_cases.cpp
        benchmark_cases.cu
        report_writer.cpp
    )
    target_include_directories(plamatrix_benchmark PRIVATE ${PROJECT_SOURCE_DIR})
    target_link_libraries(plamatrix_benchmark PRIVATE plamatrix)
elseif(PLAMATRIX_WITH_METAL_RESOLVED)
    add_executable(plamatrix_benchmark
        main.cpp
        benchmark_cases.cpp
        benchmark_cases_metal.mm
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
endif()

if(PLAMATRIX_BUILD_TESTS AND NOT PLAMATRIX_NO_GPU)
    string(TOLOWER "${PLAMATRIX_GPU_BACKEND_RESOLVED}" _plamatrix_benchmark_gpu_backend)
    set(_plamatrix_benchmark_smoke_report "${CMAKE_CURRENT_BINARY_DIR}/benchmark_gpu_smoke.md")

    add_test(
        NAME plamatrix.benchmark_gpu_smoke
        COMMAND plamatrix_benchmark
                --mode gpu
                --size smoke
                --case transpose
                --output "${_plamatrix_benchmark_smoke_report}"
    )

    add_test(
        NAME plamatrix.benchmark_gpu_report
        COMMAND ${CMAKE_COMMAND}
                -DREPORT_PATH="${_plamatrix_benchmark_smoke_report}"
                -DGPU_BACKEND="${_plamatrix_benchmark_gpu_backend}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/check_benchmark_report.cmake"
    )
    set_tests_properties(plamatrix.benchmark_gpu_report
        PROPERTIES DEPENDS plamatrix.benchmark_gpu_smoke
    )
endif()
