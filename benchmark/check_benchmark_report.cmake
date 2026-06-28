if(NOT DEFINED REPORT_PATH OR REPORT_PATH STREQUAL "")
    message(FATAL_ERROR "REPORT_PATH is required")
endif()

if(NOT DEFINED GPU_BACKEND OR GPU_BACKEND STREQUAL "")
    message(FATAL_ERROR "GPU_BACKEND is required")
endif()

string(REGEX REPLACE "^\"|\"$" "" REPORT_PATH "${REPORT_PATH}")
string(REGEX REPLACE "^\"|\"$" "" GPU_BACKEND "${GPU_BACKEND}")

if(NOT EXISTS "${REPORT_PATH}")
    message(FATAL_ERROR "Benchmark report does not exist: ${REPORT_PATH}")
endif()

file(READ "${REPORT_PATH}" report_contents)

set(expected_backend "| GPU Backend | ${GPU_BACKEND} |")
string(FIND "${report_contents}" "${expected_backend}" backend_pos)
if(backend_pos LESS 0)
    message(FATAL_ERROR "Benchmark report does not contain expected backend row: ${expected_backend}")
endif()

set(expected_gpu_column "GPU ${GPU_BACKEND} (ms)")
string(FIND "${report_contents}" "${expected_gpu_column}" gpu_column_pos)
if(gpu_column_pos LESS 0)
    message(FATAL_ERROR "Benchmark report does not contain expected GPU column: ${expected_gpu_column}")
endif()

string(FIND "${report_contents}" "CUDA (ms)" cuda_column_pos)
if(NOT GPU_BACKEND STREQUAL "cuda" AND NOT cuda_column_pos LESS 0)
    message(FATAL_ERROR "Non-CUDA benchmark report should not contain a CUDA timing column")
endif()
