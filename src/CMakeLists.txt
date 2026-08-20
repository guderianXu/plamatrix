# ---- CPU sources (always compiled) ----
target_sources(plamatrix PRIVATE
    optimization/block_normal_equations.cpp
    optimization/block_schur.cpp
    optimization/block_schur_accelerated.cpp
    optimization/block_schur_device_assembly_stub.cpp
    sparse/csr_matrix.cpp
    sparse/iterative_solver_cpu.cpp
    sparse/sparse_ops_cpu.cpp
    dense/elementwise_cpu.cpp
    ops/gemm_cpu.cpp
    ops/cpu_features.cpp
    ops/decomposition_cpu.cpp
    ops/indexing_cpu.cpp
    ops/reduction_cpu.cpp
    ops/small_matrix_cpu.cpp
    ops/solver_cpu.cpp
    ops/point_cloud_cpu.cpp
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i[3-6]86)$")
    target_sources(plamatrix PRIVATE ops/gemm_microkernel_avx2.cpp)
    target_compile_definitions(plamatrix PRIVATE PLAMATRIX_HAVE_AVX2_KERNEL=1)
    if(MSVC)
        set_source_files_properties(ops/gemm_microkernel_avx2.cpp PROPERTIES
            COMPILE_FLAGS "/arch:AVX2")
    else()
        set_source_files_properties(ops/gemm_microkernel_avx2.cpp PROPERTIES
            COMPILE_FLAGS "-mavx2 -mfma")
    endif()
endif()

# ---- OpenCL runtime sources ----
if(PLAMATRIX_WITH_OPENCL)
    target_sources(plamatrix PRIVATE
        opencl/device_enumeration.cpp
        optimization/block_schur_device_assembly_opencl.cpp
        opencl/iterative_solver.cpp
        opencl/runtime.cpp
    )
else()
    target_sources(plamatrix PRIVATE
        opencl/iterative_solver_stub.cpp
        opencl/runtime_stub.cpp
    )
endif()

# ---- GPU / CUDA sources (only when CUDA is enabled) ----
if(PLAMATRIX_WITH_CUDA)
    target_sources(plamatrix PRIVATE
        plamatrix.cu
        optimization/block_schur_device_assembly.cu
        sparse/coo_to_csr.cu
        sparse/iterative_solver.cu
        sparse/iterative_solver_workspace.cu
        sparse/sparse_ops.cu
        dense/dense_matrix.cu
        dense/dense_ops.cu
        dense/elementwise.cu
        ops/gemm.cu
        ops/decomposition.cu
        ops/indexing.cu
        ops/indexing_dispatch.cu
        ops/indexing_workspace.cu
        ops/reduction.cu
        ops/reduction_dispatch.cu
        ops/reduction_workspace.cu
        ops/small_matrix.cu
        ops/small_matrix_dispatch.cu
        ops/small_matrix_test_hooks.cu
        ops/small_matrix_workspace.cu
        ops/solver.cu
        ops/point_cloud.cu
    )
endif()
