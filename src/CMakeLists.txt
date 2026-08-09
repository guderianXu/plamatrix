# ---- CPU sources (always compiled) ----
target_sources(plamatrix PRIVATE
    sparse/csr_matrix.cpp
    sparse/iterative_solver_cpu.cpp
    sparse/sparse_ops_cpu.cpp
    dense/elementwise_cpu.cpp
    ops/gemm_cpu.cpp
    ops/decomposition_cpu.cpp
    ops/indexing_cpu.cpp
    ops/reduction_cpu.cpp
    ops/small_matrix_cpu.cpp
    ops/solver_cpu.cpp
    ops/point_cloud_cpu.cpp
)

# ---- OpenCL runtime sources ----
if(PLAMATRIX_WITH_OPENCL)
    target_sources(plamatrix PRIVATE
        opencl/device_enumeration.cpp
        opencl/runtime.cpp
    )
else()
    target_sources(plamatrix PRIVATE
        opencl/runtime_stub.cpp
    )
endif()

# ---- GPU / CUDA sources (only when CUDA is enabled) ----
if(PLAMATRIX_WITH_CUDA)
    target_sources(plamatrix PRIVATE
        plamatrix.cu
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
