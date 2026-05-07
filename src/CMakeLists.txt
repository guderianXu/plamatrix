# ---- CPU sources (always compiled) ----
target_sources(plamatrix PRIVATE
    sparse/csr_matrix.cpp
    ops/gemm_cpu.cpp
    ops/decomposition_cpu.cpp
    ops/solver_cpu.cpp
    ops/point_cloud_cpu.cpp
)

# ---- GPU / CUDA sources (only when CUDA is enabled) ----
if(PLAMATRIX_WITH_CUDA)
    target_sources(plamatrix PRIVATE
        plamatrix.cu
        dense/dense_matrix.cu
        dense/dense_ops.cu
        ops/gemm.cu
        ops/decomposition.cu
        ops/solver.cu
        ops/point_cloud.cu
    )
endif()
