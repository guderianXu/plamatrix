# ---- CPU sources (always compiled) ----
target_sources(plamatrix PRIVATE
    core/gpu_backend.cpp
    sparse/csr_matrix.cpp
    ops/gemm_cpu.cpp
    ops/decomposition_cpu.cpp
    ops/solver_cpu.cpp
    ops/point_cloud_cpu.cpp
)

# ---- GPU / CUDA sources (only when CUDA is enabled) ----
if(PLAMATRIX_WITH_CUDA)
    target_sources(plamatrix PRIVATE
        core/gpu_runtime_cuda.cpp
        plamatrix.cu
        dense/dense_matrix.cu
        dense/dense_ops.cu
        ops/gemm.cu
        ops/decomposition.cu
        ops/solver.cu
        ops/point_cloud.cu
    )
elseif(PLAMATRIX_WITH_METAL_RESOLVED)
    target_sources(plamatrix PRIVATE
        core/metal_context.mm
        core/gpu_runtime_metal.mm
        dense/dense_matrix_metal.mm
        dense/dense_ops_metal.mm
        ops/decomposition_metal.mm
        ops/gemm_metal.mm
        ops/solver_metal.mm
        ops/point_cloud_metal.mm
    )
else()
    target_sources(plamatrix PRIVATE
        core/gpu_runtime_cuda.cpp
    )
endif()
