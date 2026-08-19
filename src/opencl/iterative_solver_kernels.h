#pragma once

namespace plamatrix::opencl::iterative_solver_detail
{

inline constexpr const char* kSolverSource = R"CLC(
#if defined(REAL_DOUBLE) || defined(ACCUM_DOUBLE)
#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif
#endif
#if defined(REAL_DOUBLE)
typedef double real;
#else
typedef float real;
#endif
#if defined(REAL_DOUBLE) || defined(ACCUM_DOUBLE)
typedef double accum;
#else
typedef float accum;
#endif
typedef long index_t;

__kernel void spmv(
    __global const index_t* rows, __global const index_t* columns,
    __global const real* values, __global const real* x,
    __global real* output, const index_t size)
{
    const index_t row = (index_t)get_global_id(0);
    if (row >= size) return;
    real sum = (real)0;
    for (index_t entry = rows[row]; entry < rows[row + 1]; ++entry)
        sum += values[entry] * x[columns[entry]];
    output[row] = sum;
}

__kernel void initialize(
    __global const real* rhs, __global const real* matrix_x,
    __global const real* inverse_diagonal, __global real* residual,
    __global real* transformed, __global real* direction, const index_t size)
{
    const index_t i = (index_t)get_global_id(0);
    if (i >= size) return;
    const real r = rhs[i] - matrix_x[i];
    const real z = inverse_diagonal[i] * r;
    residual[i] = r;
    transformed[i] = z;
    direction[i] = z;
}

__kernel void initializeResidual(
    __global const real* rhs, __global const real* matrix_x,
    __global real* residual, const index_t size)
{
    const index_t i = (index_t)get_global_id(0);
    if (i < size) residual[i] = rhs[i] - matrix_x[i];
}

__kernel void updateSolutionResidual(
    __global real* solution, __global real* residual,
    __global const real* direction, __global const real* matrix_direction,
    const real alpha, const index_t size)
{
    const index_t i = (index_t)get_global_id(0);
    if (i >= size) return;
    solution[i] += alpha * direction[i];
    residual[i] -= alpha * matrix_direction[i];
}

__kernel void applyPreconditioner(
    __global const real* inverse_diagonal, __global const real* residual,
    __global real* transformed, const index_t size)
{
    const index_t i = (index_t)get_global_id(0);
    if (i < size) transformed[i] = inverse_diagonal[i] * residual[i];
}

__kernel void applyBlockPreconditioner(
    __global const real* inverse_blocks, __global const real* residual,
    __global real* transformed, const index_t size, const index_t block_size)
{
    const index_t row = (index_t)get_global_id(0);
    if (row >= size) return;
    const index_t block = row / block_size;
    const index_t local_row = row - block * block_size;
    const index_t block_offset = block * block_size * block_size;
    const index_t residual_offset = block * block_size;
    real value = (real)0;
    for (index_t column = 0; column < block_size; ++column)
        value += inverse_blocks[block_offset + local_row * block_size + column]
               * residual[residual_offset + column];
    transformed[row] = value;
}

__kernel void copyVector(
    __global const real* input, __global real* output, const index_t size)
{
    const index_t i = (index_t)get_global_id(0);
    if (i < size) output[i] = input[i];
}

__kernel void updateDirection(
    __global real* direction, __global const real* transformed,
    const real beta, const index_t size)
{
    const index_t i = (index_t)get_global_id(0);
    if (i < size) direction[i] = transformed[i] + beta * direction[i];
}

__kernel void dotPartial(
    __global const real* first, __global const real* second,
    __global accum* partial, const index_t size, __local accum* scratch)
{
    const size_t local_id = get_local_id(0);
    const size_t local_size = get_local_size(0);
    const size_t index = get_group_id(0) * local_size * 2 + local_id;
    accum sum = (accum)0;
    if (index < (size_t)size) sum = (accum)first[index] * (accum)second[index];
    if (index + local_size < (size_t)size)
        sum += (accum)first[index + local_size] * (accum)second[index + local_size];
    scratch[local_id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (size_t stride = local_size / 2; stride > 0; stride /= 2)
    {
        if (local_id < stride) scratch[local_id] += scratch[local_id + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) partial[get_group_id(0)] = scratch[0];
}

__kernel void reducePartial(
    __global const accum* input, __global accum* output,
    const index_t size, __local accum* scratch)
{
    const size_t local_id = get_local_id(0);
    const size_t local_size = get_local_size(0);
    const size_t index = get_group_id(0) * local_size * 2 + local_id;
    accum sum = index < (size_t)size ? input[index] : (accum)0;
    if (index + local_size < (size_t)size) sum += input[index + local_size];
    scratch[local_id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (size_t stride = local_size / 2; stride > 0; stride /= 2)
    {
        if (local_id < stride) scratch[local_id] += scratch[local_id + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) output[get_group_id(0)] = scratch[0];
}
)CLC";

} // namespace plamatrix::opencl::iterative_solver_detail
