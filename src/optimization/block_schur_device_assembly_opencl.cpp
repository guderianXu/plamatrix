#include "block_schur_device_assembly.h"

#include "plamatrix/opencl/execution.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace plamatrix::block_schur_detail
{
namespace
{

inline constexpr const char* kAssemblySource = R"CLC(
#if defined(REAL_DOUBLE)
#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif
typedef double real;
#else
typedef float real;
#endif
typedef long index_t;

__kernel void assembleSchurValues(
    const index_t value_count,
    const index_t primary_size,
    const index_t eliminated_size,
    __global const real* primary_diagonal,
    __global const real* eliminated_inverse,
    __global const real* primary_cross_values,
    __global const real* cross_values,
    __global const index_t* base_kinds,
    __global const index_t* base_indices,
    __global const index_t* value_block_slots,
    __global const index_t* local_rows,
    __global const index_t* local_columns,
    __global const index_t* term_offsets,
    __global const index_t* term_eliminated,
    __global const index_t* term_left_cross,
    __global const index_t* term_right_cross,
    __global real* output)
{
    const index_t value_index = (index_t)get_global_id(0);
    if (value_index >= value_count) return;
    real value = (real)0;
    if (base_kinds[value_index] == 1)
        value = primary_diagonal[base_indices[value_index]];
    else if (base_kinds[value_index] == 2)
        value = primary_cross_values[base_indices[value_index]];
    const index_t row = local_rows[value_index];
    const index_t column = local_columns[value_index];
    const index_t slot = value_block_slots[value_index];
    const index_t cross_stride = primary_size * eliminated_size;
    const index_t inverse_stride = eliminated_size * eliminated_size;
    for (index_t term = term_offsets[slot];
         term < term_offsets[slot + 1]; ++term)
    {
        const index_t left_offset = term_left_cross[term] * cross_stride;
        const index_t right_offset = term_right_cross[term] * cross_stride;
        const index_t inverse_offset = term_eliminated[term] * inverse_stride;
        real product = (real)0;
        for (index_t inner_row = 0; inner_row < eliminated_size; ++inner_row)
            for (index_t inner_column = 0; inner_column < eliminated_size; ++inner_column)
                product += cross_values[left_offset + row * eliminated_size + inner_row]
                         * eliminated_inverse[inverse_offset + inner_row * eliminated_size
                                              + inner_column]
                         * cross_values[right_offset + column * eliminated_size + inner_column];
        value -= product;
    }
    output[value_index] = value;
}
)CLC";

template <typename Value>
std::vector<Value> padded(const std::vector<Value>& values)
{
    return values.empty() ? std::vector<Value>{Value{}} : values;
}

template <typename Scalar>
std::vector<Scalar> assemble(
    Index primary_size,
    Index eliminated_size,
    const std::vector<Scalar>& primary_diagonal,
    const std::vector<Scalar>& eliminated_inverse,
    const std::vector<Scalar>& primary_cross_values,
    const std::vector<Scalar>& cross_values,
    const std::vector<Index>& base_kinds,
    const std::vector<Index>& base_indices,
    const std::vector<Index>& value_block_slots,
    const std::vector<Index>& local_rows,
    const std::vector<Index>& local_columns,
    const std::vector<Index>& term_offsets,
    const std::vector<Index>& term_eliminated,
    const std::vector<Index>& term_left_cross,
    const std::vector<Index>& term_right_cross)
{
    using namespace opencl;
    requireUsableOpenClDevice();
    auto& runtime = OpenClRuntime::instance();
    requireFp64<Scalar>(runtime);
    auto queue = CommandQueue(runtime.createQueue());
    const std::string options = std::is_same_v<Scalar, double>
        ? "-DREAL_DOUBLE=1" : std::string{};
    cl_program program = runtime.program(
        "plamatrix_block_schur_device_assembly", kAssemblySource, options);
    CompiledKernel kernel(program, "assembleSchurValues");

    auto primary_values = padded(primary_diagonal);
    auto inverse_values = padded(eliminated_inverse);
    auto direct_values = padded(primary_cross_values);
    auto cross_block_values = padded(cross_values);
    auto term_eliminated_values = padded(term_eliminated);
    auto term_left_values = padded(term_left_cross);
    auto term_right_values = padded(term_right_cross);
    auto output = std::vector<Scalar>(base_kinds.size(), Scalar(0));
    auto primary_buffer = inputVector(runtime, primary_values);
    auto inverse_buffer = inputVector(runtime, inverse_values);
    auto direct_buffer = inputVector(runtime, direct_values);
    auto cross_buffer = inputVector(runtime, cross_block_values);
    auto base_kind_buffer = inputVector(runtime, base_kinds);
    auto base_index_buffer = inputVector(runtime, base_indices);
    auto block_slot_buffer = inputVector(runtime, value_block_slots);
    auto row_buffer = inputVector(runtime, local_rows);
    auto column_buffer = inputVector(runtime, local_columns);
    auto term_offset_buffer = inputVector(runtime, term_offsets);
    auto term_eliminated_buffer = inputVector(runtime, term_eliminated_values);
    auto term_left_buffer = inputVector(runtime, term_left_values);
    auto term_right_buffer = inputVector(runtime, term_right_values);
    auto output_buffer = inOutVector(runtime, output);

    kernelArg(kernel, 0, static_cast<cl_long>(output.size()));
    kernelArg(kernel, 1, static_cast<cl_long>(primary_size));
    kernelArg(kernel, 2, static_cast<cl_long>(eliminated_size));
    kernelBufferArg(kernel, 3, primary_buffer);
    kernelBufferArg(kernel, 4, inverse_buffer);
    kernelBufferArg(kernel, 5, direct_buffer);
    kernelBufferArg(kernel, 6, cross_buffer);
    kernelBufferArg(kernel, 7, base_kind_buffer);
    kernelBufferArg(kernel, 8, base_index_buffer);
    kernelBufferArg(kernel, 9, block_slot_buffer);
    kernelBufferArg(kernel, 10, row_buffer);
    kernelBufferArg(kernel, 11, column_buffer);
    kernelBufferArg(kernel, 12, term_offset_buffer);
    kernelBufferArg(kernel, 13, term_eliminated_buffer);
    kernelBufferArg(kernel, 14, term_left_buffer);
    kernelBufferArg(kernel, 15, term_right_buffer);
    kernelBufferArg(kernel, 16, output_buffer);
    const std::size_t global_size = output.size();
    checkOpenCl(clEnqueueNDRangeKernel(
        queue.get(), kernel.get(), 1, nullptr, &global_size, nullptr,
        0, nullptr, nullptr), "clEnqueueNDRangeKernel(Schur assembly)");
    readVector(queue.get(), output_buffer, output);
    return output;
}

} // namespace

template <typename Scalar>
std::vector<Scalar> assembleSchurValuesOnOpenCl(
    Index primary_size,
    Index eliminated_size,
    const std::vector<Scalar>& primary_diagonal,
    const std::vector<Scalar>& eliminated_inverse,
    const std::vector<Scalar>& primary_cross_values,
    const std::vector<Scalar>& cross_values,
    const std::vector<Index>& base_kinds,
    const std::vector<Index>& base_indices,
    const std::vector<Index>& value_block_slots,
    const std::vector<Index>& local_rows,
    const std::vector<Index>& local_columns,
    const std::vector<Index>& term_offsets,
    const std::vector<Index>& term_eliminated,
    const std::vector<Index>& term_left_cross,
    const std::vector<Index>& term_right_cross)
{
    return assemble(
        primary_size, eliminated_size, primary_diagonal, eliminated_inverse,
        primary_cross_values, cross_values, base_kinds, base_indices,
        value_block_slots, local_rows, local_columns, term_offsets, term_eliminated,
        term_left_cross, term_right_cross);
}

template std::vector<float> assembleSchurValuesOnOpenCl(
    Index, Index, const std::vector<float>&, const std::vector<float>&,
    const std::vector<float>&, const std::vector<float>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&);
template std::vector<double> assembleSchurValuesOnOpenCl(
    Index, Index, const std::vector<double>&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<double>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&);

} // namespace plamatrix::block_schur_detail
