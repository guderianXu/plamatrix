#include "block_schur_device_assembly.h"

#include "block_schur_sparse_assembly.h"

#include "plamatrix/opencl/execution.h"

#include <algorithm>
#include <memory>
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
struct OpenClAssemblyState
{
    std::unique_ptr<opencl::CommandQueue> queue;
    std::unique_ptr<opencl::CompiledKernel> kernel;
    opencl::DeviceBuffer primary;
    opencl::DeviceBuffer inverse;
    opencl::DeviceBuffer direct;
    opencl::DeviceBuffer cross;
    opencl::DeviceBuffer baseKinds;
    opencl::DeviceBuffer baseIndices;
    opencl::DeviceBuffer blockSlots;
    opencl::DeviceBuffer rows;
    opencl::DeviceBuffer columns;
    opencl::DeviceBuffer termOffsets;
    opencl::DeviceBuffer termEliminated;
    opencl::DeviceBuffer termLeft;
    opencl::DeviceBuffer termRight;
    opencl::DeviceBuffer output;
};

bool ensureBuffer(opencl::DeviceBuffer* buffer,
                  cl_context context,
                  cl_mem_flags flags,
                  std::size_t bytes)
{
    if (buffer->size() == bytes)
    {
        return false;
    }
    *buffer = opencl::DeviceBuffer(context, flags, bytes);
    return true;
}

template <typename Value>
void uploadBuffer(cl_command_queue queue,
                  opencl::DeviceBuffer* buffer,
                  const std::vector<Value>& values)
{
    const std::size_t bytes = opencl::byteSize<Value>(values.size());
    if (bytes > buffer->size())
    {
        throw std::out_of_range("OpenCL Schur assembly upload exceeds buffer");
    }
    opencl::checkOpenCl(clEnqueueWriteBuffer(
        queue, buffer->get(), CL_FALSE, 0, bytes, values.data(),
        0, nullptr, nullptr), "clEnqueueWriteBuffer(Schur assembly)");
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
    const std::vector<Index>& term_right_cross,
    SchurComplementSolverWorkspace<Scalar>& workspace,
    bool upload_topology)
{
    using namespace opencl;
    requireUsableOpenClDevice();
    auto& runtime = OpenClRuntime::instance();
    requireFp64<Scalar>(runtime);
    const std::string options = std::is_same_v<Scalar, double>
        ? "-DREAL_DOUBLE=1" : std::string{};
    auto& opaque_state = SchurComplementSolverWorkspaceAccess::deviceAssemblyState(workspace);
    auto state = std::static_pointer_cast<OpenClAssemblyState<Scalar>>(opaque_state);
    if (!state)
    {
        state = std::make_shared<OpenClAssemblyState<Scalar>>();
        state->queue = std::make_unique<CommandQueue>(runtime.createQueue());
        cl_program program = runtime.program(
            "plamatrix_block_schur_device_assembly", kAssemblySource, options);
        state->kernel = std::make_unique<CompiledKernel>(program, "assembleSchurValues");
        opaque_state = state;
        upload_topology = true;
    }
    const cl_command_queue queue = state->queue->get();
    CompiledKernel& kernel = *state->kernel;

    auto primary_values = padded(primary_diagonal);
    auto inverse_values = padded(eliminated_inverse);
    auto direct_values = padded(primary_cross_values);
    auto cross_block_values = padded(cross_values);
    auto term_eliminated_values = padded(term_eliminated);
    auto term_left_values = padded(term_left_cross);
    auto term_right_values = padded(term_right_cross);
    auto output = std::vector<Scalar>(base_kinds.size(), Scalar(0));
    const auto read_only = CL_MEM_READ_ONLY;
    ensureBuffer(&state->primary, runtime.context(), read_only,
                 byteSize<Scalar>(primary_values.size()));
    ensureBuffer(&state->inverse, runtime.context(), read_only,
                 byteSize<Scalar>(inverse_values.size()));
    ensureBuffer(&state->direct, runtime.context(), read_only,
                 byteSize<Scalar>(direct_values.size()));
    ensureBuffer(&state->cross, runtime.context(), read_only,
                 byteSize<Scalar>(cross_block_values.size()));
    uploadBuffer(queue, &state->primary, primary_values);
    uploadBuffer(queue, &state->inverse, inverse_values);
    uploadBuffer(queue, &state->direct, direct_values);
    uploadBuffer(queue, &state->cross, cross_block_values);
    const bool topology_resized =
        ensureBuffer(&state->baseKinds, runtime.context(), read_only,
                     byteSize<Index>(base_kinds.size())) |
        ensureBuffer(&state->baseIndices, runtime.context(), read_only,
                     byteSize<Index>(base_indices.size())) |
        ensureBuffer(&state->blockSlots, runtime.context(), read_only,
                     byteSize<Index>(value_block_slots.size())) |
        ensureBuffer(&state->rows, runtime.context(), read_only,
                     byteSize<Index>(local_rows.size())) |
        ensureBuffer(&state->columns, runtime.context(), read_only,
                     byteSize<Index>(local_columns.size())) |
        ensureBuffer(&state->termOffsets, runtime.context(), read_only,
                     byteSize<Index>(term_offsets.size())) |
        ensureBuffer(&state->termEliminated, runtime.context(), read_only,
                     byteSize<Index>(term_eliminated_values.size())) |
        ensureBuffer(&state->termLeft, runtime.context(), read_only,
                     byteSize<Index>(term_left_values.size())) |
        ensureBuffer(&state->termRight, runtime.context(), read_only,
                     byteSize<Index>(term_right_values.size()));
    if (upload_topology || topology_resized)
    {
        uploadBuffer(queue, &state->baseKinds, base_kinds);
        uploadBuffer(queue, &state->baseIndices, base_indices);
        uploadBuffer(queue, &state->blockSlots, value_block_slots);
        uploadBuffer(queue, &state->rows, local_rows);
        uploadBuffer(queue, &state->columns, local_columns);
        uploadBuffer(queue, &state->termOffsets, term_offsets);
        uploadBuffer(queue, &state->termEliminated, term_eliminated_values);
        uploadBuffer(queue, &state->termLeft, term_left_values);
        uploadBuffer(queue, &state->termRight, term_right_values);
    }
    ensureBuffer(&state->output, runtime.context(), CL_MEM_READ_WRITE,
                 byteSize<Scalar>(output.size()));

    kernelArg(kernel, 0, static_cast<cl_long>(output.size()));
    kernelArg(kernel, 1, static_cast<cl_long>(primary_size));
    kernelArg(kernel, 2, static_cast<cl_long>(eliminated_size));
    kernelBufferArg(kernel, 3, state->primary);
    kernelBufferArg(kernel, 4, state->inverse);
    kernelBufferArg(kernel, 5, state->direct);
    kernelBufferArg(kernel, 6, state->cross);
    kernelBufferArg(kernel, 7, state->baseKinds);
    kernelBufferArg(kernel, 8, state->baseIndices);
    kernelBufferArg(kernel, 9, state->blockSlots);
    kernelBufferArg(kernel, 10, state->rows);
    kernelBufferArg(kernel, 11, state->columns);
    kernelBufferArg(kernel, 12, state->termOffsets);
    kernelBufferArg(kernel, 13, state->termEliminated);
    kernelBufferArg(kernel, 14, state->termLeft);
    kernelBufferArg(kernel, 15, state->termRight);
    kernelBufferArg(kernel, 16, state->output);
    const std::size_t global_size = output.size();
    checkOpenCl(clEnqueueNDRangeKernel(
        queue, kernel.get(), 1, nullptr, &global_size, nullptr,
        0, nullptr, nullptr), "clEnqueueNDRangeKernel(Schur assembly)");
    readVector(queue, state->output, output);
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
    const std::vector<Index>& term_right_cross,
    SchurComplementSolverWorkspace<Scalar>& workspace,
    bool upload_topology)
{
    return assemble(
        primary_size, eliminated_size, primary_diagonal, eliminated_inverse,
        primary_cross_values, cross_values, base_kinds, base_indices,
        value_block_slots, local_rows, local_columns, term_offsets, term_eliminated,
        term_left_cross, term_right_cross, workspace, upload_topology);
}

template std::vector<float> assembleSchurValuesOnOpenCl(
    Index, Index, const std::vector<float>&, const std::vector<float>&,
    const std::vector<float>&, const std::vector<float>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&,
    SchurComplementSolverWorkspace<float>&, bool);
template std::vector<double> assembleSchurValuesOnOpenCl(
    Index, Index, const std::vector<double>&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<double>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&,
    SchurComplementSolverWorkspace<double>&, bool);

} // namespace plamatrix::block_schur_detail
