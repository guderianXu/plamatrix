#include "plamatrix/core/metal_context.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <cstring>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace plamatrix
{
namespace detail
{
namespace
{

struct MetalState
{
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    std::unordered_map<void*, id<MTLBuffer>> buffers;
    std::unordered_map<void*, std::size_t> buffer_lengths;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;
    std::mutex mutex;
};

NSString* metalShaderSource()
{
    return @R"METAL(
        #include <metal_stdlib>
        using namespace metal;

        kernel void fill_float(device float* data [[buffer(0)]],
                               constant float& value [[buffer(1)]],
                               constant uint& count [[buffer(2)]],
                               uint id [[thread_position_in_grid]])
        {
            if (id < count)
            {
                data[id] = value;
            }
        }

        kernel void transpose_float(device const float* src [[buffer(0)]],
                                    device float* dst [[buffer(1)]],
                                    constant uint& rows [[buffer(2)]],
                                    constant uint& cols [[buffer(3)]],
                                    uint2 id [[thread_position_in_grid]])
        {
            uint row = id.x;
            uint col = id.y;
            if (row < rows && col < cols)
            {
                dst[col + row * cols] = src[row + col * rows];
            }
        }

        kernel void add_float(device const float* a [[buffer(0)]],
                              device const float* b [[buffer(1)]],
                              device float* c [[buffer(2)]],
                              constant uint& count [[buffer(3)]],
                              uint id [[thread_position_in_grid]])
        {
            if (id < count)
            {
                c[id] = a[id] + b[id];
            }
        }

        kernel void sub_float(device const float* a [[buffer(0)]],
                              device const float* b [[buffer(1)]],
                              device float* c [[buffer(2)]],
                              constant uint& count [[buffer(3)]],
                              uint id [[thread_position_in_grid]])
        {
            if (id < count)
            {
                c[id] = a[id] - b[id];
            }
        }

        kernel void transform_points_float(device const float* transform [[buffer(0)]],
                                           device const float* points [[buffer(1)]],
                                           device float* output [[buffer(2)]],
                                           constant uint& point_count [[buffer(3)]],
                                           uint id [[thread_position_in_grid]])
        {
            if (id >= point_count)
            {
                return;
            }

            float px = points[id + 0 * point_count];
            float py = points[id + 1 * point_count];
            float pz = points[id + 2 * point_count];

            output[id + 0 * point_count] =
                transform[0] * px + transform[4] * py + transform[8] * pz + transform[12];
            output[id + 1 * point_count] =
                transform[1] * px + transform[5] * py + transform[9] * pz + transform[13];
            output[id + 2 * point_count] =
                transform[2] * px + transform[6] * py + transform[10] * pz + transform[14];
        }
    )METAL";
}

MetalState& state()
{
    static MetalState s;
    static std::once_flag once;
    std::call_once(once, [] {
        s.device = MTLCreateSystemDefaultDevice();
        if (s.device == nil)
        {
            throw std::runtime_error("Metal GPU backend requested but no MTLDevice is available");
        }
        s.queue = [s.device newCommandQueue];
        if (s.queue == nil)
        {
            throw std::runtime_error("Metal GPU backend failed to create MTLCommandQueue");
        }
        NSError* error = nil;
        s.library = [s.device newLibraryWithSource:metalShaderSource() options:nil error:&error];
        if (s.library == nil)
        {
            NSString* message = (error != nil) ? [error localizedDescription] : @"unknown Metal compiler error";
            throw std::runtime_error(std::string("Metal shader library compilation failed: ")
                                     + [message UTF8String]);
        }
    });
    return s;
}

struct BufferLookup
{
    id<MTLBuffer> buffer = nil;
    NSUInteger offset = 0;
};

BufferLookup bufferForPointer(const void* ptr)
{
    MetalState& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    const auto requested = reinterpret_cast<std::uintptr_t>(ptr);
    for (const auto& entry : s.buffers)
    {
        const auto base = reinterpret_cast<std::uintptr_t>(entry.first);
        const auto length_it = s.buffer_lengths.find(entry.first);
        const std::size_t length = (length_it == s.buffer_lengths.end()) ? 0 : length_it->second;
        if (requested >= base && requested < base + length)
        {
            return {entry.second, static_cast<NSUInteger>(requested - base)};
        }
    }
    throw std::runtime_error("Metal backend could not find MTLBuffer for pointer token");
}

id<MTLComputePipelineState> pipelineForFunction(const char* function_name)
{
    MetalState& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    const std::string key(function_name);
    auto it = s.pipelines.find(key);
    if (it != s.pipelines.end())
    {
        return it->second;
    }

    NSString* name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> function = [s.library newFunctionWithName:name];
    if (function == nil)
    {
        throw std::runtime_error(std::string("Metal function not found: ") + function_name);
    }

    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [s.device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil)
    {
        NSString* message = (error != nil) ? [error localizedDescription] : @"unknown pipeline error";
        throw std::runtime_error(std::string("Metal pipeline creation failed for ")
                                 + function_name + ": " + [message UTF8String]);
    }
    s.pipelines[key] = pipeline;
    return pipeline;
}

void checkCommandBuffer(id<MTLCommandBuffer> command_buffer, const char* op)
{
    if ([command_buffer status] == MTLCommandBufferStatusError)
    {
        NSError* error = [command_buffer error];
        NSString* message = (error != nil) ? [error localizedDescription] : @"unknown Metal command error";
        throw std::runtime_error(std::string(op) + ": " + [message UTF8String]);
    }
}

} // namespace

bool metalIsAvailable()
{
    return MTLCreateSystemDefaultDevice() != nil;
}

void* metalAllocateBytes(std::size_t bytes)
{
    if (bytes == 0)
    {
        return nullptr;
    }
    MetalState& s = state();
    id<MTLBuffer> buffer = [s.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (buffer == nil || [buffer contents] == nullptr)
    {
        throw std::runtime_error("Metal backend failed to allocate MTLBuffer");
    }
    void* ptr = [buffer contents];
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        s.buffers[ptr] = buffer;
        s.buffer_lengths[ptr] = bytes;
    }
    return ptr;
}

void metalFreeBytes(void* ptr) noexcept
{
    if (ptr == nullptr)
    {
        return;
    }
    try
    {
        MetalState& s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        s.buffers.erase(ptr);
        s.buffer_lengths.erase(ptr);
    }
    catch (...)
    {
    }
}

void metalCopyHostToDevice(void* dst, const void* src, std::size_t bytes)
{
    if (bytes > 0)
    {
        static_cast<void>(bufferForPointer(dst));
        std::memcpy(dst, src, bytes);
    }
}

void metalCopyDeviceToHost(void* dst, const void* src, std::size_t bytes)
{
    if (bytes > 0)
    {
        static_cast<void>(bufferForPointer(src));
        std::memcpy(dst, src, bytes);
    }
}

void metalCopyDeviceToDevice(void* dst, const void* src, std::size_t bytes)
{
    if (bytes > 0)
    {
        static_cast<void>(bufferForPointer(dst));
        static_cast<void>(bufferForPointer(src));
        std::memcpy(dst, src, bytes);
    }
}

void metalMemset(void* dst, int value, std::size_t bytes)
{
    if (bytes > 0)
    {
        static_cast<void>(bufferForPointer(dst));
        std::memset(dst, value, bytes);
    }
}

void metalFillFloat(void* dst, Index count, float value)
{
    if (count == 0)
    {
        return;
    }
    if (count > static_cast<Index>(std::numeric_limits<unsigned int>::max()))
    {
        throw std::runtime_error("metalFillFloat: element count exceeds Metal uint range");
    }
    MetalState& s = state();
    BufferLookup dst_buffer = bufferForPointer(dst);
    id<MTLComputePipelineState> pipeline = pipelineForFunction("fill_float");
    id<MTLCommandBuffer> command_buffer = [s.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

    unsigned int count_uint = static_cast<unsigned int>(count);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:dst_buffer.buffer offset:dst_buffer.offset atIndex:0];
    [encoder setBytes:&value length:sizeof(float) atIndex:1];
    [encoder setBytes:&count_uint length:sizeof(unsigned int) atIndex:2];

    NSUInteger threads = std::min<NSUInteger>(256, [pipeline maxTotalThreadsPerThreadgroup]);
    MTLSize threads_per_group = MTLSizeMake(threads, 1, 1);
    MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(count), 1, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads_per_group];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    checkCommandBuffer(command_buffer, "metalFillFloat");
}

void metalFillDoubleFallback(void* dst, Index count, double value)
{
    double* data = static_cast<double*>(dst);
    for (Index i = 0; i < count; ++i)
    {
        data[i] = value;
    }
}

void metalTransposeFloat(const void* src, void* dst, Index rows, Index cols)
{
    if (rows == 0 || cols == 0)
    {
        return;
    }
    if (rows > static_cast<Index>(std::numeric_limits<unsigned int>::max())
        || cols > static_cast<Index>(std::numeric_limits<unsigned int>::max()))
    {
        throw std::runtime_error("metalTransposeFloat: matrix extent exceeds Metal uint range");
    }
    MetalState& s = state();
    BufferLookup src_buffer = bufferForPointer(src);
    BufferLookup dst_buffer = bufferForPointer(dst);
    id<MTLComputePipelineState> pipeline = pipelineForFunction("transpose_float");
    id<MTLCommandBuffer> command_buffer = [s.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

    unsigned int rows_uint = static_cast<unsigned int>(rows);
    unsigned int cols_uint = static_cast<unsigned int>(cols);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:src_buffer.buffer offset:src_buffer.offset atIndex:0];
    [encoder setBuffer:dst_buffer.buffer offset:dst_buffer.offset atIndex:1];
    [encoder setBytes:&rows_uint length:sizeof(unsigned int) atIndex:2];
    [encoder setBytes:&cols_uint length:sizeof(unsigned int) atIndex:3];

    MTLSize threads_per_group = MTLSizeMake(16, 16, 1);
    MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(rows), static_cast<NSUInteger>(cols), 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads_per_group];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    checkCommandBuffer(command_buffer, "metalTransposeFloat");
}

void metalTransposeDoubleFallback(const void* src, void* dst, Index rows, Index cols)
{
    const double* src_data = static_cast<const double*>(src);
    double* dst_data = static_cast<double*>(dst);
    for (Index col = 0; col < cols; ++col)
    {
        for (Index row = 0; row < rows; ++row)
        {
            dst_data[col + row * cols] = src_data[row + col * rows];
        }
    }
}

void metalBinaryFloat(const char* function_name, const void* a, const void* b, void* c, Index count)
{
    if (count == 0)
    {
        return;
    }
    if (count > static_cast<Index>(std::numeric_limits<unsigned int>::max()))
    {
        throw std::runtime_error(std::string(function_name) + ": element count exceeds Metal uint range");
    }
    MetalState& s = state();
    BufferLookup a_buffer = bufferForPointer(a);
    BufferLookup b_buffer = bufferForPointer(b);
    BufferLookup c_buffer = bufferForPointer(c);
    id<MTLComputePipelineState> pipeline = pipelineForFunction(function_name);
    id<MTLCommandBuffer> command_buffer = [s.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

    unsigned int count_uint = static_cast<unsigned int>(count);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:a_buffer.buffer offset:a_buffer.offset atIndex:0];
    [encoder setBuffer:b_buffer.buffer offset:b_buffer.offset atIndex:1];
    [encoder setBuffer:c_buffer.buffer offset:c_buffer.offset atIndex:2];
    [encoder setBytes:&count_uint length:sizeof(unsigned int) atIndex:3];

    NSUInteger threads = std::min<NSUInteger>(256, [pipeline maxTotalThreadsPerThreadgroup]);
    MTLSize threads_per_group = MTLSizeMake(threads, 1, 1);
    MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(count), 1, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads_per_group];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    checkCommandBuffer(command_buffer, function_name);
}

void metalAddFloat(const void* a, const void* b, void* c, Index count)
{
    metalBinaryFloat("add_float", a, b, c, count);
}

void metalSubFloat(const void* a, const void* b, void* c, Index count)
{
    metalBinaryFloat("sub_float", a, b, c, count);
}

void metalAddDoubleFallback(const void* a, const void* b, void* c, Index count)
{
    const double* a_data = static_cast<const double*>(a);
    const double* b_data = static_cast<const double*>(b);
    double* c_data = static_cast<double*>(c);
    for (Index i = 0; i < count; ++i)
    {
        c_data[i] = a_data[i] + b_data[i];
    }
}

void metalSubDoubleFallback(const void* a, const void* b, void* c, Index count)
{
    const double* a_data = static_cast<const double*>(a);
    const double* b_data = static_cast<const double*>(b);
    double* c_data = static_cast<double*>(c);
    for (Index i = 0; i < count; ++i)
    {
        c_data[i] = a_data[i] - b_data[i];
    }
}

NSUInteger checkedMetalExtent(Index value, const char* name)
{
    if (value < 0 || static_cast<unsigned long long>(value) > std::numeric_limits<NSUInteger>::max())
    {
        throw std::runtime_error(std::string(name) + " is outside Metal NSUInteger range");
    }
    return static_cast<NSUInteger>(value);
}

NSUInteger checkedMetalRowBytes(Index cols, std::size_t element_bytes, const char* name)
{
    if (cols < 0)
    {
        throw std::runtime_error(std::string(name) + " has negative column count");
    }
    const auto cols_size = static_cast<std::size_t>(cols);
    if (cols_size > std::numeric_limits<std::size_t>::max() / element_bytes)
    {
        throw std::runtime_error(std::string(name) + " row byte count overflows size_t");
    }
    const auto row_bytes = cols_size * element_bytes;
    if (static_cast<unsigned long long>(row_bytes) > std::numeric_limits<NSUInteger>::max())
    {
        throw std::runtime_error(std::string(name) + " row byte count exceeds Metal NSUInteger range");
    }
    return static_cast<NSUInteger>(row_bytes);
}

std::size_t checkedMetalElementBytes(Index rows, Index cols, std::size_t element_bytes, const char* name)
{
    if (rows < 0 || cols < 0)
    {
        throw std::runtime_error(std::string(name) + " has negative extent");
    }
    const auto rows_size = static_cast<std::size_t>(rows);
    const auto cols_size = static_cast<std::size_t>(cols);
    if (rows_size != 0 && cols_size > std::numeric_limits<std::size_t>::max() / rows_size)
    {
        throw std::runtime_error(std::string(name) + " element count overflows size_t");
    }
    const auto count = rows_size * cols_size;
    if (count != 0 && element_bytes > std::numeric_limits<std::size_t>::max() / count)
    {
        throw std::runtime_error(std::string(name) + " byte count overflows size_t");
    }
    return count * element_bytes;
}

class MetalTempAllocation
{
public:
    explicit MetalTempAllocation(std::size_t bytes)
        : _ptr(metalAllocateBytes(bytes))
    {
    }

    ~MetalTempAllocation()
    {
        metalFreeBytes(_ptr);
    }

    MetalTempAllocation(const MetalTempAllocation&) = delete;
    MetalTempAllocation& operator=(const MetalTempAllocation&) = delete;

    void* ptr() const
    {
        return _ptr;
    }

private:
    void* _ptr;
};

void metalGemmFloat(const void* a, const void* b, void* c, Index m, Index n, Index k)
{
    if (m == 0 || n == 0)
    {
        return;
    }
    if (k == 0)
    {
        metalMemset(c, 0, checkedMetalElementBytes(m, n, sizeof(float), "GEMM result"));
        return;
    }

    MetalState& s = state();
    BufferLookup a_buffer = bufferForPointer(a);
    BufferLookup b_buffer = bufferForPointer(b);
    BufferLookup c_buffer = bufferForPointer(c);

    NSUInteger m_size = checkedMetalExtent(m, "GEMM m");
    NSUInteger n_size = checkedMetalExtent(n, "GEMM n");
    NSUInteger k_size = checkedMetalExtent(k, "GEMM k");

    MPSMatrixDescriptor* left_descriptor =
        [MPSMatrixDescriptor matrixDescriptorWithRows:n_size
                                              columns:k_size
                                             rowBytes:checkedMetalRowBytes(k, sizeof(float), "GEMM left")
                                             dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* right_descriptor =
        [MPSMatrixDescriptor matrixDescriptorWithRows:k_size
                                              columns:m_size
                                             rowBytes:checkedMetalRowBytes(m, sizeof(float), "GEMM right")
                                             dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* result_descriptor =
        [MPSMatrixDescriptor matrixDescriptorWithRows:n_size
                                              columns:m_size
                                             rowBytes:checkedMetalRowBytes(m, sizeof(float), "GEMM result")
                                             dataType:MPSDataTypeFloat32];

    MPSMatrix* left = [[MPSMatrix alloc] initWithBuffer:b_buffer.buffer
                                                offset:b_buffer.offset
                                            descriptor:left_descriptor];
    MPSMatrix* right = [[MPSMatrix alloc] initWithBuffer:a_buffer.buffer
                                                 offset:a_buffer.offset
                                             descriptor:right_descriptor];
    MPSMatrix* result = [[MPSMatrix alloc] initWithBuffer:c_buffer.buffer
                                                  offset:c_buffer.offset
                                              descriptor:result_descriptor];

    MPSMatrixMultiplication* multiplication =
        [[MPSMatrixMultiplication alloc] initWithDevice:s.device
                                          transposeLeft:NO
                                         transposeRight:NO
                                             resultRows:n_size
                                          resultColumns:m_size
                                        interiorColumns:k_size
                                                  alpha:1.0
                                                   beta:0.0];

    id<MTLCommandBuffer> command_buffer = [s.queue commandBuffer];
    [multiplication encodeToCommandBuffer:command_buffer
                               leftMatrix:left
                              rightMatrix:right
                             resultMatrix:result];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    checkCommandBuffer(command_buffer, "metalGemmFloat");
}

void metalGemmDoubleFallback(const void* a, const void* b, void* c, Index m, Index n, Index k)
{
    const double* a_data = static_cast<const double*>(a);
    const double* b_data = static_cast<const double*>(b);
    double* c_data = static_cast<double*>(c);
    for (Index col = 0; col < n; ++col)
    {
        for (Index row = 0; row < m; ++row)
        {
            double sum = 0.0;
            for (Index p = 0; p < k; ++p)
            {
                sum += a_data[row + p * m] * b_data[p + col * k];
            }
            c_data[row + col * m] = sum;
        }
    }
}

void metalSolveFloat(const void* a, const void* b, void* x, Index n, Index nrhs)
{
    if (n == 0 || nrhs == 0)
    {
        throw std::runtime_error("Solve: input matrix has zero dimensions");
    }

    MetalState& s = state();
    static_cast<void>(bufferForPointer(a));
    static_cast<void>(bufferForPointer(b));
    BufferLookup x_buffer = bufferForPointer(x);

    NSUInteger n_size = checkedMetalExtent(n, "Solve n");
    NSUInteger nrhs_size = checkedMetalExtent(nrhs, "Solve nrhs");
    const std::size_t matrix_bytes = checkedMetalElementBytes(n, n, sizeof(float), "Solve matrix");
    const std::size_t rhs_bytes = checkedMetalElementBytes(n, nrhs, sizeof(float), "Solve RHS");
    const std::size_t pivot_bytes = checkedMetalElementBytes(1, n, sizeof(std::uint32_t), "Solve pivots");

    MetalTempAllocation source(matrix_bytes);
    MetalTempAllocation lu(matrix_bytes);
    MetalTempAllocation rhs(rhs_bytes);
    MetalTempAllocation result(rhs_bytes);
    MetalTempAllocation pivots(pivot_bytes);

    const float* a_data = static_cast<const float*>(a);
    float* source_data = static_cast<float*>(source.ptr());
    for (Index row = 0; row < n; ++row)
    {
        for (Index col = 0; col < n; ++col)
        {
            source_data[row * n + col] = a_data[row + col * n];
        }
    }

    const float* b_data = static_cast<const float*>(b);
    float* rhs_data = static_cast<float*>(rhs.ptr());
    for (Index row = 0; row < n; ++row)
    {
        for (Index col = 0; col < nrhs; ++col)
        {
            rhs_data[row * nrhs + col] = b_data[row + col * n];
        }
    }

    BufferLookup source_buffer = bufferForPointer(source.ptr());
    BufferLookup lu_buffer = bufferForPointer(lu.ptr());
    BufferLookup rhs_buffer = bufferForPointer(rhs.ptr());
    BufferLookup result_buffer = bufferForPointer(result.ptr());
    BufferLookup pivot_buffer = bufferForPointer(pivots.ptr());

    MPSMatrixDescriptor* square_descriptor =
        [MPSMatrixDescriptor matrixDescriptorWithRows:n_size
                                              columns:n_size
                                             rowBytes:checkedMetalRowBytes(n, sizeof(float), "Solve square")
                                             dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* rhs_descriptor =
        [MPSMatrixDescriptor matrixDescriptorWithRows:n_size
                                              columns:nrhs_size
                                             rowBytes:checkedMetalRowBytes(nrhs, sizeof(float), "Solve RHS")
                                             dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* pivot_descriptor =
        [MPSMatrixDescriptor matrixDescriptorWithRows:1
                                              columns:n_size
                                             rowBytes:checkedMetalRowBytes(n, sizeof(std::uint32_t), "Solve pivots")
                                             dataType:MPSDataTypeUInt32];

    MPSMatrix* source_matrix = [[MPSMatrix alloc] initWithBuffer:source_buffer.buffer
                                                         offset:source_buffer.offset
                                                     descriptor:square_descriptor];
    MPSMatrix* lu_matrix = [[MPSMatrix alloc] initWithBuffer:lu_buffer.buffer
                                                     offset:lu_buffer.offset
                                                 descriptor:square_descriptor];
    MPSMatrix* rhs_matrix = [[MPSMatrix alloc] initWithBuffer:rhs_buffer.buffer
                                                      offset:rhs_buffer.offset
                                                  descriptor:rhs_descriptor];
    MPSMatrix* result_matrix = [[MPSMatrix alloc] initWithBuffer:result_buffer.buffer
                                                         offset:result_buffer.offset
                                                     descriptor:rhs_descriptor];
    MPSMatrix* pivot_matrix = [[MPSMatrix alloc] initWithBuffer:pivot_buffer.buffer
                                                        offset:pivot_buffer.offset
                                                    descriptor:pivot_descriptor];

    id<MTLBuffer> status_buffer =
        [s.device newBufferWithLength:sizeof(MPSMatrixDecompositionStatus)
                              options:MTLResourceStorageModeShared];
    if (status_buffer == nil || [status_buffer contents] == nullptr)
    {
        throw std::runtime_error("metalSolveFloat: failed to allocate decomposition status buffer");
    }

    MPSMatrixDecompositionLU* decomposition =
        [[MPSMatrixDecompositionLU alloc] initWithDevice:s.device rows:n_size columns:n_size];
    MPSMatrixSolveLU* solve =
        [[MPSMatrixSolveLU alloc] initWithDevice:s.device
                                       transpose:NO
                                           order:n_size
                          numberOfRightHandSides:nrhs_size];

    id<MTLCommandBuffer> command_buffer = [s.queue commandBuffer];
    [decomposition encodeToCommandBuffer:command_buffer
                            sourceMatrix:source_matrix
                            resultMatrix:lu_matrix
                            pivotIndices:pivot_matrix
                                  status:status_buffer];
    [solve encodeToCommandBuffer:command_buffer
                    sourceMatrix:lu_matrix
             rightHandSideMatrix:rhs_matrix
                    pivotIndices:pivot_matrix
                  solutionMatrix:result_matrix];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    checkCommandBuffer(command_buffer, "metalSolveFloat");

    const auto status = *static_cast<MPSMatrixDecompositionStatus*>([status_buffer contents]);
    if (status == MPSMatrixDecompositionStatusSingular)
    {
        throw std::runtime_error("Solve: matrix is singular");
    }
    if (status != MPSMatrixDecompositionStatusSuccess)
    {
        throw std::runtime_error("Solve: MPS LU decomposition failed");
    }

    const float* result_data = static_cast<const float*>(result.ptr());
    float* x_data = static_cast<float*>(x);
    static_cast<void>(x_buffer);
    for (Index row = 0; row < n; ++row)
    {
        for (Index col = 0; col < nrhs; ++col)
        {
            x_data[row + col * n] = result_data[row * nrhs + col];
        }
    }
}

void metalTransformPointsFloat(const void* transform, const void* points, void* output, Index point_count)
{
    if (point_count == 0)
    {
        return;
    }
    if (point_count > static_cast<Index>(std::numeric_limits<unsigned int>::max()))
    {
        throw std::runtime_error("metalTransformPointsFloat: point count exceeds Metal uint range");
    }

    MetalState& s = state();
    BufferLookup transform_buffer = bufferForPointer(transform);
    BufferLookup points_buffer = bufferForPointer(points);
    BufferLookup output_buffer = bufferForPointer(output);
    id<MTLComputePipelineState> pipeline = pipelineForFunction("transform_points_float");
    id<MTLCommandBuffer> command_buffer = [s.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

    unsigned int point_count_uint = static_cast<unsigned int>(point_count);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:transform_buffer.buffer offset:transform_buffer.offset atIndex:0];
    [encoder setBuffer:points_buffer.buffer offset:points_buffer.offset atIndex:1];
    [encoder setBuffer:output_buffer.buffer offset:output_buffer.offset atIndex:2];
    [encoder setBytes:&point_count_uint length:sizeof(unsigned int) atIndex:3];

    NSUInteger threads = std::min<NSUInteger>(256, [pipeline maxTotalThreadsPerThreadgroup]);
    MTLSize threads_per_group = MTLSizeMake(threads, 1, 1);
    MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(point_count), 1, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads_per_group];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    checkCommandBuffer(command_buffer, "metalTransformPointsFloat");
}

void metalTransformPointsDoubleFallback(const void* transform, const void* points, void* output, Index point_count)
{
    static_cast<void>(bufferForPointer(transform));
    static_cast<void>(bufferForPointer(points));
    static_cast<void>(bufferForPointer(output));

    const double* t = static_cast<const double*>(transform);
    const double* p = static_cast<const double*>(points);
    double* out = static_cast<double*>(output);
    for (Index i = 0; i < point_count; ++i)
    {
        double px = p[i + 0 * point_count];
        double py = p[i + 1 * point_count];
        double pz = p[i + 2 * point_count];

        out[i + 0 * point_count] = t[0] * px + t[4] * py + t[8] * pz + t[12];
        out[i + 1 * point_count] = t[1] * px + t[5] * py + t[9] * pz + t[13];
        out[i + 2 * point_count] = t[2] * px + t[6] * py + t[10] * pz + t[14];
    }
}

} // namespace detail
} // namespace plamatrix
