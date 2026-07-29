#include <sstream>
#include <stdexcept>
#include <utility>

#include <cusparse.h>

#include "plamatrix/core/error_check.h"
#include "plamatrix/sparse/sparse_ops.h"

namespace plamatrix
{
namespace
{

template <typename Scalar>
constexpr cudaDataType scalarDataType();

template <>
constexpr cudaDataType scalarDataType<float>()
{
    return CUDA_R_32F;
}

template <>
constexpr cudaDataType scalarDataType<double>()
{
    return CUDA_R_64F;
}

cusparseHandle_t asHandle(void* handle)
{
    return reinterpret_cast<cusparseHandle_t>(handle);
}

cusparseSpMatDescr_t asSparseDescriptor(void* descriptor)
{
    return reinterpret_cast<cusparseSpMatDescr_t>(descriptor);
}

cusparseDnVecDescr_t asVectorDescriptor(void* descriptor)
{
    return reinterpret_cast<cusparseDnVecDescr_t>(descriptor);
}

cusparseDnMatDescr_t asMatrixDescriptor(void* descriptor)
{
    return reinterpret_cast<cusparseDnMatDescr_t>(descriptor);
}

template <typename Scalar>
void checkSpmvDimensions(const CSRMatrix<Scalar, Device::GPU>& csr,
                         const DenseMatrix<Scalar, Device::GPU>& x,
                         const DenseMatrix<Scalar, Device::GPU>& output)
{
    if (x.rows() != csr.cols() || x.cols() != 1)
    {
        std::ostringstream message;
        message << "spmv GPU dimension mismatch: CSR is " << csr.rows() << "x" << csr.cols()
                << ", x is " << x.rows() << "x" << x.cols();
        throw std::runtime_error(message.str());
    }
    if (output.rows() != csr.rows() || output.cols() != 1)
    {
        std::ostringstream message;
        message << "spmv GPU output dimension mismatch: output is " << output.rows() << "x"
                << output.cols() << ", expected " << csr.rows() << "x1";
        throw std::runtime_error(message.str());
    }
    if (x.data() != nullptr && x.data() == output.data())
    {
        throw std::invalid_argument("spmv GPU input and output data must not alias");
    }
}

template <typename Matrix>
void checkStorageStream(const char* name, const Matrix& matrix, cudaStream_t stream)
{
    if (matrix.isAsyncAllocation() && matrix.asyncAllocationStream() != stream)
    {
        std::ostringstream message;
        message << name << " must be used on the stream that owns its async allocation";
        throw std::logic_error(message.str());
    }
}

template <typename Scalar>
void checkCsrStructureStream(
    const char* name, const CSRMatrix<Scalar, Device::GPU>& matrix, cudaStream_t stream)
{
    if (!matrix.isStructureUsableOnStream(stream))
    {
        std::ostringstream message;
        message << name
                << " requires trusted CSR structure with no cross-stream pending write";
        throw std::logic_error(message.str());
    }
}

template <typename Scalar>
void checkSpmmDimensions(const CSRMatrix<Scalar, Device::GPU>& csr,
                         const DenseMatrix<Scalar, Device::GPU>& input,
                         const DenseMatrix<Scalar, Device::GPU>& output)
{
    if (input.rows() != csr.cols())
    {
        std::ostringstream message;
        message << "spmm GPU dimension mismatch: CSR is " << csr.rows() << "x" << csr.cols()
                << ", input is " << input.rows() << "x" << input.cols();
        throw std::runtime_error(message.str());
    }
    if (output.rows() != csr.rows() || output.cols() != input.cols())
    {
        std::ostringstream message;
        message << "spmm GPU output dimension mismatch: output is " << output.rows() << "x"
                << output.cols() << ", expected " << csr.rows() << "x" << input.cols();
        throw std::runtime_error(message.str());
    }
    if (input.data() != nullptr && input.data() == output.data())
    {
        throw std::invalid_argument("spmm GPU input and output data must not alias");
    }
}

} // anonymous namespace

struct SparseOpsWorkspaceAccess
{
    static cusparseHandle_t handle(SparseOpsWorkspace& workspace, cudaStream_t stream)
    {
        if (workspace._handle == nullptr)
        {
            cusparseHandle_t handle = nullptr;
            PLAMATRIX_CHECK_CUSPARSE(cusparseCreate(&handle));
            workspace._handle = handle;
        }
        PLAMATRIX_CHECK_CUSPARSE(cusparseSetStream(asHandle(workspace._handle), stream));
        return asHandle(workspace._handle);
    }

    static void bindStream(SparseOpsWorkspace& workspace, cudaStream_t stream)
    {
        if (workspace._hasStatusBatch)
        {
            throw std::logic_error(
                "SparseOpsWorkspace status must be checked before another sparse operation");
        }
        if (workspace._hasReuseStream && workspace._reuseStream != stream)
        {
            throw std::logic_error(
                "SparseOpsWorkspace cannot be reused on a different stream; close it first");
        }
        workspace._reuseStream = stream;
        workspace._hasReuseStream = true;
    }

    static void reserveAsync(
        SparseOpsWorkspace& workspace, std::size_t bytes, cudaStream_t stream)
    {
        bindStream(workspace, stream);
        if (bytes <= workspace._capacityBytes)
        {
            return;
        }
        if (workspace._buffer == nullptr)
        {
            PLAMATRIX_CHECK_CUDA(cudaMallocAsync(&workspace._buffer, bytes, stream));
            workspace._capacityBytes = bytes;
            workspace._allocationStream = stream;
            workspace._streamOrderedAllocation = true;
            return;
        }
        if (!workspace._streamOrderedAllocation || workspace._allocationStream != stream)
        {
            throw std::logic_error("SparseOpsWorkspace temporary buffer has incompatible ownership");
        }

        void* replacement = nullptr;
        PLAMATRIX_CHECK_CUDA(cudaMallocAsync(&replacement, bytes, stream));
        try
        {
            PLAMATRIX_CHECK_CUDA(cudaFreeAsync(workspace._buffer, stream));
        }
        catch (...)
        {
            static_cast<void>(cudaFreeAsync(replacement, stream));
            throw;
        }
        workspace._buffer = replacement;
        workspace._capacityBytes = bytes;
    }

    template <typename Scalar>
    static void prepareSparse(
        SparseOpsWorkspace& workspace, const CSRMatrix<Scalar, Device::GPU>& csr)
    {
        const int scalar_type = static_cast<int>(scalarDataType<Scalar>());
        const bool recreate = workspace._sparseDescriptor == nullptr
            || workspace._sparseRows != csr.rows() || workspace._sparseCols != csr.cols()
            || workspace._sparseNnz != csr.nnz()
            || workspace._descriptorScalarType != scalar_type;
        if (recreate)
        {
            workspace.destroyDescriptorsChecked();
            cusparseSpMatDescr_t descriptor = nullptr;
            PLAMATRIX_CHECK_CUSPARSE(cusparseCreateCsr(
                &descriptor,
                csr.rows(),
                csr.cols(),
                csr.nnz(),
                const_cast<Index*>(csr.rowOffsets()),
                const_cast<Index*>(csr.colIndices()),
                const_cast<Scalar*>(csr.values()),
                CUSPARSE_INDEX_64I,
                CUSPARSE_INDEX_64I,
                CUSPARSE_INDEX_BASE_ZERO,
                scalarDataType<Scalar>()));
            workspace._sparseDescriptor = descriptor;
            workspace._sparseRows = csr.rows();
            workspace._sparseCols = csr.cols();
            workspace._sparseNnz = csr.nnz();
            workspace._descriptorScalarType = scalar_type;
        }
        else
        {
            PLAMATRIX_CHECK_CUSPARSE(cusparseCsrSetPointers(
                asSparseDescriptor(workspace._sparseDescriptor),
                const_cast<Index*>(csr.rowOffsets()),
                const_cast<Index*>(csr.colIndices()),
                const_cast<Scalar*>(csr.values())));
        }
    }

    template <typename Scalar>
    static void prepareVectors(SparseOpsWorkspace& workspace,
                               const DenseMatrix<Scalar, Device::GPU>& input,
                               DenseMatrix<Scalar, Device::GPU>& output)
    {
        const bool recreate = workspace._inputDescriptor == nullptr
            || workspace._outputDescriptor == nullptr || workspace._descriptorsAreMatrices
            || workspace._inputRows != input.rows() || workspace._inputCols != input.cols()
            || workspace._outputRows != output.rows() || workspace._outputCols != output.cols();
        if (recreate)
        {
            if (workspace._inputDescriptor != nullptr)
            {
                if (workspace._descriptorsAreMatrices)
                {
                    static_cast<void>(cusparseDestroyDnMat(
                        asMatrixDescriptor(workspace._inputDescriptor)));
                }
                else
                {
                    static_cast<void>(cusparseDestroyDnVec(
                        asVectorDescriptor(workspace._inputDescriptor)));
                }
                workspace._inputDescriptor = nullptr;
            }
            if (workspace._outputDescriptor != nullptr)
            {
                if (workspace._descriptorsAreMatrices)
                {
                    static_cast<void>(cusparseDestroyDnMat(
                        asMatrixDescriptor(workspace._outputDescriptor)));
                }
                else
                {
                    static_cast<void>(cusparseDestroyDnVec(
                        asVectorDescriptor(workspace._outputDescriptor)));
                }
                workspace._outputDescriptor = nullptr;
            }
            cusparseDnVecDescr_t input_descriptor = nullptr;
            cusparseDnVecDescr_t output_descriptor = nullptr;
            PLAMATRIX_CHECK_CUSPARSE(cusparseCreateDnVec(
                &input_descriptor,
                input.rows(),
                const_cast<Scalar*>(input.data()),
                scalarDataType<Scalar>()));
            try
            {
                PLAMATRIX_CHECK_CUSPARSE(cusparseCreateDnVec(
                    &output_descriptor, output.rows(), output.data(), scalarDataType<Scalar>()));
            }
            catch (...)
            {
                static_cast<void>(cusparseDestroyDnVec(input_descriptor));
                throw;
            }
            workspace._inputDescriptor = input_descriptor;
            workspace._outputDescriptor = output_descriptor;
            workspace._inputRows = input.rows();
            workspace._inputCols = input.cols();
            workspace._outputRows = output.rows();
            workspace._outputCols = output.cols();
            workspace._descriptorsAreMatrices = false;
        }
        else
        {
            PLAMATRIX_CHECK_CUSPARSE(cusparseDnVecSetValues(
                asVectorDescriptor(workspace._inputDescriptor),
                const_cast<Scalar*>(input.data())));
            PLAMATRIX_CHECK_CUSPARSE(cusparseDnVecSetValues(
                asVectorDescriptor(workspace._outputDescriptor), output.data()));
        }
    }

    template <typename Scalar>
    static void prepareMatrices(SparseOpsWorkspace& workspace,
                                const DenseMatrix<Scalar, Device::GPU>& input,
                                DenseMatrix<Scalar, Device::GPU>& output)
    {
        const bool recreate = workspace._inputDescriptor == nullptr
            || workspace._outputDescriptor == nullptr || !workspace._descriptorsAreMatrices
            || workspace._inputRows != input.rows() || workspace._inputCols != input.cols()
            || workspace._outputRows != output.rows() || workspace._outputCols != output.cols();
        if (recreate)
        {
            if (workspace._inputDescriptor != nullptr)
            {
                if (workspace._descriptorsAreMatrices)
                {
                    static_cast<void>(cusparseDestroyDnMat(
                        asMatrixDescriptor(workspace._inputDescriptor)));
                }
                else
                {
                    static_cast<void>(cusparseDestroyDnVec(
                        asVectorDescriptor(workspace._inputDescriptor)));
                }
                workspace._inputDescriptor = nullptr;
            }
            if (workspace._outputDescriptor != nullptr)
            {
                if (workspace._descriptorsAreMatrices)
                {
                    static_cast<void>(cusparseDestroyDnMat(
                        asMatrixDescriptor(workspace._outputDescriptor)));
                }
                else
                {
                    static_cast<void>(cusparseDestroyDnVec(
                        asVectorDescriptor(workspace._outputDescriptor)));
                }
                workspace._outputDescriptor = nullptr;
            }
            cusparseDnMatDescr_t input_descriptor = nullptr;
            cusparseDnMatDescr_t output_descriptor = nullptr;
            PLAMATRIX_CHECK_CUSPARSE(cusparseCreateDnMat(
                &input_descriptor,
                input.rows(),
                input.cols(),
                input.rows(),
                const_cast<Scalar*>(input.data()),
                scalarDataType<Scalar>(),
                CUSPARSE_ORDER_COL));
            try
            {
                PLAMATRIX_CHECK_CUSPARSE(cusparseCreateDnMat(
                    &output_descriptor,
                    output.rows(),
                    output.cols(),
                    output.rows(),
                    output.data(),
                    scalarDataType<Scalar>(),
                    CUSPARSE_ORDER_COL));
            }
            catch (...)
            {
                static_cast<void>(cusparseDestroyDnMat(input_descriptor));
                throw;
            }
            workspace._inputDescriptor = input_descriptor;
            workspace._outputDescriptor = output_descriptor;
            workspace._inputRows = input.rows();
            workspace._inputCols = input.cols();
            workspace._outputRows = output.rows();
            workspace._outputCols = output.cols();
            workspace._descriptorsAreMatrices = true;
        }
        else
        {
            PLAMATRIX_CHECK_CUSPARSE(cusparseDnMatSetValues(
                asMatrixDescriptor(workspace._inputDescriptor),
                const_cast<Scalar*>(input.data())));
            PLAMATRIX_CHECK_CUSPARSE(cusparseDnMatSetValues(
                asMatrixDescriptor(workspace._outputDescriptor), output.data()));
        }
    }

    static cusparseSpMatDescr_t sparseDescriptor(SparseOpsWorkspace& workspace)
    {
        return asSparseDescriptor(workspace._sparseDescriptor);
    }

    static cusparseDnVecDescr_t inputVectorDescriptor(SparseOpsWorkspace& workspace)
    {
        return asVectorDescriptor(workspace._inputDescriptor);
    }

    static cusparseDnVecDescr_t outputVectorDescriptor(SparseOpsWorkspace& workspace)
    {
        return asVectorDescriptor(workspace._outputDescriptor);
    }

    static cusparseDnMatDescr_t inputMatrixDescriptor(SparseOpsWorkspace& workspace)
    {
        return asMatrixDescriptor(workspace._inputDescriptor);
    }

    static cusparseDnMatDescr_t outputMatrixDescriptor(SparseOpsWorkspace& workspace)
    {
        return asMatrixDescriptor(workspace._outputDescriptor);
    }

    static void* buffer(SparseOpsWorkspace& workspace) noexcept
    {
        return workspace._buffer;
    }
};

void SparseOpsWorkspace::destroyDescriptors() noexcept
{
    if (_inputDescriptor != nullptr)
    {
        if (_descriptorsAreMatrices)
        {
            static_cast<void>(cusparseDestroyDnMat(
                asMatrixDescriptor(_inputDescriptor)));
        }
        else
        {
            static_cast<void>(cusparseDestroyDnVec(
                asVectorDescriptor(_inputDescriptor)));
        }
    }
    if (_outputDescriptor != nullptr)
    {
        if (_descriptorsAreMatrices)
        {
            static_cast<void>(cusparseDestroyDnMat(
                asMatrixDescriptor(_outputDescriptor)));
        }
        else
        {
            static_cast<void>(cusparseDestroyDnVec(
                asVectorDescriptor(_outputDescriptor)));
        }
    }
    if (_sparseDescriptor != nullptr)
    {
        static_cast<void>(cusparseDestroySpMat(
            asSparseDescriptor(_sparseDescriptor)));
    }
    _inputDescriptor = nullptr;
    _outputDescriptor = nullptr;
    _sparseDescriptor = nullptr;
    _sparseRows = -1;
    _sparseCols = -1;
    _sparseNnz = -1;
    _inputRows = -1;
    _inputCols = -1;
    _outputRows = -1;
    _outputCols = -1;
    _descriptorScalarType = -1;
    _descriptorsAreMatrices = false;
}

void SparseOpsWorkspace::destroyDescriptorsChecked()
{
    if (_inputDescriptor != nullptr)
    {
        if (_descriptorsAreMatrices)
        {
            PLAMATRIX_CHECK_CUSPARSE(cusparseDestroyDnMat(
                asMatrixDescriptor(_inputDescriptor)));
        }
        else
        {
            PLAMATRIX_CHECK_CUSPARSE(cusparseDestroyDnVec(
                asVectorDescriptor(_inputDescriptor)));
        }
        _inputDescriptor = nullptr;
    }
    if (_outputDescriptor != nullptr)
    {
        if (_descriptorsAreMatrices)
        {
            PLAMATRIX_CHECK_CUSPARSE(cusparseDestroyDnMat(
                asMatrixDescriptor(_outputDescriptor)));
        }
        else
        {
            PLAMATRIX_CHECK_CUSPARSE(cusparseDestroyDnVec(
                asVectorDescriptor(_outputDescriptor)));
        }
        _outputDescriptor = nullptr;
    }
    if (_sparseDescriptor != nullptr)
    {
        PLAMATRIX_CHECK_CUSPARSE(cusparseDestroySpMat(
            asSparseDescriptor(_sparseDescriptor)));
        _sparseDescriptor = nullptr;
    }
    _sparseRows = -1;
    _sparseCols = -1;
    _sparseNnz = -1;
    _inputRows = -1;
    _inputCols = -1;
    _outputRows = -1;
    _outputCols = -1;
    _descriptorScalarType = -1;
    _descriptorsAreMatrices = false;
}

void SparseOpsWorkspace::release() noexcept
{
    bool owner_stream_valid = true;
    if (_hasReuseStream)
    {
        const cudaError_t query = cudaStreamQuery(_reuseStream);
        if (query == cudaErrorNotReady)
        {
            owner_stream_valid = cudaStreamSynchronize(_reuseStream) == cudaSuccess;
        }
        else if (query != cudaSuccess)
        {
            owner_stream_valid = false;
        }
        if (!owner_stream_valid)
        {
            static_cast<void>(cudaGetLastError());
            static_cast<void>(cudaDeviceSynchronize());
        }
    }
    if (_buffer != nullptr)
    {
        if (_streamOrderedAllocation)
        {
            if (owner_stream_valid
                && cudaFreeAsync(_buffer, _allocationStream) == cudaSuccess)
            {
                // Stream-ordered release does not require a host wait once prior work is complete.
            }
            else
            {
                static_cast<void>(cudaGetLastError());
                if (cudaFreeAsync(_buffer, nullptr) == cudaSuccess)
                {
                    static_cast<void>(cudaStreamSynchronize(nullptr));
                }
                else
                {
                    static_cast<void>(cudaGetLastError());
                    static_cast<void>(cudaFree(_buffer));
                }
            }
        }
        else
        {
            static_cast<void>(cudaFree(_buffer));
        }
    }
    destroyDescriptors();
    if (_handle != nullptr)
    {
        static_cast<void>(cusparseDestroy(asHandle(_handle)));
    }
    _capacityBytes = 0;
    _buffer = nullptr;
    _handle = nullptr;
    _allocationStream = nullptr;
    _reuseStream = nullptr;
    _streamOrderedAllocation = false;
    _hasReuseStream = false;
    _hasStatusBatch = false;
    _statusOutput = nullptr;
    _statusFinalize = nullptr;
}

SparseOpsWorkspace::~SparseOpsWorkspace() noexcept
{
    release();
}

SparseOpsWorkspace::SparseOpsWorkspace(SparseOpsWorkspace&& other) noexcept
{
    *this = std::move(other);
}

SparseOpsWorkspace& SparseOpsWorkspace::operator=(SparseOpsWorkspace&& other) noexcept
{
    if (this != &other)
    {
        release();
        _capacityBytes = other._capacityBytes;
        _buffer = other._buffer;
        _handle = other._handle;
        _sparseDescriptor = other._sparseDescriptor;
        _inputDescriptor = other._inputDescriptor;
        _outputDescriptor = other._outputDescriptor;
        _allocationStream = other._allocationStream;
        _reuseStream = other._reuseStream;
        _streamOrderedAllocation = other._streamOrderedAllocation;
        _hasReuseStream = other._hasReuseStream;
        _hasStatusBatch = other._hasStatusBatch;
        _statusOutput = other._statusOutput;
        _statusFinalize = other._statusFinalize;
        _sparseRows = other._sparseRows;
        _sparseCols = other._sparseCols;
        _sparseNnz = other._sparseNnz;
        _inputRows = other._inputRows;
        _inputCols = other._inputCols;
        _outputRows = other._outputRows;
        _outputCols = other._outputCols;
        _descriptorScalarType = other._descriptorScalarType;
        _descriptorsAreMatrices = other._descriptorsAreMatrices;

        other._capacityBytes = 0;
        other._buffer = nullptr;
        other._handle = nullptr;
        other._sparseDescriptor = nullptr;
        other._inputDescriptor = nullptr;
        other._outputDescriptor = nullptr;
        other._allocationStream = nullptr;
        other._reuseStream = nullptr;
        other._streamOrderedAllocation = false;
        other._hasReuseStream = false;
        other._hasStatusBatch = false;
        other._statusOutput = nullptr;
        other._statusFinalize = nullptr;
        other._sparseRows = -1;
        other._sparseCols = -1;
        other._sparseNnz = -1;
        other._inputRows = -1;
        other._inputCols = -1;
        other._outputRows = -1;
        other._outputCols = -1;
        other._descriptorScalarType = -1;
        other._descriptorsAreMatrices = false;
    }
    return *this;
}

void SparseOpsWorkspace::closeAsyncAllocation()
{
    if (_hasStatusBatch)
    {
        throw std::logic_error(
            "SparseOpsWorkspace::closeAsyncAllocation requires checkStatus first");
    }
    if (_hasReuseStream)
    {
        const cudaError_t query = cudaStreamQuery(_reuseStream);
        if (query == cudaErrorNotReady)
        {
            throw std::logic_error(
                "SparseOpsWorkspace::closeAsyncAllocation requires owner stream synchronization");
        }
        PLAMATRIX_CHECK_CUDA(query);
    }
    if (_buffer != nullptr)
    {
        if (!_streamOrderedAllocation)
        {
            throw std::logic_error(
                "SparseOpsWorkspace::closeAsyncAllocation requires stream-ordered storage");
        }
        PLAMATRIX_CHECK_CUDA(cudaFreeAsync(_buffer, _allocationStream));
        _buffer = nullptr;
    }
    destroyDescriptorsChecked();
    if (_handle != nullptr)
    {
        PLAMATRIX_CHECK_CUSPARSE(cusparseDestroy(asHandle(_handle)));
        _handle = nullptr;
    }
    _capacityBytes = 0;
    _allocationStream = nullptr;
    _reuseStream = nullptr;
    _streamOrderedAllocation = false;
    _hasReuseStream = false;
    _hasStatusBatch = false;
    _statusOutput = nullptr;
    _statusFinalize = nullptr;
}

template <typename Scalar>
void spmvAsync(const CSRMatrix<Scalar, Device::GPU>& csr,
               const DenseMatrix<Scalar, Device::GPU>& input,
               DenseMatrix<Scalar, Device::GPU>& output,
               SparseOpsWorkspace& workspace,
               cudaStream_t stream)
{
    checkSpmvDimensions(csr, input, output);
    checkStorageStream("spmvAsync CSR", csr, stream);
    checkCsrStructureStream("spmvAsync CSR", csr, stream);
    checkStorageStream("spmvAsync input", input, stream);
    checkStorageStream("spmvAsync output", output, stream);
    SparseOpsWorkspaceAccess::bindStream(workspace, stream);
    if (output.size() == 0)
    {
        return;
    }
    if (csr.nnz() == 0)
    {
        PLAMATRIX_CHECK_CUDA(cudaMemsetAsync(
            output.data(), 0, static_cast<std::size_t>(output.size()) * sizeof(Scalar), stream));
        return;
    }

    const Scalar alpha = Scalar{1};
    const Scalar beta = Scalar{0};
    const cusparseHandle_t handle = SparseOpsWorkspaceAccess::handle(workspace, stream);
    SparseOpsWorkspaceAccess::prepareSparse(workspace, csr);
    SparseOpsWorkspaceAccess::prepareVectors(workspace, input, output);
    std::size_t temporary_bytes = 0;
    PLAMATRIX_CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        SparseOpsWorkspaceAccess::sparseDescriptor(workspace),
        SparseOpsWorkspaceAccess::inputVectorDescriptor(workspace),
        &beta,
        SparseOpsWorkspaceAccess::outputVectorDescriptor(workspace),
        scalarDataType<Scalar>(),
        CUSPARSE_SPMV_ALG_DEFAULT,
        &temporary_bytes));
    SparseOpsWorkspaceAccess::reserveAsync(workspace, temporary_bytes, stream);
    PLAMATRIX_CHECK_CUSPARSE(cusparseSpMV(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        SparseOpsWorkspaceAccess::sparseDescriptor(workspace),
        SparseOpsWorkspaceAccess::inputVectorDescriptor(workspace),
        &beta,
        SparseOpsWorkspaceAccess::outputVectorDescriptor(workspace),
        scalarDataType<Scalar>(),
        CUSPARSE_SPMV_ALG_DEFAULT,
        SparseOpsWorkspaceAccess::buffer(workspace)));
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> spmv(
    const CSRMatrix<Scalar, Device::GPU>& csr,
    const DenseMatrix<Scalar, Device::GPU>& input)
{
    DenseMatrix<Scalar, Device::GPU> output(csr.rows(), 1);
    spmv(csr, input, output);
    return output;
}

template <typename Scalar>
void spmv(const CSRMatrix<Scalar, Device::GPU>& csr,
          const DenseMatrix<Scalar, Device::GPU>& input,
          DenseMatrix<Scalar, Device::GPU>& output)
{
    SparseOpsWorkspace workspace;
    spmvAsync(csr, input, output, workspace, nullptr);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(nullptr));
}

template <typename Scalar>
void spmmAsync(const CSRMatrix<Scalar, Device::GPU>& csr,
               const DenseMatrix<Scalar, Device::GPU>& input,
               DenseMatrix<Scalar, Device::GPU>& output,
               SparseOpsWorkspace& workspace,
               cudaStream_t stream)
{
    checkSpmmDimensions(csr, input, output);
    checkStorageStream("spmmAsync CSR", csr, stream);
    checkCsrStructureStream("spmmAsync CSR", csr, stream);
    checkStorageStream("spmmAsync input", input, stream);
    checkStorageStream("spmmAsync output", output, stream);
    SparseOpsWorkspaceAccess::bindStream(workspace, stream);
    if (output.size() == 0)
    {
        return;
    }
    if (csr.nnz() == 0)
    {
        PLAMATRIX_CHECK_CUDA(cudaMemsetAsync(
            output.data(), 0, static_cast<std::size_t>(output.size()) * sizeof(Scalar), stream));
        return;
    }

    const Scalar alpha = Scalar{1};
    const Scalar beta = Scalar{0};
    const cusparseHandle_t handle = SparseOpsWorkspaceAccess::handle(workspace, stream);
    SparseOpsWorkspaceAccess::prepareSparse(workspace, csr);
    SparseOpsWorkspaceAccess::prepareMatrices(workspace, input, output);
    std::size_t temporary_bytes = 0;
    PLAMATRIX_CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        SparseOpsWorkspaceAccess::sparseDescriptor(workspace),
        SparseOpsWorkspaceAccess::inputMatrixDescriptor(workspace),
        &beta,
        SparseOpsWorkspaceAccess::outputMatrixDescriptor(workspace),
        scalarDataType<Scalar>(),
        CUSPARSE_SPMM_ALG_DEFAULT,
        &temporary_bytes));
    SparseOpsWorkspaceAccess::reserveAsync(workspace, temporary_bytes, stream);
    PLAMATRIX_CHECK_CUSPARSE(cusparseSpMM(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        SparseOpsWorkspaceAccess::sparseDescriptor(workspace),
        SparseOpsWorkspaceAccess::inputMatrixDescriptor(workspace),
        &beta,
        SparseOpsWorkspaceAccess::outputMatrixDescriptor(workspace),
        scalarDataType<Scalar>(),
        CUSPARSE_SPMM_ALG_DEFAULT,
        SparseOpsWorkspaceAccess::buffer(workspace)));
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> spmm(
    const CSRMatrix<Scalar, Device::GPU>& csr,
    const DenseMatrix<Scalar, Device::GPU>& input)
{
    DenseMatrix<Scalar, Device::GPU> output(csr.rows(), input.cols());
    spmm(csr, input, output);
    return output;
}

template <typename Scalar>
void spmm(const CSRMatrix<Scalar, Device::GPU>& csr,
          const DenseMatrix<Scalar, Device::GPU>& input,
          DenseMatrix<Scalar, Device::GPU>& output)
{
    SparseOpsWorkspace workspace;
    spmmAsync(csr, input, output, workspace, nullptr);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(nullptr));
}

#define PLAMATRIX_INSTANTIATE_GPU_SPARSE_PRODUCTS(Scalar)                              \
    template DenseMatrix<Scalar, Device::GPU> spmv<Scalar>(                            \
        const CSRMatrix<Scalar, Device::GPU>&,                                         \
        const DenseMatrix<Scalar, Device::GPU>&);                                      \
    template void spmv<Scalar>(const CSRMatrix<Scalar, Device::GPU>&,                  \
                               const DenseMatrix<Scalar, Device::GPU>&,                 \
                               DenseMatrix<Scalar, Device::GPU>&);                      \
    template void spmvAsync<Scalar>(const CSRMatrix<Scalar, Device::GPU>&,             \
                                    const DenseMatrix<Scalar, Device::GPU>&,            \
                                    DenseMatrix<Scalar, Device::GPU>&,                  \
                                    SparseOpsWorkspace&,                               \
                                    cudaStream_t);                                     \
    template DenseMatrix<Scalar, Device::GPU> spmm<Scalar>(                            \
        const CSRMatrix<Scalar, Device::GPU>&,                                         \
        const DenseMatrix<Scalar, Device::GPU>&);                                      \
    template void spmm<Scalar>(const CSRMatrix<Scalar, Device::GPU>&,                  \
                               const DenseMatrix<Scalar, Device::GPU>&,                 \
                               DenseMatrix<Scalar, Device::GPU>&);                      \
    template void spmmAsync<Scalar>(const CSRMatrix<Scalar, Device::GPU>&,             \
                                    const DenseMatrix<Scalar, Device::GPU>&,            \
                                    DenseMatrix<Scalar, Device::GPU>&,                  \
                                    SparseOpsWorkspace&,                               \
                                    cudaStream_t)

#ifdef PLAMATRIX_USE_FLOAT
PLAMATRIX_INSTANTIATE_GPU_SPARSE_PRODUCTS(float);
#endif
#ifdef PLAMATRIX_USE_DOUBLE
PLAMATRIX_INSTANTIATE_GPU_SPARSE_PRODUCTS(double);
#endif

#undef PLAMATRIX_INSTANTIATE_GPU_SPARSE_PRODUCTS

} // namespace plamatrix
