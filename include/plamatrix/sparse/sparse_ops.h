#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "plamatrix/dense/dense_matrix.h"
#include "plamatrix/sparse/csr_matrix.h"

namespace plamatrix
{

/// Reusable cuSPARSE descriptors and temporary storage for GPU sparse products.
/// The workspace is move-only, grow-only, and not thread-safe. Async reuse is restricted to the
/// stream that first uses the workspace until closeAsyncAllocation() releases its storage. The
/// owning stream must be synchronized before closeAsyncAllocation() or cross-stream reuse.
class SparseOpsWorkspace
{
public:
    SparseOpsWorkspace() noexcept = default;

#ifdef PLAMATRIX_WITH_CUDA
    ~SparseOpsWorkspace() noexcept;
    SparseOpsWorkspace(SparseOpsWorkspace&& other) noexcept;
    SparseOpsWorkspace& operator=(SparseOpsWorkspace&& other) noexcept;
    void closeAsyncAllocation();
    void checkStatus(const char* operation);
#else
    ~SparseOpsWorkspace() noexcept = default;
    SparseOpsWorkspace(SparseOpsWorkspace&& other) noexcept = default;
    SparseOpsWorkspace& operator=(SparseOpsWorkspace&& other) noexcept = default;
    void closeAsyncAllocation() noexcept {}
    void checkStatus(const char*)
    {
        throw std::runtime_error(
            "SparseOpsWorkspace::checkStatus requires PLAMATRIX_WITH_CUDA=ON");
    }
#endif

    SparseOpsWorkspace(const SparseOpsWorkspace&) = delete;
    SparseOpsWorkspace& operator=(const SparseOpsWorkspace&) = delete;

    std::size_t capacityBytes() const noexcept { return _capacityBytes; }

private:
    friend struct SparseOpsWorkspaceAccess;
    friend struct SparseCooWorkspaceAccess;

#ifdef PLAMATRIX_WITH_CUDA
    void destroyDescriptors() noexcept;
    void destroyDescriptorsChecked();
    void release() noexcept;
#endif

    std::size_t _capacityBytes = 0;
    void* _buffer = nullptr;
    void* _handle = nullptr;
    void* _sparseDescriptor = nullptr;
    void* _inputDescriptor = nullptr;
    void* _outputDescriptor = nullptr;
    cudaStream_t _allocationStream = nullptr;
    cudaStream_t _reuseStream = nullptr;
    bool _streamOrderedAllocation = false;
    bool _hasReuseStream = false;
    bool _hasStatusBatch = false;
    void* _statusOutput = nullptr;
    void (*_statusFinalize)(void*, bool) noexcept = nullptr;
    Index _sparseRows = -1;
    Index _sparseCols = -1;
    Index _sparseNnz = -1;
    Index _inputRows = -1;
    Index _inputCols = -1;
    Index _outputRows = -1;
    Index _outputCols = -1;
    int _descriptorScalarType = -1;
    bool _descriptorsAreMatrices = false;
};

/// Convert CPU COO triplets to sorted, duplicate-combined CSR storage.
template <typename Scalar>
CSRMatrix<Scalar, Device::CPU> cooToCsr(Index rows,
                                        Index cols,
                                        const std::vector<Index>& row_indices,
                                        const std::vector<Index>& col_indices,
                                        const std::vector<Scalar>& values);

/// Convert CPU COO triplets into an existing CSR matrix.
template <typename Scalar>
void cooToCsr(Index rows,
              Index cols,
              const std::vector<Index>& row_indices,
              const std::vector<Index>& col_indices,
              const std::vector<Scalar>& values,
              CSRMatrix<Scalar, Device::CPU>& output);

/// Synchronously convert device COO triplets to sorted, duplicate-combined GPU CSR.
/// Device triplets are not modified. This allocating overload synchronizes while obtaining the
/// exact combined nnz; use cooToCsrAsync with preallocated output for a fully asynchronous path.
template <typename Scalar>
CSRMatrix<Scalar, Device::GPU> cooToCsr(
    Index rows,
    Index cols,
    const DenseMatrix<Index, Device::GPU>& row_indices,
    const DenseMatrix<Index, Device::GPU>& col_indices,
    const DenseMatrix<Scalar, Device::GPU>& values,
    SparseOpsWorkspace& workspace);

/// Enqueue device COO-to-CSR conversion without host synchronization.
/// Output must have matching dimensions and exactly the duplicate-combined nnz. Synchronize the
/// stream, then call workspace.checkStatus() before consuming output.
template <typename Scalar>
void cooToCsrAsync(
    Index rows,
    Index cols,
    const DenseMatrix<Index, Device::GPU>& row_indices,
    const DenseMatrix<Index, Device::GPU>& col_indices,
    const DenseMatrix<Scalar, Device::GPU>& values,
    CSRMatrix<Scalar, Device::GPU>& output,
    SparseOpsWorkspace& workspace,
    cudaStream_t stream = nullptr);

/// Multiply a CPU CSR matrix by a dense column vector.
template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> spmv(
    const CSRMatrix<Scalar, Device::CPU>& csr,
    const DenseMatrix<Scalar, Device::CPU>& x);

/// Multiply a CPU CSR matrix by a dense column vector into existing storage.
template <typename Scalar>
void spmv(const CSRMatrix<Scalar, Device::CPU>& csr,
          const DenseMatrix<Scalar, Device::CPU>& x,
          DenseMatrix<Scalar, Device::CPU>& output);

/// Multiply a CPU CSR matrix by a column-major dense matrix.
template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> spmm(
    const CSRMatrix<Scalar, Device::CPU>& csr,
    const DenseMatrix<Scalar, Device::CPU>& B);

/// Multiply a CPU CSR matrix by a column-major dense matrix into existing storage.
template <typename Scalar>
void spmm(const CSRMatrix<Scalar, Device::CPU>& csr,
          const DenseMatrix<Scalar, Device::CPU>& B,
          DenseMatrix<Scalar, Device::CPU>& output);

/// Synchronously multiply a GPU CSR matrix by a dense column vector.
template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> spmv(
    const CSRMatrix<Scalar, Device::GPU>& csr,
    const DenseMatrix<Scalar, Device::GPU>& x);

/// Synchronously multiply into existing GPU output storage.
template <typename Scalar>
void spmv(const CSRMatrix<Scalar, Device::GPU>& csr,
          const DenseMatrix<Scalar, Device::GPU>& x,
          DenseMatrix<Scalar, Device::GPU>& output);

/// Enqueue a GPU CSR-vector product on stream without host synchronization. CSR, input, output,
/// workspace, and stream must remain alive until the stream completes.
template <typename Scalar>
void spmvAsync(const CSRMatrix<Scalar, Device::GPU>& csr,
               const DenseMatrix<Scalar, Device::GPU>& x,
               DenseMatrix<Scalar, Device::GPU>& output,
               SparseOpsWorkspace& workspace,
               cudaStream_t stream = nullptr);

/// Synchronously multiply a GPU CSR matrix by a column-major dense matrix.
template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> spmm(
    const CSRMatrix<Scalar, Device::GPU>& csr,
    const DenseMatrix<Scalar, Device::GPU>& B);

/// Synchronously multiply into existing column-major GPU output storage.
template <typename Scalar>
void spmm(const CSRMatrix<Scalar, Device::GPU>& csr,
          const DenseMatrix<Scalar, Device::GPU>& B,
          DenseMatrix<Scalar, Device::GPU>& output);

/// Enqueue a GPU CSR-dense matrix product on stream without host synchronization. CSR, input,
/// output, workspace, and stream must remain alive until the stream completes.
template <typename Scalar>
void spmmAsync(const CSRMatrix<Scalar, Device::GPU>& csr,
               const DenseMatrix<Scalar, Device::GPU>& B,
               DenseMatrix<Scalar, Device::GPU>& output,
               SparseOpsWorkspace& workspace,
               cudaStream_t stream = nullptr);

} // namespace plamatrix
