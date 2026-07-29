#include "iterative_solver_cuda_detail.h"
#include "iterative_solver_test_hooks.h"

#include <atomic>
#include <stdexcept>
#include <utility>

namespace plamatrix
{
namespace
{

#ifdef PLAMATRIX_ITERATIVE_SOLVER_TEST_HOOKS
std::atomic<int> forced_allocation_failure_after{-1};

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> allocateWorkspaceMatrixForTest(
    Index rows, cudaStream_t stream)
{
    int remaining = forced_allocation_failure_after.load(std::memory_order_relaxed);
    while (remaining >= 0)
    {
        if (remaining == 0)
        {
            throw std::runtime_error("forced iterative solver workspace allocation failure");
        }
        if (forced_allocation_failure_after.compare_exchange_weak(
                remaining, remaining - 1, std::memory_order_relaxed))
        {
            break;
        }
    }
    return DenseMatrix<Scalar, Device::GPU>::uninitializedAsync(rows, 1, stream);
}
#endif

template <typename Scalar>
void closeMatrix(DenseMatrix<Scalar, Device::GPU>& matrix)
{
    matrix.closeAsyncAllocation();
}

} // anonymous namespace

#ifdef PLAMATRIX_ITERATIVE_SOLVER_TEST_HOOKS
void iterative_solver_detail::setForcedWorkspaceAllocationFailureAfter(
    int successful_allocations) noexcept
{
    forced_allocation_failure_after.store(successful_allocations, std::memory_order_relaxed);
}

DenseMatrix<float, Device::GPU> IterativeSolverWorkspaceAccess::allocateFloatForTest(
    Index rows, cudaStream_t stream)
{
    return allocateWorkspaceMatrixForTest<float>(rows, stream);
}

DenseMatrix<double, Device::GPU> IterativeSolverWorkspaceAccess::allocateDoubleForTest(
    Index rows, cudaStream_t stream)
{
    return allocateWorkspaceMatrixForTest<double>(rows, stream);
}
#endif

template <typename Scalar>
IterativeSolverWorkspace<Scalar>::~IterativeSolverWorkspace() noexcept
{
    if (_hasReuseStream)
    {
        static_cast<void>(cudaStreamSynchronize(_reuseStream));
    }
    try { closeAsyncAllocation(); } catch (...) {}
}

template <typename Scalar>
IterativeSolverWorkspace<Scalar>::IterativeSolverWorkspace(
    IterativeSolverWorkspace&& other) noexcept
    : _residual(std::move(other._residual))
    , _direction(std::move(other._direction))
    , _transformed(std::move(other._transformed))
    , _matrixDirection(std::move(other._matrixDirection))
    , _inverseDiagonal(std::move(other._inverseDiagonal))
    , _scalars(std::move(other._scalars))
    , _sparseWorkspace(std::move(other._sparseWorkspace))
    , _capacitySize(other._capacitySize)
    , _blasHandle(other._blasHandle)
    , _reuseStream(other._reuseStream)
    , _hasReuseStream(other._hasReuseStream)
{
    other._capacitySize = 0;
    other._blasHandle = nullptr;
    other._reuseStream = nullptr;
    other._hasReuseStream = false;
}

template <typename Scalar>
IterativeSolverWorkspace<Scalar>& IterativeSolverWorkspace<Scalar>::operator=(
    IterativeSolverWorkspace&& other) noexcept
{
    if (this != &other)
    {
        if (_hasReuseStream)
        {
            static_cast<void>(cudaStreamSynchronize(_reuseStream));
        }
        try { closeAsyncAllocation(); } catch (...) {}
        _residual = std::move(other._residual);
        _direction = std::move(other._direction);
        _transformed = std::move(other._transformed);
        _matrixDirection = std::move(other._matrixDirection);
        _inverseDiagonal = std::move(other._inverseDiagonal);
        _scalars = std::move(other._scalars);
        _sparseWorkspace = std::move(other._sparseWorkspace);
        _capacitySize = other._capacitySize;
        _blasHandle = other._blasHandle;
        _reuseStream = other._reuseStream;
        _hasReuseStream = other._hasReuseStream;
        other._capacitySize = 0;
        other._blasHandle = nullptr;
        other._reuseStream = nullptr;
        other._hasReuseStream = false;
    }
    return *this;
}

template <typename Scalar>
void IterativeSolverWorkspace<Scalar>::closeAsyncAllocation()
{
    if (_hasReuseStream)
    {
        const cudaError_t status = cudaStreamQuery(_reuseStream);
        if (status == cudaErrorNotReady)
        {
            throw std::logic_error(
                "IterativeSolverWorkspace::closeAsyncAllocation requires stream synchronization");
        }
        PLAMATRIX_CHECK_CUDA(status);
    }
    _sparseWorkspace.closeAsyncAllocation();
    closeMatrix(_residual);
    closeMatrix(_direction);
    closeMatrix(_transformed);
    closeMatrix(_matrixDirection);
    closeMatrix(_inverseDiagonal);
    closeMatrix(_scalars);
    if (_blasHandle != nullptr)
    {
        PLAMATRIX_CHECK_CUBLAS(cublasDestroy(static_cast<cublasHandle_t>(_blasHandle)));
    }
    _capacitySize = 0;
    _blasHandle = nullptr;
    _reuseStream = nullptr;
    _hasReuseStream = false;
}

AsyncIterativeSolverState::~AsyncIterativeSolverState() noexcept
{
    if (_completionEvent != nullptr)
    {
        static_cast<void>(cudaStreamSynchronize(_stream));
    }
    try { closeAsyncAllocation(); } catch (...) {}
}

AsyncIterativeSolverState::AsyncIterativeSolverState(
    AsyncIterativeSolverState&& other) noexcept
    : submittedIterations(other.submittedIterations)
    , initialResidualSquared(std::move(other.initialResidualSquared))
    , finalResidualSquared(std::move(other.finalResidualSquared))
    , _completionEvent(other._completionEvent)
    , _stream(other._stream)
{
    other.submittedIterations = 0;
    other._completionEvent = nullptr;
    other._stream = nullptr;
}

AsyncIterativeSolverState& AsyncIterativeSolverState::operator=(
    AsyncIterativeSolverState&& other) noexcept
{
    if (this != &other)
    {
        if (_completionEvent != nullptr)
        {
            static_cast<void>(cudaStreamSynchronize(_stream));
        }
        try { closeAsyncAllocation(); } catch (...) {}
        submittedIterations = other.submittedIterations;
        initialResidualSquared = std::move(other.initialResidualSquared);
        finalResidualSquared = std::move(other.finalResidualSquared);
        _completionEvent = other._completionEvent;
        _stream = other._stream;
        other.submittedIterations = 0;
        other._completionEvent = nullptr;
        other._stream = nullptr;
    }
    return *this;
}

void AsyncIterativeSolverState::closeAsyncAllocation()
{
    if (_completionEvent != nullptr)
    {
        const cudaError_t status = cudaEventQuery(static_cast<cudaEvent_t>(_completionEvent));
        if (status == cudaErrorNotReady)
        {
            throw std::logic_error(
                "AsyncIterativeSolverState::closeAsyncAllocation requires stream completion");
        }
        PLAMATRIX_CHECK_CUDA(status);
    }
    initialResidualSquared.closeAsyncAllocation();
    finalResidualSquared.closeAsyncAllocation();
    if (_completionEvent != nullptr)
    {
        PLAMATRIX_CHECK_CUDA(cudaEventDestroy(static_cast<cudaEvent_t>(_completionEvent)));
    }
    _completionEvent = nullptr;
    _stream = nullptr;
}

#ifdef PLAMATRIX_USE_FLOAT
template class IterativeSolverWorkspace<float>;
#endif
#ifdef PLAMATRIX_USE_DOUBLE
template class IterativeSolverWorkspace<double>;
#endif

} // namespace plamatrix
