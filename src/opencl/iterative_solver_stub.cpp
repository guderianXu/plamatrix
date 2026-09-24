#include "plamatrix/internal/opencl/iterative_solver.h"

#include <stdexcept>

namespace plamatrix::internal
{
namespace opencl
{

template <typename Scalar>
IterativeSolverReport pcg(
    const CsrStorage<Scalar, Device::CPU>&,
    const DenseStorage<Scalar, Device::CPU>&,
    DenseStorage<Scalar, Device::CPU>&,
    const IterativeSolverOptions&)
{
    throw std::runtime_error("OpenCL PCG requires PLAMATRIX_WITH_OPENCL=ON");
}

template <typename Scalar>
IterativeSolverReport blockPcg(
    const CsrStorage<Scalar, Device::CPU>&,
    const DenseStorage<Scalar, Device::CPU>&,
    DenseStorage<Scalar, Device::CPU>&,
    const DenseStorage<Scalar, Device::CPU>&,
    Index,
    const IterativeSolverOptions&)
{
    throw std::runtime_error("OpenCL block PCG requires PLAMATRIX_WITH_OPENCL=ON");
}

#ifdef PLAMATRIX_USE_FLOAT
template IterativeSolverReport pcg<float>(
    const CsrStorage<float, Device::CPU>&, const DenseStorage<float, Device::CPU>&,
    DenseStorage<float, Device::CPU>&, const IterativeSolverOptions&);
template IterativeSolverReport blockPcg<float>(
    const CsrStorage<float, Device::CPU>&, const DenseStorage<float, Device::CPU>&,
    DenseStorage<float, Device::CPU>&, const DenseStorage<float, Device::CPU>&,
    Index, const IterativeSolverOptions&);
#endif
#ifdef PLAMATRIX_USE_DOUBLE
template IterativeSolverReport pcg<double>(
    const CsrStorage<double, Device::CPU>&, const DenseStorage<double, Device::CPU>&,
    DenseStorage<double, Device::CPU>&, const IterativeSolverOptions&);
template IterativeSolverReport blockPcg<double>(
    const CsrStorage<double, Device::CPU>&, const DenseStorage<double, Device::CPU>&,
    DenseStorage<double, Device::CPU>&, const DenseStorage<double, Device::CPU>&,
    Index, const IterativeSolverOptions&);
#endif

} // namespace opencl
} // namespace plamatrix::internal
