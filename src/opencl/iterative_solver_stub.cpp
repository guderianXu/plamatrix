#include "plamatrix/opencl/iterative_solver.h"

#include <stdexcept>

namespace plamatrix
{
namespace opencl
{

template <typename Scalar>
IterativeSolverReport pcg(
    const CSRMatrix<Scalar, Device::CPU>&,
    const DenseMatrix<Scalar, Device::CPU>&,
    DenseMatrix<Scalar, Device::CPU>&,
    const IterativeSolverOptions&)
{
    throw std::runtime_error("OpenCL PCG requires PLAMATRIX_WITH_OPENCL=ON");
}

template <typename Scalar>
IterativeSolverReport blockPcg(
    const CSRMatrix<Scalar, Device::CPU>&,
    const DenseMatrix<Scalar, Device::CPU>&,
    DenseMatrix<Scalar, Device::CPU>&,
    const DenseMatrix<Scalar, Device::CPU>&,
    Index,
    const IterativeSolverOptions&)
{
    throw std::runtime_error("OpenCL block PCG requires PLAMATRIX_WITH_OPENCL=ON");
}

#ifdef PLAMATRIX_USE_FLOAT
template IterativeSolverReport pcg<float>(
    const CSRMatrix<float, Device::CPU>&, const DenseMatrix<float, Device::CPU>&,
    DenseMatrix<float, Device::CPU>&, const IterativeSolverOptions&);
template IterativeSolverReport blockPcg<float>(
    const CSRMatrix<float, Device::CPU>&, const DenseMatrix<float, Device::CPU>&,
    DenseMatrix<float, Device::CPU>&, const DenseMatrix<float, Device::CPU>&,
    Index, const IterativeSolverOptions&);
#endif
#ifdef PLAMATRIX_USE_DOUBLE
template IterativeSolverReport pcg<double>(
    const CSRMatrix<double, Device::CPU>&, const DenseMatrix<double, Device::CPU>&,
    DenseMatrix<double, Device::CPU>&, const IterativeSolverOptions&);
template IterativeSolverReport blockPcg<double>(
    const CSRMatrix<double, Device::CPU>&, const DenseMatrix<double, Device::CPU>&,
    DenseMatrix<double, Device::CPU>&, const DenseMatrix<double, Device::CPU>&,
    Index, const IterativeSolverOptions&);
#endif

} // namespace opencl
} // namespace plamatrix
