#include "block_schur_device_assembly.h"

#include <stdexcept>

namespace plamatrix::block_schur_detail
{

#ifndef PLAMATRIX_WITH_CUDA
template <typename Scalar>
void copyLastCudaSchurValuesToDevice(
    Scalar*, std::size_t, SchurComplementSolverWorkspace<Scalar>&)
{
    throw std::runtime_error("CUDA Schur handoff requires PLAMATRIX_WITH_CUDA=ON");
}
template <typename Scalar>
std::vector<Scalar> assembleSchurValuesOnCuda(
    Index, Index,
    const std::vector<Scalar>&, const std::vector<Scalar>&,
    const std::vector<Scalar>&, const std::vector<Scalar>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&,
    SchurComplementSolverWorkspace<Scalar>&, bool)
{
    throw std::runtime_error("CUDA Schur assembly requires PLAMATRIX_WITH_CUDA=ON");
}

template std::vector<float> assembleSchurValuesOnCuda(
    Index, Index, const std::vector<float>&, const std::vector<float>&,
    const std::vector<float>&, const std::vector<float>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&,
    SchurComplementSolverWorkspace<float>&, bool);
template std::vector<double> assembleSchurValuesOnCuda(
    Index, Index, const std::vector<double>&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<double>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&,
    SchurComplementSolverWorkspace<double>&, bool);
template void copyLastCudaSchurValuesToDevice(
    float*, std::size_t, SchurComplementSolverWorkspace<float>&);
template void copyLastCudaSchurValuesToDevice(
    double*, std::size_t, SchurComplementSolverWorkspace<double>&);
#endif

#ifndef PLAMATRIX_WITH_OPENCL
template <typename Scalar>
std::vector<Scalar> assembleSchurValuesOnOpenCl(
    Index, Index,
    const std::vector<Scalar>&, const std::vector<Scalar>&,
    const std::vector<Scalar>&, const std::vector<Scalar>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&,
    SchurComplementSolverWorkspace<Scalar>&, bool)
{
    throw std::runtime_error("OpenCL Schur assembly requires PLAMATRIX_WITH_OPENCL=ON");
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
#endif

} // namespace plamatrix::block_schur_detail
