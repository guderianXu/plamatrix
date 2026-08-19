#include "block_schur_device_assembly.h"

#include <stdexcept>

namespace plamatrix::block_schur_detail
{

#ifndef PLAMATRIX_WITH_CUDA
template <typename Scalar>
std::vector<Scalar> assembleSchurValuesOnCuda(
    Index, Index,
    const std::vector<Scalar>&, const std::vector<Scalar>&,
    const std::vector<Scalar>&, const std::vector<Scalar>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&)
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
    const std::vector<Index>&, const std::vector<Index>&);
template std::vector<double> assembleSchurValuesOnCuda(
    Index, Index, const std::vector<double>&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<double>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&);
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
    const std::vector<Index>&, const std::vector<Index>&)
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
    const std::vector<Index>&, const std::vector<Index>&);
template std::vector<double> assembleSchurValuesOnOpenCl(
    Index, Index, const std::vector<double>&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<double>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&);
#endif

} // namespace plamatrix::block_schur_detail
