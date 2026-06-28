#include "plamatrix/ops/decomposition.h"

namespace plamatrix
{

template <typename Scalar>
std::tuple<DenseMatrix<Scalar, Device::GPU>, DenseMatrix<Scalar, Device::GPU>, DenseMatrix<Scalar, Device::GPU>>
svd(const DenseMatrix<Scalar, Device::GPU>& A)
{
    auto [U_cpu, S_cpu, Vt_cpu] = svd(A.toCpu());
    return {U_cpu.toGpu(), S_cpu.toGpu(), Vt_cpu.toGpu()};
}

template <typename Scalar>
std::tuple<DenseMatrix<Scalar, Device::GPU>, DenseMatrix<Scalar, Device::GPU>>
qr(const DenseMatrix<Scalar, Device::GPU>& A)
{
    auto [Q_cpu, R_cpu] = qr(A.toCpu());
    return {Q_cpu.toGpu(), R_cpu.toGpu()};
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> eigh(const DenseMatrix<Scalar, Device::GPU>& A)
{
    return eigh(A.toCpu()).toGpu();
}

#ifdef PLAMATRIX_USE_FLOAT
template std::tuple<DenseMatrix<float, Device::GPU>, DenseMatrix<float, Device::GPU>, DenseMatrix<float, Device::GPU>>
svd(const DenseMatrix<float, Device::GPU>&);

template std::tuple<DenseMatrix<float, Device::GPU>, DenseMatrix<float, Device::GPU>>
qr(const DenseMatrix<float, Device::GPU>&);

template DenseMatrix<float, Device::GPU> eigh(const DenseMatrix<float, Device::GPU>&);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template std::tuple<DenseMatrix<double, Device::GPU>, DenseMatrix<double, Device::GPU>, DenseMatrix<double, Device::GPU>>
svd(const DenseMatrix<double, Device::GPU>&);

template std::tuple<DenseMatrix<double, Device::GPU>, DenseMatrix<double, Device::GPU>>
qr(const DenseMatrix<double, Device::GPU>&);

template DenseMatrix<double, Device::GPU> eigh(const DenseMatrix<double, Device::GPU>&);
#endif

} // namespace plamatrix
