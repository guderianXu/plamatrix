#include "plamatrix/sparse/coo_matrix.h"
#include "plamatrix/sparse/csr_matrix.h"

namespace plamatrix
{

// Explicit instantiations for supported scalar types and devices

// CSRMatrix
template class CSRMatrix<float, Device::CPU>;
template class CSRMatrix<float, Device::GPU>;
template class CSRMatrix<double, Device::CPU>;
template class CSRMatrix<double, Device::GPU>;

// COOMatrix
template class COOMatrix<float, Device::CPU>;
template class COOMatrix<float, Device::GPU>;
template class COOMatrix<double, Device::CPU>;
template class COOMatrix<double, Device::GPU>;

} // namespace plamatrix
