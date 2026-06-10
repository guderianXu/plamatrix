#include "plamatrix/sparse/coo_matrix.h"
#include "plamatrix/sparse/csr_matrix.h"

namespace plamatrix
{

// Explicit instantiations for supported scalar types and devices

// CSRMatrix
#ifdef PLAMATRIX_USE_FLOAT
template class CSRMatrix<float, Device::CPU>;
template class CSRMatrix<float, Device::GPU>;
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template class CSRMatrix<double, Device::CPU>;
template class CSRMatrix<double, Device::GPU>;
#endif

// COOMatrix
#ifdef PLAMATRIX_USE_FLOAT
template class COOMatrix<float, Device::CPU>;
template class COOMatrix<float, Device::GPU>;
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template class COOMatrix<double, Device::CPU>;
template class COOMatrix<double, Device::GPU>;
#endif

} // namespace plamatrix
