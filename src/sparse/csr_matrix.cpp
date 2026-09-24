#include "plamatrix/internal/sparse/coo_storage.h"
#include "plamatrix/internal/sparse/csr_storage.h"

namespace plamatrix::internal
{

// Explicit instantiations for supported scalar types and devices

// CsrStorage
#ifdef PLAMATRIX_USE_FLOAT
template class CsrStorage<float, Device::CPU>;
template class CsrStorage<float, Device::GPU>;
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template class CsrStorage<double, Device::CPU>;
template class CsrStorage<double, Device::GPU>;
#endif

// CooStorage
#ifdef PLAMATRIX_USE_FLOAT
template class CooStorage<float, Device::CPU>;
template class CooStorage<float, Device::GPU>;
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template class CooStorage<double, Device::CPU>;
template class CooStorage<double, Device::GPU>;
#endif

} // namespace plamatrix::internal
