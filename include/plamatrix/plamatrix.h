#pragma once

// PlaMatrix — High-performance matrix library for point cloud processing
// This header will aggregate all public headers as features are added.

#include "plamatrix/core/types.h"
#include "plamatrix/core/error_check.h"
#include "plamatrix/core/allocator.h"
#include "plamatrix/core/device_matrix.h"
#include "plamatrix/dense/dense_matrix.h"
#include "plamatrix/dense/dense_ops.h"
#include "plamatrix/ops/gemm.h"
#include "plamatrix/ops/decomposition.h"
#include "plamatrix/ops/solver.h"
#include "plamatrix/sparse/coo_matrix.h"
#include "plamatrix/sparse/csr_matrix.h"
