#pragma once

// Public Eigen-style numerical interface. Backend implementation headers are private contracts.
#include "plamatrix/core/types.h"
#include "plamatrix/dense/matrix.h"
#include "plamatrix/dense/permutation_matrix.h"
#include "plamatrix/dense/solver_base.h"
#include "plamatrix/geometry/geometry.h"
#include "plamatrix/sparse/ordering_methods.h"
#include "plamatrix/sparse/sparse_matrix.h"
#include "plamatrix/sparse/sparse_solver_base.h"
#include "plamatrix/sparse/preconditioners.h"
#include "plamatrix/sparse/sparse_cholesky.h"
#include "plamatrix/sparse/sparse_lu.h"
#include "plamatrix/sparse/sparse_qr.h"
#include "plamatrix/sparse/conjugate_gradient.h"
#include "plamatrix/sparse/bicgstab.h"
