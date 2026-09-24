cmake_minimum_required(VERSION 3.18)
project(PlaMatrixPublicApiContract LANGUAGES CXX)

if(NOT IS_DIRECTORY "${PLAMATRIX_INCLUDE_DIR}")
    message(FATAL_ERROR "PLAMATRIX_INCLUDE_DIR must name the library include directory")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_REQUIRED_INCLUDES "${PLAMATRIX_INCLUDE_DIR}")
set(CMAKE_REQUIRED_DEFINITIONS -DPLAMATRIX_NO_CUDA=1)
include(CheckCXXSourceCompiles)

# First prove that the public header works, so missing dependencies cannot make
# the negative checks below pass accidentally.
unset(public_header_works CACHE)
check_cxx_source_compiles([=[
#include <plamatrix/plamatrix.h>
#include <type_traits>
#include <utility>
template <typename T, typename = void> struct HasCsr : std::false_type {};
template <typename T> struct HasCsr<T, std::void_t<decltype(std::declval<T>().csr())>> : std::true_type {};
template <typename T, typename = void> struct HasEntries : std::false_type {};
template <typename T> struct HasEntries<T, std::void_t<decltype(std::declval<T>().entries())>> : std::true_type {};
template <typename T, typename = void> struct HasExecutionInfo : std::false_type {};
template <typename T> struct HasExecutionInfo<T, std::void_t<decltype(std::declval<T>().executionInfo())>> : std::true_type {};
template <typename T, typename = void> struct HasResidency : std::false_type {};
template <typename T> struct HasResidency<T, std::void_t<decltype(std::declval<T>().isDeviceResident())>> : std::true_type {};
template <typename T, typename = void> struct HasView : std::false_type {};
template <typename T> struct HasView<T, std::void_t<decltype(std::declval<T>().view())>> : std::true_type {};
template <typename T, typename = void> struct HasTransposeView : std::false_type {};
template <typename T> struct HasTransposeView<T, std::void_t<decltype(std::declval<T>().transposeView())>> : std::true_type {};
template <typename T, typename = void> struct HasScaled : std::false_type {};
template <typename T> struct HasScaled<T, std::void_t<decltype(std::declval<T>().scaled(1.0))>> : std::true_type {};
template <typename T, typename = void> struct HasAdd : std::false_type {};
template <typename T> struct HasAdd<T, std::void_t<decltype(std::declval<T>().add(std::declval<T>()))>> : std::true_type {};
template <typename T, typename = void> struct HasSubtract : std::false_type {};
template <typename T> struct HasSubtract<T, std::void_t<decltype(std::declval<T>().subtract(std::declval<T>()))>> : std::true_type {};
template <typename T, typename = void> struct HasPendingDownload : std::false_type {};
template <typename T> struct HasPendingDownload<T, std::void_t<decltype(std::declval<T>().pendingHostDownloadBytes())>> : std::true_type {};
struct UnaryFunctor { double operator()(double value) const { return value; } };
struct BinaryFunctor { double operator()(double left, double right) const { return left + right; } };
using Matrix = plamatrix::Matrix3d;
using ComplexMatrix = plamatrix::Matrix<std::complex<double>, 3, 3>;
using Expression = decltype(std::declval<Matrix>() + std::declval<Matrix>());
using Sparse = plamatrix::SparseMatrix<double>;
using SparseMap = plamatrix::Map<Sparse>;
using SparseView = decltype(std::declval<const Matrix&>().sparseView());
using SparseLu = plamatrix::SparseLU<Sparse>;
using SparseQr = plamatrix::SparseQR<Sparse, plamatrix::COLAMDOrdering<int>>;
using SparseLlt = plamatrix::SimplicialLLT<Sparse>;
using SparseLdlt = plamatrix::SimplicialLDLT<Sparse>;
using SparseCg = plamatrix::ConjugateGradient<Sparse>;
using SparseBicg = plamatrix::BiCGSTAB<Sparse>;
using IncompleteCholesky = plamatrix::IncompleteCholesky<double>;
using IncompleteLut = plamatrix::IncompleteLUT<double>;
using RowMajorMatrix = plamatrix::Matrix<double,
                                         plamatrix::Dynamic,
                                         plamatrix::Dynamic,
                                         plamatrix::RowMajor,
                                         8,
                                         8>;
using DynamicStride = plamatrix::Stride<plamatrix::Dynamic, plamatrix::Dynamic>;
using StridedMap = plamatrix::Map<RowMajorMatrix, plamatrix::Unaligned, DynamicStride>;
using MutableRef = plamatrix::Ref<RowMajorMatrix>;
using DynamicBlock = plamatrix::Block<Matrix>;
using FixedBlock = plamatrix::Block<Matrix, 2, 2>;
using VectorSegment = plamatrix::VectorBlock<plamatrix::Vector3d, 2>;
using MatrixDiagonal = plamatrix::Diagonal<Matrix>;
using MatrixUpper = plamatrix::TriangularView<Matrix, plamatrix::Upper>;
using MatrixSelfAdjoint = plamatrix::SelfAdjointView<Matrix, plamatrix::Lower>;
using MatrixReshaped = plamatrix::Reshaped<Matrix, 1, 9>;
using MatrixLinearized = plamatrix::Reshaped<Matrix, 9, 1>;
using DynamicReshaped = plamatrix::Reshaped<Matrix>;
using MatrixReplicated = plamatrix::Replicate<Matrix, 2, 3>;
using MatrixReversed = plamatrix::Reverse<Matrix, plamatrix::BothDirections>;
using MatrixTranspose = plamatrix::Transpose<Matrix>;
using UnaryExpression = plamatrix::CwiseUnaryOp<UnaryFunctor, const Matrix>;
using BinaryExpression = plamatrix::CwiseBinaryOp<BinaryFunctor, const Matrix, const Matrix>;
using AbsExpression = plamatrix::CwiseUnaryOp<plamatrix::internal::scalar_abs_op<double>, const Matrix>;
using Quaternion = plamatrix::Quaterniond;
using AngleAxis = plamatrix::AngleAxisd;
using Isometry = plamatrix::Isometry3d;
using Affine = plamatrix::Affine3d;
using Translation = plamatrix::Translation3d;
static_assert(Matrix::RowsAtCompileTime == 3);
static_assert(RowMajorMatrix::Options == plamatrix::RowMajor);
static_assert(RowMajorMatrix::MaxRowsAtCompileTime == 8);
static_assert(std::is_base_of_v<plamatrix::EigenBase<Matrix>, Matrix>);
static_assert(std::is_base_of_v<plamatrix::DenseBase<Matrix>, Matrix>);
static_assert(std::is_base_of_v<plamatrix::MatrixBase<Matrix>, Matrix>);
static_assert(std::is_base_of_v<plamatrix::MatrixBase<StridedMap>, StridedMap>);
static_assert(std::is_base_of_v<plamatrix::MatrixBase<MutableRef>, MutableRef>);
static_assert(std::is_base_of_v<plamatrix::MatrixBase<DynamicBlock>, DynamicBlock>);
static_assert(std::is_base_of_v<plamatrix::MatrixBase<plamatrix::Block<plamatrix::Vector3d, 2, 1>>,
                                VectorSegment>);
static_assert(std::is_base_of_v<plamatrix::MatrixBase<MatrixDiagonal>, MatrixDiagonal>);
static_assert(std::is_base_of_v<plamatrix::MatrixBase<MatrixUpper>, MatrixUpper>);
static_assert(std::is_base_of_v<plamatrix::MatrixBase<MatrixSelfAdjoint>, MatrixSelfAdjoint>);
static_assert(std::is_base_of_v<plamatrix::MatrixBase<MatrixReshaped>, MatrixReshaped>);
static_assert(std::is_base_of_v<plamatrix::MatrixBase<MatrixReplicated>, MatrixReplicated>);
static_assert(std::is_base_of_v<plamatrix::MatrixBase<MatrixReversed>, MatrixReversed>);
static_assert(std::is_base_of_v<plamatrix::MatrixBase<MatrixTranspose>, MatrixTranspose>);
static_assert(std::is_base_of_v<plamatrix::MatrixBase<UnaryExpression>, UnaryExpression>);
static_assert(std::is_base_of_v<plamatrix::MatrixBase<BinaryExpression>, BinaryExpression>);
static_assert(std::is_base_of_v<plamatrix::QuaternionBase<Quaternion>, Quaternion>);
static_assert(std::is_base_of_v<plamatrix::RotationBase<AngleAxis, 3>, AngleAxis>);
static_assert(std::is_same_v<Isometry, plamatrix::Transform<double, 3, plamatrix::Isometry>>);
static_assert(std::is_same_v<Affine, plamatrix::Transform<double, 3, plamatrix::Affine>>);
static_assert(std::is_same_v<Translation, plamatrix::Translation<double, 3>>);
static_assert(std::is_same_v<decltype(std::declval<const Quaternion&>().toRotationMatrix()), plamatrix::Matrix3d>);
static_assert(std::is_same_v<decltype(std::declval<const plamatrix::Matrix3d&>().canonicalEulerAngles(2, 1, 0)),
                             plamatrix::Vector3d>);
static_assert(std::is_same_v<decltype(std::declval<const Isometry&>().rotation()), Isometry::ConstLinearPart>);
static_assert(std::is_same_v<decltype(std::declval<const Affine&>().rotation()), plamatrix::Matrix3d>);
static_assert(std::is_same_v<decltype(std::declval<Matrix&>().block(0, 0, 2, 2)), DynamicBlock>);
static_assert(std::is_same_v<decltype(std::declval<Matrix&>().template block<2, 2>(0, 0)), FixedBlock>);
static_assert(std::is_same_v<decltype(std::declval<Matrix&>().transpose()), MatrixTranspose>);
static_assert(std::is_same_v<decltype(std::declval<plamatrix::Vector3d&>().template segment<2>(0)), VectorSegment>);
static_assert(std::is_same_v<decltype(std::declval<Matrix&>().diagonal()), MatrixDiagonal>);
static_assert(std::is_same_v<decltype(std::declval<const Matrix&>().diagonal()),
                             const plamatrix::Diagonal<const Matrix>>);
static_assert(std::is_same_v<decltype(std::declval<const plamatrix::Vector3d&>().asDiagonal()),
                             const plamatrix::DiagonalWrapper<const plamatrix::Vector3d>>);
static_assert(std::is_same_v<decltype(std::declval<Matrix&>().template triangularView<plamatrix::Upper>()), MatrixUpper>);
static_assert(std::is_same_v<decltype(std::declval<Matrix&>().template selfadjointView<plamatrix::Lower>()),
                             MatrixSelfAdjoint>);
static_assert(std::is_same_v<decltype(std::declval<Matrix&>().template replicate<2, 3>()),
                             const MatrixReplicated>);
static_assert(std::is_same_v<decltype(std::declval<Matrix&>().reverse()), MatrixReversed>);
static_assert(std::is_same_v<decltype(std::declval<Matrix&>().reshaped()), MatrixLinearized>);
static_assert(std::is_same_v<decltype(std::declval<Matrix&>().reshaped(3, 3)), DynamicReshaped>);
static_assert(std::is_same_v<decltype(std::declval<Matrix&>().transposeInPlace()), void>);
static_assert(std::is_same_v<decltype(std::declval<Matrix&&>().transpose()), MatrixTranspose>);
static_assert(std::is_same_v<decltype(std::declval<FixedBlock&&>().transpose()),
                             plamatrix::Transpose<FixedBlock>>);
static_assert(std::is_same_v<decltype(std::declval<Expression&&>().transpose()),
                             plamatrix::Transpose<Expression>>);
static_assert(std::is_same_v<decltype(std::declval<const Matrix&>().unaryExpr(UnaryFunctor{})),
                             const UnaryExpression>);
static_assert(std::is_same_v<decltype(std::declval<const Matrix&>().binaryExpr(
                                 std::declval<const Matrix&>(), BinaryFunctor{})),
                             const BinaryExpression>);
static_assert(std::is_same_v<decltype(std::declval<const Matrix&>().cwiseAbs()), const AbsExpression>);
static_assert(std::is_same_v<decltype(std::declval<const Matrix&>().prod()), double>);
static_assert(std::is_same_v<decltype(std::declval<const Matrix&>().count()), plamatrix::Index>);
static_assert(std::is_same_v<decltype(std::declval<const Matrix&>().template lpNorm<plamatrix::Infinity>()), double>);
static_assert(std::is_same_v<decltype(std::declval<const Matrix&>().minCoeff(
                                 static_cast<plamatrix::Index*>(nullptr),
                                 static_cast<plamatrix::Index*>(nullptr))),
                             double>);
static_assert(std::is_base_of_v<plamatrix::SparseSolverBase<SparseLu>, SparseLu>);
static_assert(std::is_base_of_v<plamatrix::SparseSolverBase<SparseQr>, SparseQr>);
static_assert(std::is_base_of_v<plamatrix::SparseSolverBase<SparseLlt>, SparseLlt>);
static_assert(std::is_base_of_v<plamatrix::SparseSolverBase<SparseLdlt>, SparseLdlt>);
static_assert(std::is_base_of_v<plamatrix::SparseSolverBase<SparseCg>, SparseCg>);
static_assert(std::is_base_of_v<plamatrix::SparseSolverBase<SparseBicg>, SparseBicg>);
static_assert(std::is_base_of_v<plamatrix::SparseMatrixBase<Sparse>, Sparse>);
static_assert(std::is_base_of_v<plamatrix::SparseMatrixBase<SparseMap>, SparseMap>);
static_assert(std::is_base_of_v<plamatrix::SparseMatrixBase<SparseView>, SparseView>);
static_assert(std::is_same_v<decltype(std::declval<Sparse&>().valuePtr()), double*>);
static_assert(std::is_same_v<decltype(std::declval<Sparse&>().innerIndexPtr()), int*>);
static_assert(std::is_same_v<decltype(std::declval<Sparse&>().outerIndexPtr()), int*>);
static_assert(std::is_same_v<typename plamatrix::ScalarBinaryOpTraits<float, double>::ReturnType, double>);
static_assert(std::is_same_v<typename decltype(std::declval<const plamatrix::Matrix2f&>() +
                                               std::declval<const plamatrix::Matrix2d&>())::Scalar,
                             double>);
static_assert(std::is_same_v<decltype(std::declval<const Matrix&>().partialPivLu().rcond()), double>);
static_assert(std::is_same_v<decltype(std::declval<const Matrix&>().completeOrthogonalDecomposition()),
                             plamatrix::CompleteOrthogonalDecomposition<Matrix>>);
static_assert(std::is_default_constructible_v<plamatrix::HessenbergDecomposition<Matrix>>);
static_assert(std::is_default_constructible_v<plamatrix::Tridiagonalization<Matrix>>);
static_assert(std::is_default_constructible_v<plamatrix::RealSchur<Matrix>>);
static_assert(std::is_default_constructible_v<plamatrix::ComplexSchur<ComplexMatrix>>);
static_assert(std::is_same_v<decltype(std::declval<plamatrix::RealSchur<Matrix>&>().computeFromHessenberg(
                                 std::declval<const Matrix&>(), std::declval<const Matrix&>(), true)),
                             plamatrix::RealSchur<Matrix>&>);
static_assert(std::is_default_constructible_v<plamatrix::ComplexEigenSolver<ComplexMatrix>>);
static_assert(std::is_same_v<typename plamatrix::NumTraits<std::complex<double>>::Real, double>);
static_assert(plamatrix::NumTraits<std::complex<double>>::IsComplex);
static_assert(std::is_same_v<typename decltype(std::declval<const ComplexMatrix&>().conjugate())::Scalar,
                             std::complex<double>>);
static_assert(std::is_same_v<typename decltype(std::declval<const ComplexMatrix&>().adjoint())::Scalar,
                             std::complex<double>>);
static_assert(std::is_same_v<decltype(std::declval<const ComplexMatrix&>().norm()), double>);
static_assert(std::is_default_constructible_v<IncompleteCholesky>);
static_assert(std::is_default_constructible_v<IncompleteLut>);
static_assert(std::is_default_constructible_v<plamatrix::AMDOrdering<int>>);
static_assert(std::is_default_constructible_v<plamatrix::COLAMDOrdering<int>>);
static_assert(std::is_default_constructible_v<plamatrix::NaturalOrdering<int>>);
static_assert(!HasCsr<Sparse>::value && !HasEntries<Sparse>::value);
static_assert(!HasExecutionInfo<Matrix>::value && !HasResidency<Matrix>::value);
static_assert(!HasExecutionInfo<Expression>::value && !HasResidency<Expression>::value);
static_assert(!HasView<Matrix>::value);
static_assert(!HasTransposeView<Matrix>::value);
static_assert(!HasScaled<Matrix>::value);
static_assert(!HasAdd<Matrix>::value);
static_assert(!HasSubtract<Matrix>::value);
static_assert(!HasPendingDownload<Matrix>::value);
int main() { return 0; }
]=] public_header_works)
if(NOT public_header_works)
    message(FATAL_ERROR "Public Eigen-style header or implementation-access boundary is broken")
endif()

set(_private_types
    "DenseMatrix"
    "CSRMatrix"
    "COOMatrix"
    "Vec3"
    "ResidentMatrix"
    "ResidentVector"
    "ExecutionContext"
    "ExecutionPolicy"
    "Device"
    "DeviceMatrix"
    "DenseStorage"
    "DeviceStorage"
    "CsrStorage"
    "CooStorage"
    "PackedVector3"
    "Int32Key3"
    "MatrixView"
    "FixedMatrixView"
    "ResidentCsrMatrix"
    "GroupingWorkspace"
    "BlockNormalEquations"
)
foreach(_type IN LISTS _private_types)
    unset(private_type_leaks CACHE)
    check_cxx_source_compiles(
        "#include <plamatrix/plamatrix.h>\nusing plamatrix::${_type};\nint main() { return 0; }"
        private_type_leaks)
    if(private_type_leaks)
        message(FATAL_ERROR "Internal or removed type leaked into the public API: ${_type}")
    endif()
endforeach()

# Removed implementation headers must not return as forwarding shims.
set(_removed_headers
    "plamatrix/ops/grouping.h"
    "plamatrix/ops/decomposition.h"
    "plamatrix/ops/gemm.h"
    "plamatrix/ops/reduction.h"
    "plamatrix/ops/small_matrix.h"
    "plamatrix/ops/indexing.h"
    "plamatrix/ops/statistics.h"
    "plamatrix/ops/vector.h"
    "plamatrix/ops/solver.h"
    "plamatrix/ops/point_cloud.h"
    "plamatrix/device/device_csr_matrix.h"
    "plamatrix/device/device_matrix.h"
    "plamatrix/device/device_vector.h"
    "plamatrix/core/api.h"
    "plamatrix/core/event.h"
    "plamatrix/core/execution_context.h"
    "plamatrix/core/parallel.h"
    "plamatrix/core/scalar_traits.h"
    "plamatrix/core/layout.h"
    "plamatrix/core/cuda_buffer.h"
    "plamatrix/core/device_matrix.h"
    "plamatrix/core/error.h"
    "plamatrix/core/memory_resource.h"
    "plamatrix/core/diagnostics.h"
    "plamatrix/core/cancellation.h"
    "plamatrix/core/error_check.h"
    "plamatrix/core/allocator.h"
    "plamatrix/core/shape.h"
    "plamatrix/core/capabilities.h"
    "plamatrix/core/no_cuda_stubs.h"
    "plamatrix/core/execution_policy.h"
    "plamatrix/core/memory_space.h"
    "plamatrix/core/checked_math.h"
    "plamatrix/core/backend.h"
    "plamatrix/opencl/execution.h"
    "plamatrix/opencl/native.h"
    "plamatrix/opencl/runtime.h"
    "plamatrix/opencl/iterative_solver.h"
    "plamatrix/optimization/levenberg_marquardt.h"
    "plamatrix/optimization/robust_loss.h"
    "plamatrix/optimization/block_schur.h"
    "plamatrix/dense/elementwise.h"
    "plamatrix/dense/dense_matrix.h"
    "plamatrix/dense/dense_ops.h"
    "plamatrix/dense/matrix_view.h"
    "plamatrix/vulkan/execution.h"
    "plamatrix/vulkan/native.h"
    "plamatrix/vulkan/runtime.h"
    "plamatrix/vulkan/iterative_solver.h"
    "plamatrix/vulkan/cooperative_matrix.h"
    "plamatrix/sparse/sparse_ops.h"
    "plamatrix/sparse/coo_matrix.h"
    "plamatrix/sparse/csr_matrix.h"
    "plamatrix/sparse/iterative_solver.h"
    "plamatrix/cuda/native.h"
    "plamatrix/device/detail/resident_vulkan.h"
    "plamatrix/device/detail/resident_opencl.h"
    "plamatrix/device/detail/resident_buffer.h"
    "plamatrix/dense/detail/auto_backend.h"
)
foreach(_header IN LISTS _removed_headers)
    if(EXISTS "${PLAMATRIX_INCLUDE_DIR}/${_header}")
        message(FATAL_ERROR "Removed public header still exists: ${_header}")
    endif()
endforeach()
