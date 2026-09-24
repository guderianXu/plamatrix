#pragma once

#include <type_traits>

#include "plamatrix/core/types.h"

namespace plamatrix::internal
{
    template <typename Scalar> struct scalar_abs_op;
    template <typename Scalar> struct scalar_conjugate_op;
    template <typename LeftScalar, typename RightScalar, int NaNPropagation> struct scalar_min_op;
    template <typename LeftScalar, typename RightScalar, int NaNPropagation> struct scalar_max_op;
    template <typename Scalar, int NaNPropagation> struct scalar_min_constant_op;
    template <typename Scalar, int NaNPropagation> struct scalar_max_constant_op;
} // namespace plamatrix::internal

namespace plamatrix::v1
{
    enum UpLoType
    {
        Lower = 0x1,
        Upper = 0x2,
        UnitDiag = 0x4,
        ZeroDiag = 0x8,
        UnitLower = UnitDiag | Lower,
        UnitUpper = UnitDiag | Upper,
        StrictlyLower = ZeroDiag | Lower,
        StrictlyUpper = ZeroDiag | Upper,
        SelfAdjoint = 0x10,
        Symmetric = 0x20
    };

    enum DirectionType
    {
        Vertical,
        Horizontal,
        BothDirections
    };

    inline constexpr int AutoOrder = 2;

    enum DecompositionOptions
    {
        Pivoting = 0x01,
        NoPivoting = 0x02,
        ComputeFullU = 0x04,
        ComputeThinU = 0x08,
        ComputeFullV = 0x10,
        ComputeThinV = 0x20,
        EigenvaluesOnly = 0x40,
        ComputeEigenvectors = 0x80,
        EigVecMask = EigenvaluesOnly | ComputeEigenvectors,
        Ax_lBx = 0x100,
        ABx_lx = 0x200,
        BAx_lx = 0x400,
        GenEigMask = Ax_lBx | ABx_lx | BAx_lx
    };

    using DefaultPermutationIndex = int;

    template <typename Derived> struct EigenBase;
    template <typename Derived> class DenseBase;
    template <typename Derived> class MatrixBase;
    template <typename Derived> class ArrayBase;

    template <typename Scalar_,
              int Rows_,
              int Cols_,
              int Options_ = AutoAlign | ((Rows_ == 1 && Cols_ != 1) ? RowMajor : ColMajor),
              int MaxRows_ = Rows_,
              int MaxCols_ = Cols_>
    class Matrix;

    template <int OuterStrideAtCompileTime, int InnerStrideAtCompileTime> class Stride;
    template <int Value = Dynamic> class InnerStride;
    template <int Value = Dynamic> class OuterStride;

    template <typename MatrixType, int MapOptions = Unaligned, typename StrideType = Stride<0, 0>> class Map;

    template <typename PlainObjectType,
              int Options = 0,
              typename StrideType =
                  std::conditional_t<PlainObjectType::IsVectorAtCompileTime, InnerStride<1>, OuterStride<>>>
    class Ref;

    template <typename XprType, int BlockRows = Dynamic, int BlockCols = Dynamic, bool InnerPanel = false> class Block;
    template <typename VectorType, int Size = Dynamic> class VectorBlock;
    template <typename MatrixType, int DiagIndex = 0> class Diagonal;
    template <typename DiagonalVectorType> class DiagonalWrapper;
    template <typename MatrixType, unsigned int Mode> class TriangularView;
    template <typename MatrixType, unsigned int UpLo> class SelfAdjointView;
    template <typename XprType, int Rows = Dynamic, int Cols = Dynamic, int Order = ColMajor> class Reshaped;
    template <typename MatrixType, int RowFactor = Dynamic, int ColFactor = Dynamic> class Replicate;
    template <typename MatrixType, int Direction = BothDirections> class Reverse;
    template <typename MatrixType> class Transpose;
    template <typename UnaryOp, typename XprType> class CwiseUnaryOp;
    template <typename BinaryOp, typename LhsType, typename RhsType> class CwiseBinaryOp;
    template <int SizeAtCompileTime = Dynamic,
              int MaxSizeAtCompileTime = SizeAtCompileTime,
              typename StorageIndex = DefaultPermutationIndex>
    class PermutationMatrix;

    template <typename MatrixType, typename PermutationIndex = DefaultPermutationIndex> class PartialPivLU;
    template <typename MatrixType, typename PermutationIndex = DefaultPermutationIndex> class FullPivLU;
    template <typename MatrixType, int UpLo = Lower> class LLT;
    template <typename MatrixType, int UpLo = Lower> class LDLT;
    template <typename MatrixType> class HouseholderQR;
    template <typename MatrixType, typename PermutationIndex = DefaultPermutationIndex> class ColPivHouseholderQR;
    template <typename MatrixType, int Options = 0> class JacobiSVD;
    template <typename MatrixType, int Options = 0> class BDCSVD;
    template <typename MatrixType> class SelfAdjointEigenSolver;
    template <typename MatrixType> class EigenSolver;
    template <typename MatrixType> class GeneralizedSelfAdjointEigenSolver;
    template <typename MatrixType> class GeneralizedEigenSolver;
    template <typename MatrixType> class HessenbergDecomposition;
    template <typename MatrixType> class Tridiagonalization;
    template <typename MatrixType> class RealSchur;
    template <typename MatrixType> class RealQZ;
    template <typename MatrixType> class ComplexSchur;
    template <typename MatrixType> class ComplexEigenSolver;
    template <typename MatrixType> class CompleteOrthogonalDecomposition;
    template <typename Derived, int Dim> class RotationBase;
    template <typename Derived> class QuaternionBase;
    template <typename Scalar, int Options = AutoAlign> class Quaternion;
    template <typename Scalar> class AngleAxis;
    template <typename Scalar, int Dim> class Translation;
    template <typename Scalar, int Dim, int Mode, int Options = AutoAlign> class Transform;
    template <typename Scalar> class UniformScaling;
    template <typename Derived> class SolverBase;
    template <typename Derived> class SparseMatrixBase;
    template <typename MatrixType> class SparseView;
    template <typename Scalar, int Options = ColMajor, typename StorageIndex = int> class SparseMatrix;
    template <typename Scalar, typename StorageIndex = int> class Triplet;
    template <typename Derived> class SparseSolverBase;
    template <typename StorageIndex> class AMDOrdering;
    template <typename StorageIndex> class COLAMDOrdering;
    template <typename StorageIndex> class NaturalOrdering;
    template <typename MatrixType, typename OrderingType = COLAMDOrdering<typename MatrixType::StorageIndex>>
    class SparseLU;
    template <typename MatrixType, typename OrderingType = COLAMDOrdering<typename MatrixType::StorageIndex>>
    class SparseQR;
    template <typename MatrixType,
              int UpLo = Lower,
              typename OrderingType = AMDOrdering<typename MatrixType::StorageIndex>>
    class SimplicialLLT;
    template <typename MatrixType,
              int UpLo = Lower,
              typename OrderingType = AMDOrdering<typename MatrixType::StorageIndex>>
    class SimplicialLDLT;
    template <typename Scalar> class DiagonalPreconditioner;
    class IdentityPreconditioner;
    template <typename Scalar, int UpLo = Lower, typename OrderingType = NaturalOrdering<DefaultPermutationIndex>>
    class IncompleteCholesky;
    template <typename Scalar, typename StorageIndex = DefaultPermutationIndex> class IncompleteLUT;
    template <typename MatrixType,
              int UpLo = Lower,
              typename Preconditioner = DiagonalPreconditioner<typename MatrixType::Scalar>>
    class ConjugateGradient;
    template <typename MatrixType, typename Preconditioner = DiagonalPreconditioner<typename MatrixType::Scalar>>
    class BiCGSTAB;
    template <typename Node> class DenseExpression;
    template <typename Storage> class ArrayExpressionProxy;
    template <bool ByRows, typename Storage> class AxiswiseExpressionProxy;
} // namespace plamatrix::v1
