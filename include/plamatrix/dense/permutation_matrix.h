#pragma once

#include <numeric>
#include <stdexcept>
#include <vector>

#include "plamatrix/dense/matrix.h"
#include "plamatrix/dense/detail/static_or_dynamic_vector.h"

namespace plamatrix::v1
{
    template <int SizeAtCompileTime, int MaxSizeAtCompileTime, typename StorageIndex> class PermutationMatrix
    {
    public:
        using IndicesType = Matrix<StorageIndex, SizeAtCompileTime, 1>;

        PermutationMatrix() = default;
        explicit PermutationMatrix(Index size) : _indices(size, 1)
        {
            setIdentity();
        }

        template <typename Indices>
        explicit PermutationMatrix(const Indices& indices) : _indices(static_cast<Index>(indices.size()), 1)
        {
            for (Index index = 0; index < _indices.size(); ++index)
            {
                _indices(index) = indices[static_cast<std::size_t>(index)];
            }
            validate();
        }

        void setIdentity()
        {
            for (Index index = 0; index < _indices.size(); ++index)
            {
                _indices(index) = static_cast<StorageIndex>(index);
            }
        }

        void resize(Index size)
        {
            _indices.resize(size, 1);
            setIdentity();
        }

        Index size() const noexcept
        {
            return _indices.size();
        }

        const IndicesType& indices() const noexcept
        {
            return _indices;
        }

        IndicesType& indices() noexcept
        {
            return _indices;
        }

        PermutationMatrix inverse() const
        {
            validate();
            PermutationMatrix result(size());
            for (Index index = 0; index < size(); ++index)
            {
                result._indices(_indices(index)) = static_cast<StorageIndex>(index);
            }
            return result;
        }

        PermutationMatrix transpose() const
        {
            return inverse();
        }

        int determinant() const
        {
            validate();
            int sign = 1;
            for (Index left = 0; left < size(); ++left)
            {
                for (Index right = left + 1; right < size(); ++right)
                {
                    if (_indices(left) > _indices(right))
                    {
                        sign = -sign;
                    }
                }
            }
            return sign;
        }

        bool isValid() const noexcept
        {
            decomposition_detail::StaticOrDynamicVector<unsigned char, SizeAtCompileTime> seen(
                static_cast<std::size_t>(size()), 0);
            for (Index index = 0; index < size(); ++index)
            {
                const auto value = static_cast<Index>(_indices(index));
                if (value < 0 || value >= size() || seen[static_cast<std::size_t>(value)])
                {
                    return false;
                }
                seen[static_cast<std::size_t>(value)] = true;
            }
            return true;
        }

        void validate() const
        {
            if (!isValid())
            {
                throw std::invalid_argument("PermutationMatrix indices do not form a permutation");
            }
        }

        template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
        Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>
        operator*(const Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>& matrix) const
        {
            validate();
            if (matrix.rows() != size())
            {
                throw std::invalid_argument("permutation and matrix row counts differ");
            }
            Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols> result(matrix.rows(), matrix.cols());
            for (Index row = 0; row < size(); ++row)
            {
                for (Index column = 0; column < matrix.cols(); ++column)
                {
                    result(row, column) = matrix(_indices(row), column);
                }
            }
            return result;
        }

    private:
        IndicesType _indices;
    };
} // namespace plamatrix::v1
