#pragma once

#include "plamatrix/core/types.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace plamatrix::block_schur_detail
{

template <typename Scalar>
Scalar dotProduct(const std::vector<Scalar>& left, const std::vector<Scalar>& right)
{
    Scalar result = Scalar(0);
    for (std::size_t index = 0; index < left.size(); ++index)
    {
        result += left[index] * right[index];
    }
    return result;
}

template <typename Scalar>
Scalar vectorNorm(const std::vector<Scalar>& vector)
{
    return std::sqrt(std::max(Scalar(0), dotProduct(vector, vector)));
}

template <typename Scalar>
bool invertPositiveDefinite(const Scalar* matrix, Index size, std::vector<Scalar>* inverse)
{
    std::vector<Scalar> lower(static_cast<std::size_t>(size * size), Scalar(0));
    for (Index row = 0; row < size; ++row)
    {
        for (Index col = 0; col <= row; ++col)
        {
            Scalar value = matrix[row * size + col];
            for (Index inner = 0; inner < col; ++inner)
            {
                value -= lower[static_cast<std::size_t>(row * size + inner)] *
                         lower[static_cast<std::size_t>(col * size + inner)];
            }
            if (row == col)
            {
                if (!(value > Scalar(0)) || !std::isfinite(value))
                {
                    return false;
                }
                lower[static_cast<std::size_t>(row * size + col)] = std::sqrt(value);
            }
            else
            {
                const Scalar diagonal = lower[static_cast<std::size_t>(col * size + col)];
                lower[static_cast<std::size_t>(row * size + col)] = value / diagonal;
            }
        }
    }

    inverse->assign(static_cast<std::size_t>(size * size), Scalar(0));
    std::vector<Scalar> intermediate(static_cast<std::size_t>(size), Scalar(0));
    for (Index column = 0; column < size; ++column)
    {
        for (Index row = 0; row < size; ++row)
        {
            Scalar value = row == column ? Scalar(1) : Scalar(0);
            for (Index inner = 0; inner < row; ++inner)
            {
                value -= lower[static_cast<std::size_t>(row * size + inner)] *
                         intermediate[static_cast<std::size_t>(inner)];
            }
            intermediate[static_cast<std::size_t>(row)] =
                value / lower[static_cast<std::size_t>(row * size + row)];
        }
        for (Index offset = 0; offset < size; ++offset)
        {
            const Index row = size - 1 - offset;
            Scalar value = intermediate[static_cast<std::size_t>(row)];
            for (Index inner = row + 1; inner < size; ++inner)
            {
                value -= lower[static_cast<std::size_t>(inner * size + row)] *
                         (*inverse)[static_cast<std::size_t>(inner * size + column)];
            }
            (*inverse)[static_cast<std::size_t>(row * size + column)] =
                value / lower[static_cast<std::size_t>(row * size + row)];
        }
    }
    return true;
}

template <typename Scalar>
bool invertPositiveDefiniteInto(const Scalar* matrix,
                                Index size,
                                Scalar* inverse,
                                Scalar* lower,
                                Scalar* intermediate)
{
    std::fill(lower, lower + size * size, Scalar(0));
    for (Index row = 0; row < size; ++row)
    {
        for (Index column = 0; column <= row; ++column)
        {
            Scalar value = matrix[row * size + column];
            for (Index inner = 0; inner < column; ++inner)
            {
                value -= lower[row * size + inner] * lower[column * size + inner];
            }
            if (row == column)
            {
                if (!(value > Scalar(0)) || !std::isfinite(value))
                {
                    return false;
                }
                lower[row * size + column] = std::sqrt(value);
            }
            else
            {
                lower[row * size + column] = value / lower[column * size + column];
            }
        }
    }

    std::fill(inverse, inverse + size * size, Scalar(0));
    for (Index column = 0; column < size; ++column)
    {
        for (Index row = 0; row < size; ++row)
        {
            Scalar value = row == column ? Scalar(1) : Scalar(0);
            for (Index inner = 0; inner < row; ++inner)
            {
                value -= lower[row * size + inner] * intermediate[inner];
            }
            intermediate[row] = value / lower[row * size + row];
        }
        for (Index offset = 0; offset < size; ++offset)
        {
            const Index row = size - 1 - offset;
            Scalar value = intermediate[row];
            for (Index inner = row + 1; inner < size; ++inner)
            {
                value -= lower[inner * size + row] * inverse[inner * size + column];
            }
            inverse[row * size + column] = value / lower[row * size + row];
        }
    }
    return true;
}

template <typename Scalar>
bool invertPositiveDefinite3x3(const Scalar* matrix, Scalar* inverse)
{
    std::array<Scalar, 9> lower{};
    std::array<Scalar, 3> intermediate{};
    return invertPositiveDefiniteInto(
        matrix, 3, inverse, lower.data(), intermediate.data());
}

template <typename Scalar>
void transform9x3(const Scalar* inverse, const Scalar* cross, Scalar* transformed)
{
    for (Index row = 0; row < 9; ++row)
    {
        const Scalar x0 = cross[row * 3];
        const Scalar x1 = cross[row * 3 + 1];
        const Scalar x2 = cross[row * 3 + 2];
        transformed[row * 3] = inverse[0] * x0 + inverse[1] * x1 + inverse[2] * x2;
        transformed[row * 3 + 1] =
            inverse[3] * x0 + inverse[4] * x1 + inverse[5] * x2;
        transformed[row * 3 + 2] =
            inverse[6] * x0 + inverse[7] * x1 + inverse[8] * x2;
    }
}

template <typename Scalar>
Scalar dot3(const Scalar* left, const Scalar* right)
{
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

template <typename Scalar>
void multiplyMatrixVector(const Scalar* matrix,
                          Index rows,
                          Index cols,
                          const Scalar* vector,
                          Scalar* result)
{
    for (Index row = 0; row < rows; ++row)
    {
        Scalar value = Scalar(0);
        for (Index col = 0; col < cols; ++col)
        {
            value += matrix[row * cols + col] * vector[col];
        }
        result[row] = value;
    }
}

template <typename Scalar>
void addMatrixVector(const Scalar* matrix,
                     Index rows,
                     Index cols,
                     const Scalar* vector,
                     Scalar scale,
                     Scalar* result)
{
    for (Index row = 0; row < rows; ++row)
    {
        Scalar value = Scalar(0);
        for (Index col = 0; col < cols; ++col)
        {
            value += matrix[row * cols + col] * vector[col];
        }
        result[row] += scale * value;
    }
}

template <typename Scalar>
void addTransposeMatrixVector(const Scalar* matrix,
                              Index rows,
                              Index cols,
                              const Scalar* vector,
                              Scalar* result)
{
    for (Index col = 0; col < cols; ++col)
    {
        Scalar value = Scalar(0);
        for (Index row = 0; row < rows; ++row)
        {
            value += matrix[row * cols + col] * vector[row];
        }
        result[col] += value;
    }
}

} // namespace plamatrix::block_schur_detail
