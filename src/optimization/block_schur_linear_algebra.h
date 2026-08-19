#pragma once

#include "plamatrix/core/types.h"

#include <algorithm>
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
