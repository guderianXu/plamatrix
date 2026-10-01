#include <array>
#include <iostream>

#include <plamatrix/dense/matrix.h>
#include <plamatrix/dense/small_inverse.h>

int main()
{
    const std::array<double, 9> point_hessian{{4.0, 1.0, 2.0, 1.0, 3.0, 0.0, 2.0, 0.0, 5.0}};
    std::array<double, 9> inverse_block{};
    double determinant = 0.0;
    if (!plamatrix::tryInverse3x3RowMajor(point_hessian, &inverse_block, 1.0e-15, &determinant))
    {
        std::cerr << "Point Hessian is singular or non-finite\n";
        return 1;
    }

    plamatrix::Matrix3d matrix;
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            matrix(row, column) = point_hessian[static_cast<std::size_t>(row * 3 + column)];
        }
    }
    plamatrix::Matrix3d inverse_matrix;
    bool invertible = false;
    matrix.computeInverseAndDetWithCheck(inverse_matrix, determinant, invertible, 1.0e-15);
    if (!invertible)
    {
        return 1;
    }
    std::cout << "determinant: " << determinant << ", inverse(0, 0): " << inverse_matrix(0, 0) << '\n';
}
