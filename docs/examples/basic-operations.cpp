// PlaMatrix Eigen-style dense matrix API: build with CMake and link plamatrix::plamatrix.

#include <iostream>

#include <plamatrix/dense/matrix.h>

int main()
{
    plamatrix::Vector3d point(1.0, 2.0, 3.0);
    plamatrix::Matrix3d rotation;
    rotation << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
    plamatrix::Vector3d rotated = rotation * point;

    plamatrix::MatrixXd measurements = plamatrix::MatrixXd::Zero(100, 3);
    plamatrix::VectorXd weights = plamatrix::VectorXd::Ones(100);
    measurements(0, 0) = rotated(0);
    measurements(0, 1) = rotated(1);
    measurements(0, 2) = rotated(2);

    plamatrix::MatrixXd pose = plamatrix::MatrixXd::Identity(4, 4);
    pose.block<3, 3>(0, 0) = rotation;
    plamatrix::VectorXd update = plamatrix::VectorXd::Zero(6);
    update.segment<3>(0) += rotated.normalized();
    const plamatrix::Vector3d solved = rotation.partialPivLu().solve(rotated);
    const plamatrix::Vector3d spd_solved = rotation.ldlt().solve(rotated);
    const plamatrix::SelfAdjointEigenSolver<plamatrix::Matrix3d> eigensolver(rotation);

    plamatrix::MatrixXd gram(3, 3);
    gram.noalias() = measurements.transpose() * measurements;
    double external_point[3] = {1.0, 4.0, 9.0};
    plamatrix::Map<plamatrix::Vector3d> mapped_point(external_point);
    const plamatrix::Vector3d square_roots = mapped_point.array().sqrt();
    const plamatrix::MatrixXd squared = (measurements + measurements).array().square();
    const plamatrix::RowVectorXd column_sums = squared.colwise().sum();
    const auto expression = (rotation * point).transpose() * rotation;
    std::cout << "rotation:\n" << rotation << '\n';
    std::cout << "weighted point: " << weights(0) * rotated(0) << '\n';
    std::cout << "Gram(0,0): " << gram(0, 0) << '\n';
    std::cout << "pose(0,0): " << pose(0, 0) << ", update(0): " << update(0) << ", solved(0): " << solved(0) << '\n';
    std::cout << "SPD solved(0): " << spd_solved(0) << ", column mean: " << measurements.colwise().mean()(0, 0) << '\n';
    std::cout << "smallest eigenvalue: " << eigensolver.eigenvalues()(0) << '\n';
    std::cout << "expression(0,0): " << expression(0, 0) << '\n';
    std::cout << "squared column sum: " << column_sums(0, 0) << '\n';
    std::cout << "mapped square root: " << square_roots(1) << '\n';
}
