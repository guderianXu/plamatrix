#pragma once

#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

#include "plamatrix/dense/dense_matrix.h"
#include "plamatrix/sparse/csr_matrix.h"

namespace plamatrix::benchmark
{

    struct BackendFixture
    {
        std::string scenario;
        Index parameter = 0;
        CSRMatrix<float, Device::CPU> matrix;
        DenseMatrix<float, Device::CPU> rhs;

        BackendFixture(std::string scenarioName,
                       Index scenarioParameter,
                       CSRMatrix<float, Device::CPU>&& matrixValue,
                       DenseMatrix<float, Device::CPU>&& rhsValue);
    };

    BackendFixture makeBackendFixture(const std::string& scenario, Index parameter);
    BackendFixture loadMatrixMarketFixture(std::istream& input, std::string label);
    BackendFixture loadMatrixMarketFixture(const std::string& path, std::string label = {});

    std::vector<std::pair<std::string, Index>> backendSuite();

} // namespace plamatrix::benchmark
