#pragma once

#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

#include "plamatrix/internal/dense/dense_storage.h"
#include "plamatrix/internal/sparse/csr_storage.h"

namespace plamatrix::internal::benchmark
{

    struct BackendFixture
    {
        std::string scenario;
        Index parameter = 0;
        CsrStorage<float, Device::CPU> matrix;
        DenseStorage<float, Device::CPU> rhs;

        BackendFixture(std::string scenarioName,
                       Index scenarioParameter,
                       CsrStorage<float, Device::CPU>&& matrixValue,
                       DenseStorage<float, Device::CPU>&& rhsValue);
    };

    BackendFixture makeBackendFixture(const std::string& scenario, Index parameter);
    BackendFixture loadMatrixMarketFixture(std::istream& input, std::string label);
    BackendFixture loadMatrixMarketFixture(const std::string& path, std::string label = {});

    std::vector<std::pair<std::string, Index>> backendSuite();

} // namespace plamatrix::internal::benchmark
