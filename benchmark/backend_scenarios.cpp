#include "backend_scenarios.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace plamatrix::internal::benchmark
{
    namespace
    {

        using Row = std::vector<std::pair<Index, float>>;

        Index checkedProduct(Index left, Index right, const char* name)
        {
            if (left <= 0 || right <= 0 || left > std::numeric_limits<Index>::max() / right)
                throw std::invalid_argument(std::string(name) + " is too large");
            return left * right;
        }

        struct SparseRows
        {
            Index dimension = 0;
            std::vector<Row> rows;
        };

        void addSymmetricEdge(std::vector<std::vector<Index>>& adjacency, Index row, Index column)
        {
            if (row == column)
                return;
            adjacency[static_cast<std::size_t>(row)].push_back(column);
            adjacency[static_cast<std::size_t>(column)].push_back(row);
        }

        CsrStorage<float, Device::CPU> finalizeMatrix(const SparseRows& sparse)
        {
            Index nnz = 0;
            for (const Row& row : sparse.rows)
                nnz += static_cast<Index>(row.size());
            CsrStorage<float, Device::CPU> matrix(sparse.dimension, sparse.dimension, nnz);
            Index offset = 0;
            for (Index row = 0; row < sparse.dimension; ++row)
            {
                matrix.rowOffsets()[row] = offset;
                for (const auto& [column, value] : sparse.rows[static_cast<std::size_t>(row)])
                {
                    matrix.colIndices()[offset] = column;
                    matrix.values()[offset] = value;
                    ++offset;
                }
            }
            matrix.rowOffsets()[sparse.dimension] = offset;
            matrix.validateStructure();
            return matrix;
        }

        BackendFixture makeFixture(std::string scenario, Index parameter, SparseRows sparse)
        {
            auto matrix = finalizeMatrix(sparse);
            DenseStorage<float, Device::CPU> rhs(sparse.dimension, 1);
            for (Index row = 0; row < sparse.dimension; ++row)
            {
                const float coordinate = static_cast<float>(row + 1);
                rhs(row, 0) = 0.5f + 0.25f * std::sin(coordinate * 0.013f) + 0.125f * std::cos(coordinate * 0.007f);
            }
            return BackendFixture(std::move(scenario), parameter, std::move(matrix), std::move(rhs));
        }

        SparseRows
        makeGraphMatrix(Index dimension, std::vector<std::vector<Index>> adjacency, float offDiagonalMagnitude)
        {
            SparseRows result;
            result.dimension = dimension;
            result.rows.resize(static_cast<std::size_t>(dimension));
            for (Index row = 0; row < dimension; ++row)
            {
                auto& neighbors = adjacency[static_cast<std::size_t>(row)];
                std::sort(neighbors.begin(), neighbors.end());
                neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
                Row& output = result.rows[static_cast<std::size_t>(row)];
                output.reserve(neighbors.size() + 1);
                float diagonal = 1.0f;
                for (const Index column : neighbors)
                {
                    if (column == row)
                        continue;
                    output.emplace_back(column, -offDiagonalMagnitude);
                    diagonal += offDiagonalMagnitude;
                }
                output.emplace_back(row, diagonal);
                std::sort(output.begin(), output.end());
            }
            return result;
        }

        SparseRows makeTridiagonal(Index dimension)
        {
            if (dimension <= 0)
                throw std::invalid_argument("tridiagonal size must be positive");
            SparseRows result;
            result.dimension = dimension;
            result.rows.resize(static_cast<std::size_t>(dimension));
            for (Index row = 0; row < dimension; ++row)
            {
                Row& output = result.rows[static_cast<std::size_t>(row)];
                if (row > 0)
                    output.emplace_back(row - 1, -1.0f);
                output.emplace_back(row, row == 0 || row + 1 == dimension ? 2.0f : 3.0f);
                if (row + 1 < dimension)
                    output.emplace_back(row + 1, -1.0f);
            }
            return result;
        }

        SparseRows makeStencil2d(Index side)
        {
            if (side <= 0)
                throw std::invalid_argument("stencil2d side must be positive");
            const Index dimension = checkedProduct(side, side, "stencil2d side");
            std::vector<std::vector<Index>> adjacency(static_cast<std::size_t>(dimension));
            const auto index = [side](Index x, Index y) { return y * side + x; };
            for (Index y = 0; y < side; ++y)
            {
                for (Index x = 0; x < side; ++x)
                {
                    const Index center = index(x, y);
                    if (x > 0)
                        addSymmetricEdge(adjacency, center, index(x - 1, y));
                    if (y > 0)
                        addSymmetricEdge(adjacency, center, index(x, y - 1));
                    if (x + 1 < side)
                        addSymmetricEdge(adjacency, center, index(x + 1, y));
                    if (y + 1 < side)
                        addSymmetricEdge(adjacency, center, index(x, y + 1));
                }
            }
            return makeGraphMatrix(dimension, std::move(adjacency), 1.0f);
        }

        SparseRows makeStencil3d(Index side)
        {
            if (side <= 0)
                throw std::invalid_argument("stencil3d side must be positive");
            const Index plane = checkedProduct(side, side, "stencil3d side");
            const Index dimension = checkedProduct(plane, side, "stencil3d side");
            std::vector<std::vector<Index>> adjacency(static_cast<std::size_t>(dimension));
            const auto index = [side, plane](Index x, Index y, Index z) { return z * plane + y * side + x; };
            for (Index z = 0; z < side; ++z)
            {
                for (Index y = 0; y < side; ++y)
                {
                    for (Index x = 0; x < side; ++x)
                    {
                        const Index center = index(x, y, z);
                        if (x + 1 < side)
                            addSymmetricEdge(adjacency, center, index(x + 1, y, z));
                        if (y + 1 < side)
                            addSymmetricEdge(adjacency, center, index(x, y + 1, z));
                        if (z + 1 < side)
                            addSymmetricEdge(adjacency, center, index(x, y, z + 1));
                    }
                }
            }
            return makeGraphMatrix(dimension, std::move(adjacency), 1.0f);
        }

        SparseRows makeBaSchur(Index cameraCount)
        {
            if (cameraCount <= 0)
                throw std::invalid_argument("ba_schur camera count must be positive");
            constexpr Index blockSize = 6;
            const Index dimension = checkedProduct(cameraCount, blockSize, "ba_schur camera count");
            std::vector<std::vector<Index>> adjacency(static_cast<std::size_t>(dimension));
            const Index trackCount = checkedProduct(cameraCount, 4, "ba_schur camera count");
            for (Index track = 0; track < trackCount; ++track)
            {
                const Index candidates[] = {
                    track % cameraCount, (track * 7 + 3) % cameraCount, (track * 13 + 5) % cameraCount};
                std::vector<Index> cameras;
                for (const Index camera : candidates)
                    if (std::find(cameras.begin(), cameras.end(), camera) == cameras.end())
                        cameras.push_back(camera);
                for (const Index left : cameras)
                {
                    for (const Index right : cameras)
                    {
                        if (left == right)
                            continue;
                        for (Index localRow = 0; localRow < blockSize; ++localRow)
                            for (Index localColumn = 0; localColumn < blockSize; ++localColumn)
                                addSymmetricEdge(
                                    adjacency, left * blockSize + localRow, right * blockSize + localColumn);
                    }
                }
            }
            return makeGraphMatrix(dimension, std::move(adjacency), 0.02f);
        }

        SparseRows makeMvsVisibility(Index side)
        {
            if (side <= 0)
                throw std::invalid_argument("mvs_visibility side must be positive");
            const Index dimension = checkedProduct(side, side, "mvs_visibility side");
            std::vector<std::vector<Index>> adjacency(static_cast<std::size_t>(dimension));
            const auto index = [side](Index x, Index y) { return y * side + x; };
            for (Index y = 0; y < side; ++y)
            {
                for (Index x = 0; x < side; ++x)
                {
                    const Index center = index(x, y);
                    for (Index dy = -1; dy <= 1; ++dy)
                    {
                        for (Index dx = -1; dx <= 1; ++dx)
                        {
                            if (dx == 0 && dy == 0)
                                continue;
                            const Index nx = x + dx;
                            const Index ny = y + dy;
                            if (nx >= 0 && nx < side && ny >= 0 && ny < side)
                                addSymmetricEdge(adjacency, center, index(nx, ny));
                        }
                    }
                    for (Index view = 1; view <= 3; ++view)
                    {
                        const Index nx = x + ((x * 17 + view * 3) % 5) - 2;
                        const Index ny = y + ((y * 31 + view * 5) % 5) - 2;
                        if (nx >= 0 && nx < side && ny >= 0 && ny < side)
                            addSymmetricEdge(adjacency, center, index(nx, ny));
                    }
                }
            }
            return makeGraphMatrix(dimension, std::move(adjacency), 0.25f);
        }

        std::string_view trim(std::string_view value)
        {
            while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front())))
                value.remove_prefix(1);
            while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())))
                value.remove_suffix(1);
            return value;
        }

    } // namespace

    BackendFixture::BackendFixture(std::string scenarioName,
                                   Index scenarioParameter,
                                   CsrStorage<float, Device::CPU>&& matrixValue,
                                   DenseStorage<float, Device::CPU>&& rhsValue)
        : scenario(std::move(scenarioName)), parameter(scenarioParameter), matrix(std::move(matrixValue)),
          rhs(std::move(rhsValue))
    {
    }

    BackendFixture makeBackendFixture(const std::string& scenario, Index parameter)
    {
        if (scenario == "tridiagonal")
            return makeFixture(scenario, parameter, makeTridiagonal(parameter));
        if (scenario == "stencil2d")
            return makeFixture(scenario, parameter, makeStencil2d(parameter));
        if (scenario == "stencil3d")
            return makeFixture(scenario, parameter, makeStencil3d(parameter));
        if (scenario == "ba_schur")
            return makeFixture(scenario, parameter, makeBaSchur(parameter));
        if (scenario == "mvs_visibility")
            return makeFixture(scenario, parameter, makeMvsVisibility(parameter));
        throw std::invalid_argument("unknown backend benchmark scenario: " + scenario);
    }

    BackendFixture loadMatrixMarketFixture(std::istream& input, std::string label)
    {
        std::string line;
        if (!std::getline(input, line) || line.rfind("%%MatrixMarket", 0) != 0)
            throw std::invalid_argument("MatrixMarket input must start with a MatrixMarket header");
        std::istringstream header(line);
        std::string banner;
        std::string object;
        std::string format;
        std::string field;
        std::string symmetry;
        header >> banner >> object >> format >> field >> symmetry;
        if (object != "matrix" || format != "coordinate" ||
            (field != "real" && field != "integer" && field != "pattern") ||
            (symmetry != "general" && symmetry != "symmetric"))
            throw std::invalid_argument(
                "only real/integer/pattern general or symmetric MatrixMarket input is supported");

        do
        {
            if (!std::getline(input, line))
                throw std::invalid_argument("MatrixMarket input is missing its dimensions");
        } while (trim(line).empty() || trim(line).front() == '%');
        std::istringstream dimensions(line);
        Index rows = 0;
        Index columns = 0;
        Index entries = 0;
        dimensions >> rows >> columns >> entries;
        if (!dimensions || rows <= 0 || rows != columns || entries < 0)
            throw std::invalid_argument("MatrixMarket input must describe a non-empty square matrix");

        std::vector<std::map<Index, float>> accumulated(static_cast<std::size_t>(rows));
        for (Index entry = 0; entry < entries; ++entry)
        {
            if (!std::getline(input, line))
                throw std::invalid_argument("MatrixMarket input ended before all entries were read");
            if (trim(line).empty() || trim(line).front() == '%')
            {
                --entry;
                continue;
            }
            std::istringstream values(line);
            Index row = 0;
            Index column = 0;
            float value = 1.0f;
            values >> row >> column;
            if (field != "pattern")
                values >> value;
            if (!values || row <= 0 || row > rows || column <= 0 || column > columns || !std::isfinite(value))
                throw std::invalid_argument("invalid MatrixMarket entry");
            --row;
            --column;
            accumulated[static_cast<std::size_t>(row)][column] += value;
            if (symmetry == "symmetric" && row != column)
                accumulated[static_cast<std::size_t>(column)][row] += value;
        }

        SparseRows sparse;
        sparse.dimension = rows;
        sparse.rows.resize(static_cast<std::size_t>(rows));
        for (Index row = 0; row < rows; ++row)
        {
            for (const auto& [column, value] : accumulated[static_cast<std::size_t>(row)])
                sparse.rows[static_cast<std::size_t>(row)].emplace_back(column, value);
        }
        if (label.empty())
            label = "matrix_market";
        return makeFixture(std::move(label), rows, std::move(sparse));
    }

    BackendFixture loadMatrixMarketFixture(const std::string& path, std::string label)
    {
        std::ifstream input(path);
        if (!input)
            throw std::runtime_error("failed to open MatrixMarket file: " + path);
        if (label.empty())
            label = path;
        return loadMatrixMarketFixture(input, std::move(label));
    }

    std::vector<std::pair<std::string, Index>> backendSuite()
    {
        return {{"tridiagonal", 4096},
                {"tridiagonal", 16384},
                {"tridiagonal", 65536},
                {"tridiagonal", 262144},
                {"tridiagonal", 1048576},
                {"stencil2d", 64},
                {"stencil2d", 128},
                {"stencil2d", 256},
                {"stencil3d", 16},
                {"stencil3d", 24},
                {"stencil3d", 32},
                {"ba_schur", 32},
                {"ba_schur", 64},
                {"ba_schur", 128},
                {"ba_schur", 256},
                {"mvs_visibility", 64},
                {"mvs_visibility", 128},
                {"mvs_visibility", 256}};
    }

} // namespace plamatrix::internal::benchmark
