#include "backend_scenarios.h"

#include <sstream>

#include <gtest/gtest.h>

namespace plamatrix::benchmark
{

    TEST(BackendScenarios, BuildsStructuredSparseSystems)
    {
        const auto stencil = makeBackendFixture("stencil2d", 4);
        EXPECT_EQ(stencil.matrix.rows(), 16);
        EXPECT_EQ(stencil.matrix.rows(), stencil.matrix.cols());
        EXPECT_GT(stencil.matrix.nnz(), stencil.matrix.rows());
        EXPECT_EQ(stencil.rhs.rows(), stencil.matrix.rows());
        stencil.matrix.validateStructure();

        const auto ba = makeBackendFixture("ba_schur", 8);
        EXPECT_EQ(ba.matrix.rows(), 48);
        EXPECT_EQ(ba.matrix.rows(), ba.matrix.cols());
        EXPECT_GT(ba.matrix.nnz(), ba.matrix.rows());
        ba.matrix.validateStructure();

        const auto mvs = makeBackendFixture("mvs_visibility", 8);
        EXPECT_EQ(mvs.matrix.rows(), 64);
        EXPECT_EQ(mvs.matrix.rows(), mvs.matrix.cols());
        EXPECT_GT(mvs.matrix.nnz(), mvs.matrix.rows());
        mvs.matrix.validateStructure();
    }

    TEST(BackendScenarios, LoadsSymmetricMatrixMarket)
    {
        std::istringstream input("%%MatrixMarket matrix coordinate real symmetric\n"
                                 "% damped BA Schur sample\n"
                                 "3 3 5\n"
                                 "1 1 4\n"
                                 "2 1 -1\n"
                                 "2 2 4\n"
                                 "3 2 -1\n"
                                 "3 3 4\n");
        const auto fixture = loadMatrixMarketFixture(input, "ba_real");
        EXPECT_EQ(fixture.scenario, "ba_real");
        EXPECT_EQ(fixture.matrix.rows(), 3);
        EXPECT_EQ(fixture.matrix.nnz(), 7);
        EXPECT_FLOAT_EQ(fixture.matrix.values()[0], 4.0f);
        fixture.matrix.validateStructure();
    }

} // namespace plamatrix::benchmark
