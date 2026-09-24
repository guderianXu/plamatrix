#pragma once

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "plamatrix/dense/permutation_matrix.h"

namespace plamatrix::v1
{
    namespace sparse_ordering_detail
    {
        template <typename StorageIndex> StorageIndex checkedIndex(Index index)
        {
            if (index < static_cast<Index>(std::numeric_limits<StorageIndex>::lowest()) ||
                index > static_cast<Index>(std::numeric_limits<StorageIndex>::max()))
            {
                throw std::overflow_error("sparse ordering index does not fit StorageIndex");
            }
            return static_cast<StorageIndex>(index);
        }

        template <typename StorageIndex>
        void minimumDegree(std::vector<std::set<Index>> adjacency,
                           PermutationMatrix<Dynamic, Dynamic, StorageIndex>& permutation)
        {
            const Index size = static_cast<Index>(adjacency.size());
            permutation = PermutationMatrix<Dynamic, Dynamic, StorageIndex>(size);
            std::vector<bool> active(static_cast<std::size_t>(size), true);
            for (Index position = 0; position < size; ++position)
            {
                Index pivot = -1;
                std::size_t best_degree = std::numeric_limits<std::size_t>::max();
                for (Index candidate = 0; candidate < size; ++candidate)
                {
                    if (!active[static_cast<std::size_t>(candidate)])
                    {
                        continue;
                    }
                    std::size_t degree = 0;
                    for (const Index neighbor : adjacency[static_cast<std::size_t>(candidate)])
                    {
                        degree += active[static_cast<std::size_t>(neighbor)] ? 1U : 0U;
                    }
                    if (degree < best_degree)
                    {
                        best_degree = degree;
                        pivot = candidate;
                    }
                }

                permutation.indices()(position) = checkedIndex<StorageIndex>(pivot);
                std::vector<Index> neighbors;
                for (const Index neighbor : adjacency[static_cast<std::size_t>(pivot)])
                {
                    if (active[static_cast<std::size_t>(neighbor)])
                    {
                        neighbors.push_back(neighbor);
                    }
                }
                for (std::size_t left = 0; left < neighbors.size(); ++left)
                {
                    adjacency[static_cast<std::size_t>(neighbors[left])].erase(pivot);
                    for (std::size_t right = left + 1; right < neighbors.size(); ++right)
                    {
                        adjacency[static_cast<std::size_t>(neighbors[left])].insert(neighbors[right]);
                        adjacency[static_cast<std::size_t>(neighbors[right])].insert(neighbors[left]);
                    }
                }
                active[static_cast<std::size_t>(pivot)] = false;
            }
        }

        template <typename MatrixType> std::vector<std::set<Index>> symmetricAdjacency(const MatrixType& matrix)
        {
            if (matrix.rows() != matrix.cols())
            {
                throw std::invalid_argument("AMDOrdering requires a square matrix");
            }
            std::vector<std::set<Index>> adjacency(static_cast<std::size_t>(matrix.cols()));
            for (Index outer = 0; outer < matrix.outerSize(); ++outer)
            {
                for (typename MatrixType::InnerIterator entry(matrix, outer); entry; ++entry)
                {
                    if (entry.row() != entry.col())
                    {
                        adjacency[static_cast<std::size_t>(entry.row())].insert(entry.col());
                        adjacency[static_cast<std::size_t>(entry.col())].insert(entry.row());
                    }
                }
            }
            return adjacency;
        }

        template <typename MatrixType> std::vector<std::set<Index>> columnIntersection(const MatrixType& matrix)
        {
            std::vector<std::vector<Index>> row_columns(static_cast<std::size_t>(matrix.rows()));
            for (Index outer = 0; outer < matrix.outerSize(); ++outer)
            {
                for (typename MatrixType::InnerIterator entry(matrix, outer); entry; ++entry)
                {
                    row_columns[static_cast<std::size_t>(entry.row())].push_back(entry.col());
                }
            }
            std::vector<std::set<Index>> adjacency(static_cast<std::size_t>(matrix.cols()));
            for (auto& columns : row_columns)
            {
                std::sort(columns.begin(), columns.end());
                columns.erase(std::unique(columns.begin(), columns.end()), columns.end());
                for (std::size_t left = 0; left < columns.size(); ++left)
                {
                    for (std::size_t right = left + 1; right < columns.size(); ++right)
                    {
                        adjacency[static_cast<std::size_t>(columns[left])].insert(columns[right]);
                        adjacency[static_cast<std::size_t>(columns[right])].insert(columns[left]);
                    }
                }
            }
            return adjacency;
        }
    } // namespace sparse_ordering_detail

    template <typename StorageIndex> class AMDOrdering
    {
        static_assert(std::is_integral_v<StorageIndex>, "AMDOrdering requires an integral storage index");

    public:
        template <typename MatrixType>
        void operator()(const MatrixType& matrix, PermutationMatrix<Dynamic, Dynamic, StorageIndex>& permutation) const
        {
            sparse_ordering_detail::minimumDegree(sparse_ordering_detail::symmetricAdjacency(matrix), permutation);
        }
    };

    template <typename StorageIndex> class COLAMDOrdering
    {
        static_assert(std::is_integral_v<StorageIndex>, "COLAMDOrdering requires an integral storage index");

    public:
        template <typename MatrixType>
        void operator()(const MatrixType& matrix, PermutationMatrix<Dynamic, Dynamic, StorageIndex>& permutation) const
        {
            sparse_ordering_detail::minimumDegree(sparse_ordering_detail::columnIntersection(matrix), permutation);
        }
    };

    template <typename StorageIndex> class NaturalOrdering
    {
        static_assert(std::is_integral_v<StorageIndex>, "NaturalOrdering requires an integral storage index");

    public:
        template <typename MatrixType>
        void operator()(const MatrixType& matrix, PermutationMatrix<Dynamic, Dynamic, StorageIndex>& permutation) const
        {
            permutation = PermutationMatrix<Dynamic, Dynamic, StorageIndex>(matrix.cols());
        }
    };
} // namespace plamatrix::v1
