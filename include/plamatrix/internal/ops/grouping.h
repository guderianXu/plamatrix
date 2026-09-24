#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "plamatrix/internal/ops/reduction.h"

namespace plamatrix::internal
{

#if defined(__CUDACC__)
#define PLAMATRIX_GROUPING_HOST_DEVICE __host__ __device__
#else
#define PLAMATRIX_GROUPING_HOST_DEVICE
#endif

    /// Stable-ABI three-component signed integer key ordered lexicographically by x, y, then z.
    struct Int32Key3
    {
        std::int32_t x;
        std::int32_t y;
        std::int32_t z;

        PLAMATRIX_GROUPING_HOST_DEVICE constexpr bool operator==(const Int32Key3& other) const noexcept
        {
            return x == other.x && y == other.y && z == other.z;
        }

        PLAMATRIX_GROUPING_HOST_DEVICE constexpr bool operator!=(const Int32Key3& other) const noexcept
        {
            return !(*this == other);
        }

        PLAMATRIX_GROUPING_HOST_DEVICE constexpr bool operator<(const Int32Key3& other) const noexcept
        {
            return x < other.x || (x == other.x && (y < other.y || (y == other.y && z < other.z)));
        }
    };

    static_assert(std::is_standard_layout_v<Int32Key3>, "Int32Key3 must have standard layout");
    static_assert(std::is_trivially_copyable_v<Int32Key3>, "Int32Key3 must be trivially copyable");
    static_assert(sizeof(Int32Key3) == 12, "Int32Key3 ABI must remain exactly 12 bytes");
    static_assert(alignof(Int32Key3) == alignof(std::int32_t), "Int32Key3 alignment must match int32_t");
    static_assert(offsetof(Int32Key3, x) == 0 && offsetof(Int32Key3, y) == 4 && offsetof(Int32Key3, z) == 8,
                  "Int32Key3 field offsets are part of its ABI");

#undef PLAMATRIX_GROUPING_HOST_DEVICE

    enum class GroupReduction
    {
        Sum,
        Minimum,
        Maximum
    };

    template <typename Key, typename Value, Device Dev> struct SortedKeyValueResult
    {
        DenseStorage<Key, Dev> keys;
        DenseStorage<Value, Dev> values;
    };

    /// CPU results are exact-sized. GPU asynchronous results use input-sized capacity arrays;
    /// only the first runCount[0] elements are valid.
    template <typename Key, Device Dev> struct RunLengthEncodeResult
    {
        DenseStorage<Key, Dev> uniqueKeys;
        DenseStorage<Index, Dev> counts;
        DenseStorage<Index, Dev> runCount;
    };

    /// CPU and synchronous GPU results are exact-sized. GPU asynchronous results use input-sized
    /// capacity arrays; only the first runCount[0] elements are valid.
    template <typename Key, typename Value, Device Dev> struct ReduceByKeyResult
    {
        DenseStorage<Key, Dev> uniqueKeys;
        DenseStorage<Value, Dev> aggregates;
        DenseStorage<Index, Dev> runCount;
    };

    using GroupingWorkspace = ReductionWorkspace;

    template <typename Key, typename Value>
    SortedKeyValueResult<Key, Value, Device::CPU> sortByKey(const DenseStorage<Key, Device::CPU>& keys,
                                                            const DenseStorage<Value, Device::CPU>& values);

    template <typename Key>
    RunLengthEncodeResult<Key, Device::CPU> runLengthEncode(const DenseStorage<Key, Device::CPU>& keys);

    template <typename Key, typename Value>
    ReduceByKeyResult<Key, Value, Device::CPU> reduceByKey(const DenseStorage<Key, Device::CPU>& keys,
                                                           const DenseStorage<Value, Device::CPU>& values,
                                                           GroupReduction operation);

    /// Reduce values in the half-open ranges [offsets[i], offsets[i + 1]). Offsets must be a
    /// nondecreasing (S + 1) x 1 vector beginning at zero and ending at values.rows().
    template <typename Value>
    DenseStorage<Value, Device::CPU> segmentedReduce(const DenseStorage<Value, Device::CPU>& values,
                                                    const DenseStorage<Index, Device::CPU>& offsets,
                                                    GroupReduction operation);

#ifdef PLAMATRIX_WITH_CUDA

    template <typename Key, typename Value>
    SortedKeyValueResult<Key, Value, Device::GPU> sortByKey(const DenseStorage<Key, Device::GPU>& keys,
                                                            const DenseStorage<Value, Device::GPU>& values);

    template <typename Key, typename Value>
    SortedKeyValueResult<Key, Value, Device::GPU> sortByKey(const DenseStorage<Key, Device::GPU>& keys,
                                                            const DenseStorage<Value, Device::GPU>& values,
                                                            GroupingWorkspace& workspace,
                                                            cudaStream_t stream = nullptr);

    template <typename Key, typename Value>
    void sortByKey(const DenseStorage<Key, Device::GPU>& keys,
                   const DenseStorage<Value, Device::GPU>& values,
                   DenseStorage<Key, Device::GPU>& sorted_keys,
                   DenseStorage<Value, Device::GPU>& sorted_values,
                   GroupingWorkspace& workspace,
                   cudaStream_t stream = nullptr);

    template <typename Key, typename Value>
    SortedKeyValueResult<Key, Value, Device::GPU> sortByKeyAsync(const DenseStorage<Key, Device::GPU>& keys,
                                                                 const DenseStorage<Value, Device::GPU>& values,
                                                                 GroupingWorkspace& workspace,
                                                                 cudaStream_t stream);

    template <typename Key, typename Value>
    void sortByKeyAsync(const DenseStorage<Key, Device::GPU>& keys,
                        const DenseStorage<Value, Device::GPU>& values,
                        DenseStorage<Key, Device::GPU>& sorted_keys,
                        DenseStorage<Value, Device::GPU>& sorted_values,
                        GroupingWorkspace& workspace,
                        cudaStream_t stream);

    /// Sort external contiguous Int32Key3 GPU keys with int32_t or Index values into caller-owned storage.
    template <typename Value>
    void sortByKey(ConstMatrixView<Int32Key3, Device::GPU> keys,
                   ConstMatrixView<Value, Device::GPU> values,
                   MatrixView<Int32Key3, Device::GPU> sorted_keys,
                   MatrixView<Value, Device::GPU> sorted_values,
                   GroupingWorkspace& workspace,
                   cudaStream_t stream = nullptr);

    template <typename Value>
    void sortByKeyAsync(ConstMatrixView<Int32Key3, Device::GPU> keys,
                        ConstMatrixView<Value, Device::GPU> values,
                        MatrixView<Int32Key3, Device::GPU> sorted_keys,
                        MatrixView<Value, Device::GPU> sorted_values,
                        GroupingWorkspace& workspace,
                        cudaStream_t stream);

    /// Sort caller-owned contiguous GPU column vectors without allocating result matrices.
    /// Supported numeric keys are uint32_t, uint64_t, and Index; values are float, double, or Index.
    template <typename Key, typename Value>
    void sortByKey(ConstMatrixView<Key, Device::GPU> keys,
                   ConstMatrixView<Value, Device::GPU> values,
                   MatrixView<Key, Device::GPU> sorted_keys,
                   MatrixView<Value, Device::GPU> sorted_values,
                   GroupingWorkspace& workspace,
                   cudaStream_t stream = nullptr);

    template <typename Key, typename Value>
    void sortByKeyAsync(ConstMatrixView<Key, Device::GPU> keys,
                        ConstMatrixView<Value, Device::GPU> values,
                        MatrixView<Key, Device::GPU> sorted_keys,
                        MatrixView<Value, Device::GPU> sorted_values,
                        GroupingWorkspace& workspace,
                        cudaStream_t stream);

    template <typename Key>
    RunLengthEncodeResult<Key, Device::GPU> runLengthEncode(const DenseStorage<Key, Device::GPU>& keys);

    template <typename Key>
    RunLengthEncodeResult<Key, Device::GPU> runLengthEncode(const DenseStorage<Key, Device::GPU>& keys,
                                                            GroupingWorkspace& workspace,
                                                            cudaStream_t stream = nullptr);

    template <typename Key>
    void runLengthEncode(const DenseStorage<Key, Device::GPU>& keys,
                         DenseStorage<Key, Device::GPU>& unique_keys,
                         DenseStorage<Index, Device::GPU>& counts,
                         DenseStorage<Index, Device::GPU>& run_count,
                         GroupingWorkspace& workspace,
                         cudaStream_t stream = nullptr);

    template <typename Key>
    RunLengthEncodeResult<Key, Device::GPU>
    runLengthEncodeAsync(const DenseStorage<Key, Device::GPU>& keys, GroupingWorkspace& workspace, cudaStream_t stream);

    template <typename Key>
    void runLengthEncodeAsync(const DenseStorage<Key, Device::GPU>& keys,
                              DenseStorage<Key, Device::GPU>& unique_keys,
                              DenseStorage<Index, Device::GPU>& counts,
                              DenseStorage<Index, Device::GPU>& run_count,
                              GroupingWorkspace& workspace,
                              cudaStream_t stream);

    /// Encode external contiguous Int32Key3 GPU column vectors into caller-owned capacities.
    void runLengthEncode(ConstMatrixView<Int32Key3, Device::GPU> keys,
                         MatrixView<Int32Key3, Device::GPU> unique_keys,
                         MatrixView<Index, Device::GPU> counts,
                         MatrixView<Index, Device::GPU> run_count,
                         GroupingWorkspace& workspace,
                         cudaStream_t stream = nullptr);

    void runLengthEncodeAsync(ConstMatrixView<Int32Key3, Device::GPU> keys,
                              MatrixView<Int32Key3, Device::GPU> unique_keys,
                              MatrixView<Index, Device::GPU> counts,
                              MatrixView<Index, Device::GPU> run_count,
                              GroupingWorkspace& workspace,
                              cudaStream_t stream);

    /// Encode into caller-owned input-sized key/count capacities and a 1 x 1 run count.
    template <typename Key>
    void runLengthEncode(ConstMatrixView<Key, Device::GPU> keys,
                         MatrixView<Key, Device::GPU> unique_keys,
                         MatrixView<Index, Device::GPU> counts,
                         MatrixView<Index, Device::GPU> run_count,
                         GroupingWorkspace& workspace,
                         cudaStream_t stream = nullptr);

    template <typename Key>
    void runLengthEncodeAsync(ConstMatrixView<Key, Device::GPU> keys,
                              MatrixView<Key, Device::GPU> unique_keys,
                              MatrixView<Index, Device::GPU> counts,
                              MatrixView<Index, Device::GPU> run_count,
                              GroupingWorkspace& workspace,
                              cudaStream_t stream);

    template <typename Key, typename Value>
    ReduceByKeyResult<Key, Value, Device::GPU> reduceByKey(const DenseStorage<Key, Device::GPU>& keys,
                                                           const DenseStorage<Value, Device::GPU>& values,
                                                           GroupReduction operation);

    template <typename Key, typename Value>
    ReduceByKeyResult<Key, Value, Device::GPU> reduceByKey(const DenseStorage<Key, Device::GPU>& keys,
                                                           const DenseStorage<Value, Device::GPU>& values,
                                                           GroupReduction operation,
                                                           GroupingWorkspace& workspace,
                                                           cudaStream_t stream = nullptr);

    template <typename Key, typename Value>
    void reduceByKey(const DenseStorage<Key, Device::GPU>& keys,
                     const DenseStorage<Value, Device::GPU>& values,
                     GroupReduction operation,
                     DenseStorage<Key, Device::GPU>& unique_keys,
                     DenseStorage<Value, Device::GPU>& aggregates,
                     DenseStorage<Index, Device::GPU>& run_count,
                     GroupingWorkspace& workspace,
                     cudaStream_t stream = nullptr);

    template <typename Key, typename Value>
    ReduceByKeyResult<Key, Value, Device::GPU> reduceByKeyAsync(const DenseStorage<Key, Device::GPU>& keys,
                                                                const DenseStorage<Value, Device::GPU>& values,
                                                                GroupReduction operation,
                                                                GroupingWorkspace& workspace,
                                                                cudaStream_t stream);

    template <typename Key, typename Value>
    void reduceByKeyAsync(const DenseStorage<Key, Device::GPU>& keys,
                          const DenseStorage<Value, Device::GPU>& values,
                          GroupReduction operation,
                          DenseStorage<Key, Device::GPU>& unique_keys,
                          DenseStorage<Value, Device::GPU>& aggregates,
                          DenseStorage<Index, Device::GPU>& run_count,
                          GroupingWorkspace& workspace,
                          cudaStream_t stream);

    template <typename Value>
    DenseStorage<Value, Device::GPU> segmentedReduce(const DenseStorage<Value, Device::GPU>& values,
                                                    const DenseStorage<Index, Device::GPU>& offsets,
                                                    GroupReduction operation);

    template <typename Value>
    DenseStorage<Value, Device::GPU> segmentedReduce(const DenseStorage<Value, Device::GPU>& values,
                                                    const DenseStorage<Index, Device::GPU>& offsets,
                                                    GroupReduction operation,
                                                    GroupingWorkspace& workspace,
                                                    cudaStream_t stream = nullptr);

    template <typename Value>
    void segmentedReduce(const DenseStorage<Value, Device::GPU>& values,
                         const DenseStorage<Index, Device::GPU>& offsets,
                         GroupReduction operation,
                         DenseStorage<Value, Device::GPU>& output,
                         GroupingWorkspace& workspace,
                         cudaStream_t stream = nullptr);

    template <typename Value>
    DenseStorage<Value, Device::GPU> segmentedReduceAsync(const DenseStorage<Value, Device::GPU>& values,
                                                         const DenseStorage<Index, Device::GPU>& offsets,
                                                         GroupReduction operation,
                                                         GroupingWorkspace& workspace,
                                                         cudaStream_t stream);

    template <typename Value>
    void segmentedReduceAsync(const DenseStorage<Value, Device::GPU>& values,
                              const DenseStorage<Index, Device::GPU>& offsets,
                              GroupReduction operation,
                              DenseStorage<Value, Device::GPU>& output,
                              GroupingWorkspace& workspace,
                              cudaStream_t stream);

    /// Reduce caller-owned contiguous GPU vectors over offsets into caller-owned output.
    template <typename Value>
    void segmentedReduce(ConstMatrixView<Value, Device::GPU> values,
                         ConstMatrixView<Index, Device::GPU> offsets,
                         GroupReduction operation,
                         MatrixView<Value, Device::GPU> output,
                         GroupingWorkspace& workspace,
                         cudaStream_t stream = nullptr);

    template <typename Value>
    void segmentedReduceAsync(ConstMatrixView<Value, Device::GPU> values,
                              ConstMatrixView<Index, Device::GPU> offsets,
                              GroupReduction operation,
                              MatrixView<Value, Device::GPU> output,
                              GroupingWorkspace& workspace,
                              cudaStream_t stream);

#else

    namespace grouping_detail
    {
        [[noreturn]] inline void throwNoCuda(const char* operation)
        {
            throw std::runtime_error(std::string(operation) + " requires PLAMATRIX_WITH_CUDA=ON");
        }
    } // namespace grouping_detail

    template <typename Key, typename Value>
    inline SortedKeyValueResult<Key, Value, Device::GPU> sortByKey(const DenseStorage<Key, Device::GPU>&,
                                                                   const DenseStorage<Value, Device::GPU>&)
    {
        grouping_detail::throwNoCuda("sortByKey");
    }

    template <typename Key, typename Value>
    inline SortedKeyValueResult<Key, Value, Device::GPU> sortByKey(const DenseStorage<Key, Device::GPU>&,
                                                                   const DenseStorage<Value, Device::GPU>&,
                                                                   GroupingWorkspace&,
                                                                   cudaStream_t = nullptr)
    {
        grouping_detail::throwNoCuda("sortByKey");
    }

    template <typename Key, typename Value>
    inline void sortByKey(const DenseStorage<Key, Device::GPU>&,
                          const DenseStorage<Value, Device::GPU>&,
                          DenseStorage<Key, Device::GPU>&,
                          DenseStorage<Value, Device::GPU>&,
                          GroupingWorkspace&,
                          cudaStream_t = nullptr)
    {
        grouping_detail::throwNoCuda("sortByKey");
    }

    template <typename Key, typename Value>
    inline SortedKeyValueResult<Key, Value, Device::GPU> sortByKeyAsync(const DenseStorage<Key, Device::GPU>&,
                                                                        const DenseStorage<Value, Device::GPU>&,
                                                                        GroupingWorkspace&,
                                                                        cudaStream_t)
    {
        grouping_detail::throwNoCuda("sortByKeyAsync");
    }

    template <typename Key, typename Value>
    inline void sortByKeyAsync(const DenseStorage<Key, Device::GPU>&,
                               const DenseStorage<Value, Device::GPU>&,
                               DenseStorage<Key, Device::GPU>&,
                               DenseStorage<Value, Device::GPU>&,
                               GroupingWorkspace&,
                               cudaStream_t)
    {
        grouping_detail::throwNoCuda("sortByKeyAsync");
    }

    template <typename Value>
    inline void sortByKey(ConstMatrixView<Int32Key3, Device::GPU>,
                          ConstMatrixView<Value, Device::GPU>,
                          MatrixView<Int32Key3, Device::GPU>,
                          MatrixView<Value, Device::GPU>,
                          GroupingWorkspace&,
                          cudaStream_t = nullptr)
    {
        grouping_detail::throwNoCuda("sortByKey");
    }

    template <typename Value>
    inline void sortByKeyAsync(ConstMatrixView<Int32Key3, Device::GPU>,
                               ConstMatrixView<Value, Device::GPU>,
                               MatrixView<Int32Key3, Device::GPU>,
                               MatrixView<Value, Device::GPU>,
                               GroupingWorkspace&,
                               cudaStream_t)
    {
        grouping_detail::throwNoCuda("sortByKeyAsync");
    }

    template <typename Key, typename Value>
    inline void sortByKey(ConstMatrixView<Key, Device::GPU>,
                          ConstMatrixView<Value, Device::GPU>,
                          MatrixView<Key, Device::GPU>,
                          MatrixView<Value, Device::GPU>,
                          GroupingWorkspace&,
                          cudaStream_t = nullptr)
    {
        grouping_detail::throwNoCuda("sortByKey");
    }

    template <typename Key, typename Value>
    inline void sortByKeyAsync(ConstMatrixView<Key, Device::GPU>,
                               ConstMatrixView<Value, Device::GPU>,
                               MatrixView<Key, Device::GPU>,
                               MatrixView<Value, Device::GPU>,
                               GroupingWorkspace&,
                               cudaStream_t)
    {
        grouping_detail::throwNoCuda("sortByKeyAsync");
    }

    template <typename Key>
    inline RunLengthEncodeResult<Key, Device::GPU> runLengthEncode(const DenseStorage<Key, Device::GPU>&)
    {
        grouping_detail::throwNoCuda("runLengthEncode");
    }

    template <typename Key>
    inline RunLengthEncodeResult<Key, Device::GPU>
    runLengthEncode(const DenseStorage<Key, Device::GPU>&, GroupingWorkspace&, cudaStream_t = nullptr)
    {
        grouping_detail::throwNoCuda("runLengthEncode");
    }

    template <typename Key>
    inline void runLengthEncode(const DenseStorage<Key, Device::GPU>&,
                                DenseStorage<Key, Device::GPU>&,
                                DenseStorage<Index, Device::GPU>&,
                                DenseStorage<Index, Device::GPU>&,
                                GroupingWorkspace&,
                                cudaStream_t = nullptr)
    {
        grouping_detail::throwNoCuda("runLengthEncode");
    }

    template <typename Key>
    inline RunLengthEncodeResult<Key, Device::GPU>
    runLengthEncodeAsync(const DenseStorage<Key, Device::GPU>&, GroupingWorkspace&, cudaStream_t)
    {
        grouping_detail::throwNoCuda("runLengthEncodeAsync");
    }

    template <typename Key>
    inline void runLengthEncodeAsync(const DenseStorage<Key, Device::GPU>&,
                                     DenseStorage<Key, Device::GPU>&,
                                     DenseStorage<Index, Device::GPU>&,
                                     DenseStorage<Index, Device::GPU>&,
                                     GroupingWorkspace&,
                                     cudaStream_t)
    {
        grouping_detail::throwNoCuda("runLengthEncodeAsync");
    }

    inline void runLengthEncode(ConstMatrixView<Int32Key3, Device::GPU>,
                                MatrixView<Int32Key3, Device::GPU>,
                                MatrixView<Index, Device::GPU>,
                                MatrixView<Index, Device::GPU>,
                                GroupingWorkspace&,
                                cudaStream_t = nullptr)
    {
        grouping_detail::throwNoCuda("runLengthEncode");
    }

    inline void runLengthEncodeAsync(ConstMatrixView<Int32Key3, Device::GPU>,
                                     MatrixView<Int32Key3, Device::GPU>,
                                     MatrixView<Index, Device::GPU>,
                                     MatrixView<Index, Device::GPU>,
                                     GroupingWorkspace&,
                                     cudaStream_t)
    {
        grouping_detail::throwNoCuda("runLengthEncodeAsync");
    }

    template <typename Key>
    inline void runLengthEncode(ConstMatrixView<Key, Device::GPU>,
                                MatrixView<Key, Device::GPU>,
                                MatrixView<Index, Device::GPU>,
                                MatrixView<Index, Device::GPU>,
                                GroupingWorkspace&,
                                cudaStream_t = nullptr)
    {
        grouping_detail::throwNoCuda("runLengthEncode");
    }

    template <typename Key>
    inline void runLengthEncodeAsync(ConstMatrixView<Key, Device::GPU>,
                                     MatrixView<Key, Device::GPU>,
                                     MatrixView<Index, Device::GPU>,
                                     MatrixView<Index, Device::GPU>,
                                     GroupingWorkspace&,
                                     cudaStream_t)
    {
        grouping_detail::throwNoCuda("runLengthEncodeAsync");
    }

    template <typename Key, typename Value>
    inline ReduceByKeyResult<Key, Value, Device::GPU>
    reduceByKey(const DenseStorage<Key, Device::GPU>&, const DenseStorage<Value, Device::GPU>&, GroupReduction)
    {
        grouping_detail::throwNoCuda("reduceByKey");
    }

    template <typename Key, typename Value>
    inline ReduceByKeyResult<Key, Value, Device::GPU> reduceByKey(const DenseStorage<Key, Device::GPU>&,
                                                                  const DenseStorage<Value, Device::GPU>&,
                                                                  GroupReduction,
                                                                  GroupingWorkspace&,
                                                                  cudaStream_t = nullptr)
    {
        grouping_detail::throwNoCuda("reduceByKey");
    }

    template <typename Key, typename Value>
    inline void reduceByKey(const DenseStorage<Key, Device::GPU>&,
                            const DenseStorage<Value, Device::GPU>&,
                            GroupReduction,
                            DenseStorage<Key, Device::GPU>&,
                            DenseStorage<Value, Device::GPU>&,
                            DenseStorage<Index, Device::GPU>&,
                            GroupingWorkspace&,
                            cudaStream_t = nullptr)
    {
        grouping_detail::throwNoCuda("reduceByKey");
    }

    template <typename Key, typename Value>
    inline ReduceByKeyResult<Key, Value, Device::GPU> reduceByKeyAsync(const DenseStorage<Key, Device::GPU>&,
                                                                       const DenseStorage<Value, Device::GPU>&,
                                                                       GroupReduction,
                                                                       GroupingWorkspace&,
                                                                       cudaStream_t)
    {
        grouping_detail::throwNoCuda("reduceByKeyAsync");
    }

    template <typename Key, typename Value>
    inline void reduceByKeyAsync(const DenseStorage<Key, Device::GPU>&,
                                 const DenseStorage<Value, Device::GPU>&,
                                 GroupReduction,
                                 DenseStorage<Key, Device::GPU>&,
                                 DenseStorage<Value, Device::GPU>&,
                                 DenseStorage<Index, Device::GPU>&,
                                 GroupingWorkspace&,
                                 cudaStream_t)
    {
        grouping_detail::throwNoCuda("reduceByKeyAsync");
    }

    template <typename Value>
    inline DenseStorage<Value, Device::GPU>
    segmentedReduce(const DenseStorage<Value, Device::GPU>&, const DenseStorage<Index, Device::GPU>&, GroupReduction)
    {
        grouping_detail::throwNoCuda("segmentedReduce");
    }

    template <typename Value>
    inline DenseStorage<Value, Device::GPU> segmentedReduce(const DenseStorage<Value, Device::GPU>&,
                                                           const DenseStorage<Index, Device::GPU>&,
                                                           GroupReduction,
                                                           GroupingWorkspace&,
                                                           cudaStream_t = nullptr)
    {
        grouping_detail::throwNoCuda("segmentedReduce");
    }

    template <typename Value>
    inline void segmentedReduce(const DenseStorage<Value, Device::GPU>&,
                                const DenseStorage<Index, Device::GPU>&,
                                GroupReduction,
                                DenseStorage<Value, Device::GPU>&,
                                GroupingWorkspace&,
                                cudaStream_t = nullptr)
    {
        grouping_detail::throwNoCuda("segmentedReduce");
    }

    template <typename Value>
    inline DenseStorage<Value, Device::GPU> segmentedReduceAsync(const DenseStorage<Value, Device::GPU>&,
                                                                const DenseStorage<Index, Device::GPU>&,
                                                                GroupReduction,
                                                                GroupingWorkspace&,
                                                                cudaStream_t)
    {
        grouping_detail::throwNoCuda("segmentedReduceAsync");
    }

    template <typename Value>
    inline void segmentedReduceAsync(const DenseStorage<Value, Device::GPU>&,
                                     const DenseStorage<Index, Device::GPU>&,
                                     GroupReduction,
                                     DenseStorage<Value, Device::GPU>&,
                                     GroupingWorkspace&,
                                     cudaStream_t)
    {
        grouping_detail::throwNoCuda("segmentedReduceAsync");
    }

    template <typename Value>
    inline void segmentedReduce(ConstMatrixView<Value, Device::GPU>,
                                ConstMatrixView<Index, Device::GPU>,
                                GroupReduction,
                                MatrixView<Value, Device::GPU>,
                                GroupingWorkspace&,
                                cudaStream_t = nullptr)
    {
        grouping_detail::throwNoCuda("segmentedReduce");
    }

    template <typename Value>
    inline void segmentedReduceAsync(ConstMatrixView<Value, Device::GPU>,
                                     ConstMatrixView<Index, Device::GPU>,
                                     GroupReduction,
                                     MatrixView<Value, Device::GPU>,
                                     GroupingWorkspace&,
                                     cudaStream_t)
    {
        grouping_detail::throwNoCuda("segmentedReduceAsync");
    }

#endif

} // namespace plamatrix::internal
