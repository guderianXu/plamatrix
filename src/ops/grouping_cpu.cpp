#include "plamatrix/internal/ops/grouping.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace plamatrix::internal
{
    namespace
    {

        template <typename Scalar> Scalar combine(Scalar left, Scalar right, GroupReduction operation)
        {
            switch (operation)
            {
            case GroupReduction::Sum:
                return left + right;
            case GroupReduction::Minimum:
                return right < left ? right : left;
            case GroupReduction::Maximum:
                return right > left ? right : left;
            }
            throw std::invalid_argument("group reduction operation is invalid");
        }

        template <typename Scalar> Scalar emptyValue(GroupReduction operation)
        {
            switch (operation)
            {
            case GroupReduction::Sum:
                return Scalar(0);
            case GroupReduction::Minimum:
                return std::numeric_limits<Scalar>::max();
            case GroupReduction::Maximum:
                return std::numeric_limits<Scalar>::lowest();
            }
            throw std::invalid_argument("group reduction operation is invalid");
        }

        template <typename Scalar, Device Dev>
        void requireColumnVector(const char* operation, const DenseStorage<Scalar, Dev>& input)
        {
            if (input.cols() != 1)
            {
                throw std::invalid_argument(std::string(operation) + ": input must be a column vector");
            }
        }

        void validateOperation(GroupReduction operation)
        {
            switch (operation)
            {
            case GroupReduction::Sum:
            case GroupReduction::Minimum:
            case GroupReduction::Maximum:
                return;
            }
            throw std::invalid_argument("group reduction operation is invalid");
        }

    } // namespace

    template <typename Key, typename Value>
    SortedKeyValueResult<Key, Value, Device::CPU> sortByKey(const DenseStorage<Key, Device::CPU>& keys,
                                                            const DenseStorage<Value, Device::CPU>& values)
    {
        requireColumnVector("sortByKey", keys);
        requireColumnVector("sortByKey", values);
        if (keys.rows() != values.rows())
        {
            throw std::invalid_argument("sortByKey: keys and values must have the same number of rows");
        }

        std::vector<Index> order(static_cast<std::size_t>(keys.rows()));
        std::iota(order.begin(), order.end(), Index(0));
        std::stable_sort(order.begin(),
                         order.end(),
                         [&](Index left, Index right) { return keys.data()[left] < keys.data()[right]; });

        SortedKeyValueResult<Key, Value, Device::CPU> result{
            DenseStorage<Key, Device::CPU>::uninitialized(keys.rows(), 1),
            DenseStorage<Value, Device::CPU>::uninitialized(values.rows(), 1),
        };
        for (Index output = 0; output < keys.rows(); ++output)
        {
            const Index source = order[static_cast<std::size_t>(output)];
            result.keys.data()[output] = keys.data()[source];
            result.values.data()[output] = values.data()[source];
        }
        return result;
    }

    template <typename Key>
    RunLengthEncodeResult<Key, Device::CPU> runLengthEncode(const DenseStorage<Key, Device::CPU>& keys)
    {
        requireColumnVector("runLengthEncode", keys);
        Index run_count = 0;
        for (Index row = 0; row < keys.rows(); ++row)
        {
            if (row == 0 || keys.data()[row] != keys.data()[row - 1])
            {
                ++run_count;
            }
        }

        RunLengthEncodeResult<Key, Device::CPU> result{
            DenseStorage<Key, Device::CPU>::uninitialized(run_count, 1),
            DenseStorage<Index, Device::CPU>::uninitialized(run_count, 1),
            DenseStorage<Index, Device::CPU>::uninitialized(1, 1),
        };
        result.runCount.data()[0] = run_count;
        Index output = -1;
        for (Index row = 0; row < keys.rows(); ++row)
        {
            if (row == 0 || keys.data()[row] != keys.data()[row - 1])
            {
                ++output;
                result.uniqueKeys.data()[output] = keys.data()[row];
                result.counts.data()[output] = 1;
            }
            else
            {
                ++result.counts.data()[output];
            }
        }
        return result;
    }

    template <typename Key, typename Value>
    ReduceByKeyResult<Key, Value, Device::CPU> reduceByKey(const DenseStorage<Key, Device::CPU>& keys,
                                                           const DenseStorage<Value, Device::CPU>& values,
                                                           GroupReduction operation)
    {
        requireColumnVector("reduceByKey", keys);
        requireColumnVector("reduceByKey", values);
        validateOperation(operation);
        if (keys.rows() != values.rows())
        {
            throw std::invalid_argument("reduceByKey: keys and values must have the same number of rows");
        }

        const auto runs = runLengthEncode(keys);
        const Index run_count = runs.runCount.data()[0];
        ReduceByKeyResult<Key, Value, Device::CPU> result{
            DenseStorage<Key, Device::CPU>::uninitialized(run_count, 1),
            DenseStorage<Value, Device::CPU>::uninitialized(run_count, 1),
            DenseStorage<Index, Device::CPU>::uninitialized(1, 1),
        };
        result.runCount.data()[0] = run_count;
        for (Index run = 0; run < run_count; ++run)
        {
            result.uniqueKeys.data()[run] = runs.uniqueKeys.data()[run];
        }
        Index output = -1;
        for (Index row = 0; row < keys.rows(); ++row)
        {
            if (row == 0 || keys.data()[row] != keys.data()[row - 1])
            {
                ++output;
                result.aggregates.data()[output] = values.data()[row];
            }
            else
            {
                result.aggregates.data()[output] =
                    combine(result.aggregates.data()[output], values.data()[row], operation);
            }
        }
        return result;
    }

    template <typename Value>
    DenseStorage<Value, Device::CPU> segmentedReduce(const DenseStorage<Value, Device::CPU>& values,
                                                    const DenseStorage<Index, Device::CPU>& offsets,
                                                    GroupReduction operation)
    {
        requireColumnVector("segmentedReduce", values);
        requireColumnVector("segmentedReduce", offsets);
        validateOperation(operation);
        if (offsets.rows() == 0)
        {
            throw std::invalid_argument("segmentedReduce: offsets must contain at least one element");
        }
        if (offsets.data()[0] != 0 || offsets.data()[offsets.rows() - 1] != values.rows())
        {
            throw std::invalid_argument("segmentedReduce: offsets must begin at zero and end at values.rows()");
        }
        for (Index row = 1; row < offsets.rows(); ++row)
        {
            if (offsets.data()[row] < offsets.data()[row - 1])
            {
                throw std::invalid_argument("segmentedReduce: offsets must be nondecreasing");
            }
        }

        DenseStorage<Value, Device::CPU> output = DenseStorage<Value, Device::CPU>::uninitialized(offsets.rows() - 1, 1);
        for (Index segment = 0; segment < output.rows(); ++segment)
        {
            const Index begin = offsets.data()[segment];
            const Index end = offsets.data()[segment + 1];
            if (begin == end)
            {
                output.data()[segment] = emptyValue<Value>(operation);
                continue;
            }
            Value aggregate = values.data()[begin];
            for (Index row = begin + 1; row < end; ++row)
            {
                aggregate = combine(aggregate, values.data()[row], operation);
            }
            output.data()[segment] = aggregate;
        }
        return output;
    }

#define PLAMATRIX_INSTANTIATE_CPU_KEY_VALUE(Key, Value)                                                                \
    template SortedKeyValueResult<Key, Value, Device::CPU> sortByKey(const DenseStorage<Key, Device::CPU>&,             \
                                                                     const DenseStorage<Value, Device::CPU>&);          \
    template ReduceByKeyResult<Key, Value, Device::CPU> reduceByKey(                                                   \
        const DenseStorage<Key, Device::CPU>&, const DenseStorage<Value, Device::CPU>&, GroupReduction)

#define PLAMATRIX_INSTANTIATE_CPU_KEY(Key)                                                                             \
    template RunLengthEncodeResult<Key, Device::CPU> runLengthEncode(const DenseStorage<Key, Device::CPU>&);            \
    PLAMATRIX_INSTANTIATE_CPU_KEY_VALUE(Key, float);                                                                   \
    PLAMATRIX_INSTANTIATE_CPU_KEY_VALUE(Key, double);                                                                  \
    PLAMATRIX_INSTANTIATE_CPU_KEY_VALUE(Key, Index)

    PLAMATRIX_INSTANTIATE_CPU_KEY(std::uint32_t);
    PLAMATRIX_INSTANTIATE_CPU_KEY(std::uint64_t);
    PLAMATRIX_INSTANTIATE_CPU_KEY(Index);

    template SortedKeyValueResult<Int32Key3, Index, Device::CPU> sortByKey(const DenseStorage<Int32Key3, Device::CPU>&,
                                                                           const DenseStorage<Index, Device::CPU>&);
    template SortedKeyValueResult<Int32Key3, std::int32_t, Device::CPU>
    sortByKey(const DenseStorage<Int32Key3, Device::CPU>&, const DenseStorage<std::int32_t, Device::CPU>&);
    template RunLengthEncodeResult<Int32Key3, Device::CPU> runLengthEncode(const DenseStorage<Int32Key3, Device::CPU>&);

    template DenseStorage<float, Device::CPU>
    segmentedReduce(const DenseStorage<float, Device::CPU>&, const DenseStorage<Index, Device::CPU>&, GroupReduction);
    template DenseStorage<double, Device::CPU>
    segmentedReduce(const DenseStorage<double, Device::CPU>&, const DenseStorage<Index, Device::CPU>&, GroupReduction);
    template DenseStorage<Index, Device::CPU>
    segmentedReduce(const DenseStorage<Index, Device::CPU>&, const DenseStorage<Index, Device::CPU>&, GroupReduction);

#undef PLAMATRIX_INSTANTIATE_CPU_KEY
#undef PLAMATRIX_INSTANTIATE_CPU_KEY_VALUE

} // namespace plamatrix::internal
