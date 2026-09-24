#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "plamatrix/internal/ops/indexing.h"

namespace plamatrix::internal
{
namespace
{

void validateIndexVector(const DenseStorage<Index, Device::CPU>& indices, const char* operation)
{
    if (indices.cols() != 1)
    {
        throw std::invalid_argument(std::string(operation) + ": indices must have shape K x 1");
    }
}

void validateIndexRange(
    const DenseStorage<Index, Device::CPU>& indices,
    Index row_count,
    const char* operation)
{
    for (Index source_row = 0; source_row < indices.rows(); ++source_row)
    {
        const Index destination_row = indices.data()[source_row];
        if (destination_row < 0 || destination_row >= row_count)
        {
            throw std::out_of_range(std::string(operation) + ": row index is out of range");
        }
    }
}

} // namespace

DenseStorage<Index, Device::CPU> exclusiveScan(
    const DenseStorage<Index, Device::CPU>& counts)
{
    DenseStorage<Index, Device::CPU> result(counts.rows(), counts.cols());
    Index prefix = 0;
    for (Index offset = 0; offset < counts.size(); ++offset)
    {
        result.data()[offset] = prefix;
        const Index value = counts.data()[offset];
        if ((value > 0 && prefix > std::numeric_limits<Index>::max() - value) ||
            (value < 0 && prefix < std::numeric_limits<Index>::min() - value))
        {
            throw std::overflow_error("exclusiveScan: Index prefix overflow");
        }
        prefix += value;
    }
    return result;
}

template <typename Scalar>
DenseStorage<Scalar, Device::CPU> gatherRows(
    const DenseStorage<Scalar, Device::CPU>& input,
    const DenseStorage<Index, Device::CPU>& indices)
{
    validateIndexVector(indices, "gatherRows");
    validateIndexRange(indices, input.rows(), "gatherRows");

    DenseStorage<Scalar, Device::CPU> result(indices.rows(), input.cols());
    for (Index col = 0; col < input.cols(); ++col)
    {
        for (Index output_row = 0; output_row < indices.rows(); ++output_row)
        {
            const Index input_row = indices.data()[output_row];
            result.data()[output_row + col * result.rows()] =
                input.data()[input_row + col * input.rows()];
        }
    }
    return result;
}

template <typename Scalar>
void scatterRows(
    const DenseStorage<Scalar, Device::CPU>& values,
    const DenseStorage<Index, Device::CPU>& indices,
    DenseStorage<Scalar, Device::CPU>& output)
{
    validateIndexVector(indices, "scatterRows");
    if (values.rows() != indices.rows() || values.cols() != output.cols())
    {
        throw std::invalid_argument("scatterRows: values and output shapes are incompatible");
    }
    validateIndexRange(indices, output.rows(), "scatterRows");
    if (values.cols() == 0)
    {
        return;
    }

    std::vector<std::pair<Index, Index>> sorted_sources;
    const std::uintmax_t source_count = static_cast<std::uintmax_t>(values.rows());
    if (source_count > static_cast<std::uintmax_t>(sorted_sources.max_size()))
    {
        throw std::length_error("scatterRows: source row count exceeds scratch capacity");
    }
    sorted_sources.reserve(static_cast<std::size_t>(source_count));
    for (Index source_row = 0; source_row < values.rows(); ++source_row)
    {
        sorted_sources.emplace_back(indices.data()[source_row], source_row);
    }
    std::sort(sorted_sources.begin(), sorted_sources.end());

    std::size_t pair_index = 0;
    while (pair_index < sorted_sources.size())
    {
        const Index destination_row = sorted_sources[pair_index].first;
        const Index source_row = sorted_sources[pair_index].second;
        for (Index col = 0; col < values.cols(); ++col)
        {
            output.data()[destination_row + col * output.rows()] =
                values.data()[source_row + col * values.rows()];
        }

        do
        {
            ++pair_index;
        }
        while (pair_index < sorted_sources.size() &&
               sorted_sources[pair_index].first == destination_row);
    }
}

template <typename Scalar>
CompactRowsResult<Scalar, Device::CPU> compactRows(
    const DenseStorage<Scalar, Device::CPU>& input,
    const DenseStorage<std::uint8_t, Device::CPU>& keep_mask)
{
    if (keep_mask.cols() != 1 || keep_mask.rows() != input.rows())
    {
        throw std::invalid_argument("compactRows: keep_mask must have shape input.rows() x 1");
    }

    Index selected_count = 0;
    for (Index source_row = 0; source_row < input.rows(); ++source_row)
    {
        if (keep_mask.data()[source_row] != 0)
        {
            ++selected_count;
        }
    }

    CompactRowsResult<Scalar, Device::CPU> result{
        DenseStorage<Scalar, Device::CPU>(selected_count, input.cols()),
        DenseStorage<Index, Device::CPU>(selected_count, 1)
    };
    Index output_row = 0;
    for (Index source_row = 0; source_row < input.rows(); ++source_row)
    {
        if (keep_mask.data()[source_row] == 0)
        {
            continue;
        }

        result.sourceIndices.data()[output_row] = source_row;
        for (Index col = 0; col < input.cols(); ++col)
        {
            result.values.data()[output_row + col * selected_count] =
                input.data()[source_row + col * input.rows()];
        }
        ++output_row;
    }
    return result;
}

#ifdef PLAMATRIX_USE_FLOAT
template DenseStorage<float, Device::CPU> gatherRows(
    const DenseStorage<float, Device::CPU>&,
    const DenseStorage<Index, Device::CPU>&);
template void scatterRows(
    const DenseStorage<float, Device::CPU>&,
    const DenseStorage<Index, Device::CPU>&,
    DenseStorage<float, Device::CPU>&);
template CompactRowsResult<float, Device::CPU> compactRows(
    const DenseStorage<float, Device::CPU>&,
    const DenseStorage<std::uint8_t, Device::CPU>&);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template DenseStorage<double, Device::CPU> gatherRows(
    const DenseStorage<double, Device::CPU>&,
    const DenseStorage<Index, Device::CPU>&);
template void scatterRows(
    const DenseStorage<double, Device::CPU>&,
    const DenseStorage<Index, Device::CPU>&,
    DenseStorage<double, Device::CPU>&);
template CompactRowsResult<double, Device::CPU> compactRows(
    const DenseStorage<double, Device::CPU>&,
    const DenseStorage<std::uint8_t, Device::CPU>&);
#endif

#define PLAMATRIX_INSTANTIATE_CPU_INDEXING(Scalar)                                    \
    template DenseStorage<Scalar, Device::CPU> gatherRows(                             \
        const DenseStorage<Scalar, Device::CPU>&, const DenseStorage<Index, Device::CPU>&); \
    template void scatterRows(                                                        \
        const DenseStorage<Scalar, Device::CPU>&, const DenseStorage<Index, Device::CPU>&, \
        DenseStorage<Scalar, Device::CPU>&);                                            \
    template CompactRowsResult<Scalar, Device::CPU> compactRows(                      \
        const DenseStorage<Scalar, Device::CPU>&,                                      \
        const DenseStorage<std::uint8_t, Device::CPU>&)

PLAMATRIX_INSTANTIATE_CPU_INDEXING(std::uint8_t);
PLAMATRIX_INSTANTIATE_CPU_INDEXING(std::uint16_t);
PLAMATRIX_INSTANTIATE_CPU_INDEXING(Index);

#undef PLAMATRIX_INSTANTIATE_CPU_INDEXING

} // namespace plamatrix::internal
