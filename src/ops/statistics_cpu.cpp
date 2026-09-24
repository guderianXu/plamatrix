#include "plamatrix/internal/ops/statistics.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

namespace plamatrix::internal
{

    template <typename Scalar>
    DenseStorage<std::uint8_t, Device::CPU> finiteRowMask(const DenseStorage<Scalar, Device::CPU>& input)
    {
        static_assert(std::is_floating_point_v<Scalar>, "finiteRowMask requires floating-point input");
        DenseStorage<std::uint8_t, Device::CPU> mask(input.rows(), 1);
        for (Index row = 0; row < input.rows(); ++row)
        {
            bool finite = true;
            for (Index column = 0; column < input.cols(); ++column)
            {
                if (!std::isfinite(input.data()[row + column * input.rows()]))
                {
                    finite = false;
                    break;
                }
            }
            mask.data()[row] = finite ? std::uint8_t{1} : std::uint8_t{0};
        }
        return mask;
    }

    template <typename Scalar>
    FiniteColumnBoundsWithMaskResult<Scalar, Device::CPU>
    finiteColumnBoundsWithMask(const DenseStorage<Scalar, Device::CPU>& input)
    {
        static_assert(std::is_floating_point_v<Scalar>, "finiteColumnBoundsWithMask requires floating-point input");
        FiniteColumnBoundsWithMaskResult<Scalar, Device::CPU> result{
            finiteRowMask(input),
            DenseStorage<Scalar, Device::CPU>(1, input.cols()),
            DenseStorage<Scalar, Device::CPU>(1, input.cols()),
            DenseStorage<Index, Device::CPU>(1, 1),
        };
        Index valid_count = 0;
        for (Index row = 0; row < input.rows(); ++row)
        {
            valid_count += result.rowMask.data()[row] != 0 ? 1 : 0;
        }
        result.validRowCount.data()[0] = valid_count;

        for (Index column = 0; column < input.cols(); ++column)
        {
            if (valid_count == 0)
            {
                const Scalar invalid = std::numeric_limits<Scalar>::quiet_NaN();
                result.minimum.data()[column] = invalid;
                result.maximum.data()[column] = invalid;
                continue;
            }
            Scalar minimum = std::numeric_limits<Scalar>::max();
            Scalar maximum = std::numeric_limits<Scalar>::lowest();
            for (Index row = 0; row < input.rows(); ++row)
            {
                if (result.rowMask.data()[row] == 0)
                    continue;
                const Scalar value = input.data()[row + column * input.rows()];
                minimum = value < minimum ? value : minimum;
                maximum = value > maximum ? value : maximum;
            }
            result.minimum.data()[column] = minimum;
            result.maximum.data()[column] = maximum;
        }
        return result;
    }

    template <typename Scalar>
    FiniteColumnBoundsResult<Scalar, Device::CPU> finiteColumnBounds(const DenseStorage<Scalar, Device::CPU>& input)
    {
        auto result = finiteColumnBoundsWithMask(input);
        return {
            std::move(result.minimum),
            std::move(result.maximum),
            std::move(result.validRowCount),
        };
    }

#ifdef PLAMATRIX_USE_FLOAT
    template DenseStorage<std::uint8_t, Device::CPU> finiteRowMask(const DenseStorage<float, Device::CPU>&);
    template FiniteColumnBoundsResult<float, Device::CPU> finiteColumnBounds(const DenseStorage<float, Device::CPU>&);
    template FiniteColumnBoundsWithMaskResult<float, Device::CPU>
    finiteColumnBoundsWithMask(const DenseStorage<float, Device::CPU>&);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
    template DenseStorage<std::uint8_t, Device::CPU> finiteRowMask(const DenseStorage<double, Device::CPU>&);
    template FiniteColumnBoundsResult<double, Device::CPU> finiteColumnBounds(const DenseStorage<double, Device::CPU>&);
    template FiniteColumnBoundsWithMaskResult<double, Device::CPU>
    finiteColumnBoundsWithMask(const DenseStorage<double, Device::CPU>&);
#endif

} // namespace plamatrix::internal
