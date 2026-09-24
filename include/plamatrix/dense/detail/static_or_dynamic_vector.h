#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "plamatrix/core/types.h"

namespace plamatrix::v1::decomposition_detail
{
    template <typename Value, int Capacity> class StaticOrDynamicVector
    {
        static constexpr bool IsDynamic = Capacity == Dynamic;
        using Storage = std::conditional_t<IsDynamic, std::vector<Value>, std::array<Value, Capacity>>;

    public:
        StaticOrDynamicVector() = default;
        explicit StaticOrDynamicVector(std::size_t size)
        {
            resize(size);
        }
        StaticOrDynamicVector(std::size_t size, const Value& value)
        {
            assign(size, value);
        }

        void resize(std::size_t size)
        {
            if constexpr (IsDynamic)
                _storage.resize(size);
            else
            {
                if (size > _storage.size())
                    throw std::length_error("fixed decomposition workspace capacity exceeded");
                _size = size;
            }
        }

        void assign(std::size_t size, const Value& value)
        {
            resize(size);
            std::fill(begin(), end(), value);
        }

        std::size_t size() const noexcept
        {
            if constexpr (IsDynamic)
                return _storage.size();
            return _size;
        }
        bool empty() const noexcept
        {
            return size() == 0;
        }
        Value* data() noexcept
        {
            return _storage.data();
        }
        const Value* data() const noexcept
        {
            return _storage.data();
        }
        Value* begin() noexcept
        {
            return data();
        }
        const Value* begin() const noexcept
        {
            return data();
        }
        Value* end() noexcept
        {
            return data() + size();
        }
        const Value* end() const noexcept
        {
            return data() + size();
        }
        Value& operator[](std::size_t index) noexcept
        {
            return data()[index];
        }
        const Value& operator[](std::size_t index) const noexcept
        {
            return data()[index];
        }

    private:
        Storage _storage{};
        std::size_t _size = 0;
    };
} // namespace plamatrix::v1::decomposition_detail
