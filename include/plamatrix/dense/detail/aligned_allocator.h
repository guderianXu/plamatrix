#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace plamatrix::v1::storage_detail
{
    template <typename Value> class DefaultInitAllocator
    {
    public:
        using value_type = Value;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using propagate_on_container_move_assignment = std::true_type;
        using is_always_equal = std::true_type;

        template <typename Other> struct rebind
        {
            using other = DefaultInitAllocator<Other>;
        };

        DefaultInitAllocator() noexcept = default;

        template <typename Other> constexpr DefaultInitAllocator(const DefaultInitAllocator<Other>&) noexcept
        {
        }

        [[nodiscard]] Value* allocate(size_type count)
        {
            return count == 0 ? nullptr : std::allocator<Value>{}.allocate(count);
        }

        void deallocate(Value* pointer, size_type count) noexcept
        {
            if (pointer != nullptr)
            {
                std::allocator<Value>{}.deallocate(pointer, count);
            }
        }

        template <typename Other> void construct(Other* pointer)
        {
            ::new (static_cast<void*>(pointer)) Other;
        }

        template <typename Other, typename First, typename... Rest>
        void construct(Other* pointer, First&& first, Rest&&... rest)
        {
            ::new (static_cast<void*>(pointer)) Other(std::forward<First>(first), std::forward<Rest>(rest)...);
        }

        template <typename Other> void destroy(Other* pointer) noexcept
        {
            pointer->~Other();
        }
    };

    template <typename Left, typename Right>
    constexpr bool operator==(const DefaultInitAllocator<Left>&, const DefaultInitAllocator<Right>&) noexcept
    {
        return true;
    }

    template <typename Left, typename Right>
    constexpr bool operator!=(const DefaultInitAllocator<Left>&, const DefaultInitAllocator<Right>&) noexcept
    {
        return false;
    }

    template <typename Value, std::size_t Alignment> class AlignedAllocator
    {
        static_assert(Alignment >= alignof(Value), "Allocator alignment must satisfy the value type");
        static_assert((Alignment & (Alignment - 1)) == 0, "Allocator alignment must be a power of two");

    public:
        using value_type = Value;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using propagate_on_container_move_assignment = std::true_type;
        using is_always_equal = std::true_type;

        template <typename Other> struct rebind
        {
            using other = AlignedAllocator<Other, Alignment>;
        };

        AlignedAllocator() noexcept = default;

        template <typename Other> constexpr AlignedAllocator(const AlignedAllocator<Other, Alignment>&) noexcept
        {
        }

        [[nodiscard]] Value* allocate(size_type count)
        {
            if (count > std::numeric_limits<size_type>::max() / sizeof(Value))
            {
                throw std::bad_array_new_length();
            }
            if (count == 0)
            {
                return nullptr;
            }
            return static_cast<Value*>(::operator new(count * sizeof(Value), std::align_val_t{Alignment}));
        }

        void deallocate(Value* pointer, size_type) noexcept
        {
            ::operator delete(pointer, std::align_val_t{Alignment});
        }

        template <typename Other> void construct(Other* pointer)
        {
            // Default-initialization starts the object lifetime without value-initializing
            // arithmetic scalars. Matrix factories explicitly initialize their outputs.
            ::new (static_cast<void*>(pointer)) Other;
        }

        template <typename Other, typename First, typename... Rest>
        void construct(Other* pointer, First&& first, Rest&&... rest)
        {
            ::new (static_cast<void*>(pointer)) Other(std::forward<First>(first), std::forward<Rest>(rest)...);
        }

        template <typename Other> void destroy(Other* pointer) noexcept
        {
            pointer->~Other();
        }
    };

    template <typename Left, typename Right, std::size_t Alignment>
    constexpr bool operator==(const AlignedAllocator<Left, Alignment>&,
                              const AlignedAllocator<Right, Alignment>&) noexcept
    {
        return true;
    }

    template <typename Left, typename Right, std::size_t Alignment>
    constexpr bool operator!=(const AlignedAllocator<Left, Alignment>&,
                              const AlignedAllocator<Right, Alignment>&) noexcept
    {
        return false;
    }
} // namespace plamatrix::v1::storage_detail
