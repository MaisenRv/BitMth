#pragma once

#include <new>
#include <cstdint>
#include <stdexcept>
#include <cstddef>

namespace BitMth::core{
    struct [[nodiscard]] Arena{
        uint8_t *buffer{nullptr};
        size_t capacity{0};
        size_t offset{0};

        explicit Arena(size_t capacity):capacity(capacity){
            buffer = new uint8_t[capacity];
        }

        ~Arena() { delete[] buffer; }
        Arena(const Arena&) = delete;
        Arena& operator=(const Arena&) = delete;

        Arena(Arena&& other) noexcept: capacity(other.capacity), offset(other.offset), buffer(other.buffer) {
            other.buffer = nullptr;
            other.capacity = 0;
            other.offset = 0;
        }

        void* alloc(size_t size, size_t alignment = 16){
            uintptr_t currentAddress = reinterpret_cast<uintptr_t>(buffer + offset);
            uintptr_t remainder = currentAddress % alignment;
            size_t adjustment = 0;
            
            if (remainder != 0) adjustment = alignment - remainder;

            size_t nextOffset = offset + adjustment + size;
            if (nextOffset > capacity) throw std::bad_alloc();

            void* alignedPtr = buffer + offset + adjustment;
            
            offset = nextOffset;
            return alignedPtr;
        }

        void freeToMarker(size_t marker){
            if (marker > offset) throw std::out_of_range("Overflow offset: freeToMarker");
            offset = marker;
        }

        void reset() noexcept { offset = 0; }

        size_t getCapacity() const noexcept { return capacity; }
        size_t getOffset()   const noexcept { return offset; }
    };
}
