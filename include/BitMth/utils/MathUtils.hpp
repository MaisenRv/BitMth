#pragma once

#include <cmath>
#include <cstdint>
#include <bit>

#include <BitMth/utils/Constants.hpp>

namespace BitMth::utils {

  template <typename T>
  inline bool isClose(T a, T b, T epsilon = EPSILON<T>) {
      return std::abs(a - b) < epsilon;
  }

  template <typename T>
  inline bool isNaN(T value){
    if constexpr (!std::is_floating_point_v<T>) return false; 
    else {
      #if defined(__FAST_MATH__)
        if constexpr (sizeof(T) == 4) { // float
            auto bits = std::bit_cast<uint32_t>(value);
            if ((bits & 0x7F800000U) == 0x7F800000U && (bits & 0x007FFFFFU) != 0) {
                return true;
            }
        } else if constexpr (sizeof(T) == 8) { // double
            auto bits = std::bit_cast<uint64_t>(value);
            if ((bits & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL && 
                (bits & 0x000FFFFFFFFFFFFFULL) != 0) {
                return true;
            }
        }
        return false;
      #else
        return std::isnan(value);
      #endif
    }
  }

}
