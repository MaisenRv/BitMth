#pragma once

#include <cstddef>
namespace BitMth::core::config {
  inline constexpr size_t DEFAULT_NUMBER_THREADS = 1;
  inline constexpr size_t PARALLEL_THRESHOLD_SIMPLE = 150000;
  inline constexpr size_t PARALLEL_THRESHOLD_COMPLEX = 200000;

}
