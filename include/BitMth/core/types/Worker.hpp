#pragma once

#include <atomic>
#include <thread>

namespace BitMth::linalg {
    template <typename T>
    class Matrix;
}

namespace BitMth::core {
  struct alignas(64) Worker{
    std::atomic_bool state{true};
    std::atomic_bool hasWork{false};
    std::atomic_bool done{true}; 
    unsigned int start{0};
    unsigned int end{0};
    std::thread thr;
  };


  template<typename T, typename Op>
  struct TaskData{
    linalg::Matrix<T>& matrix;
    T scalar;
    Op funct;
  };
}
