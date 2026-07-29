#pragma once

#include <BitMth/linalg/Matrix.hpp>

namespace BitMth::ia::types{
  enum class OptimizerType: unsigned char{
    SGD,
    SGD_MOMENTUM,
    ADAM,
    ADAM_W
  };

  template<typename T>
  struct OptimizerState{
    linalg::Matrix<T> firstMoment{nullptr};
    linalg::Matrix<T> secondMoment{nullptr};
    size_t stepCount = 0;
  };

  template<typename T>
  struct OptimizerContent{
    void (* opt)(linalg::Matrix<T>&, const linalg::Matrix<T> &,T, types::OptimizerState<T>& , T);
  };
}
