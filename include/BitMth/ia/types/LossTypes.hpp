#pragma once

#include <BitMth/linalg/Matrix.hpp>

namespace BitMth::ia::types{
    enum class LossFunctType: unsigned char {
        MSE,
        BINARY_CROSS_ENTROPY
        // CATEGORICAL_CROSS_ENTROPY
    };

    template<typename T>
    struct LossContent {
      T (* lossFunct)(const linalg::Matrix<T>&, const linalg::Matrix<T>&, core::Arena*);
      linalg::Matrix<T> (* lossFunctDev)(const linalg::Matrix<T>&, const linalg::Matrix<T>&, core::Arena* );
    };
}
