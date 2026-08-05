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
    private:
        using Matrix = linalg::Matrix<T>;
    public:
      T (* lossFunct)(const Matrix&, const Matrix&, core::Arena*);
      Matrix  (* lossFunctDev)(const Matrix&, const Matrix&, core::Arena* );
    };
}
