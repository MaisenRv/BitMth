#pragma once

#include <BitMth/linalg/Matrix.hpp>

namespace BitMth::ia::types{
    enum class ActivationFunctType: unsigned char {
        RELU,
        SIGMOID,
        TANH,
        SOFTMAX
    };

    template<typename T>
    struct ActivationContent{
        linalg::Matrix<T> (* function)(const linalg::Matrix<T>&, core::Arena*);
        linalg::Matrix<T> (* devFunction)(const linalg::Matrix<T>&,const linalg::Matrix<T>&, core::Arena*);
    };
}
