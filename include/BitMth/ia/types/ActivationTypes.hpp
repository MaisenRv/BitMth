#pragma once
#include <BitMth/linalg/Matrix.hpp>

namespace BitMth::ia::types{
    enum class ActivationFunct: unsigned char {
        RELU,
        SIGMOID,
        TANH,
        SOFTMAX
    };

    template<typename T>
    struct activationsFunct{
        linalg::Matrix<T> (* function)(const linalg::Matrix<T>&);
        linalg::Matrix<T> (* devFunction)(const linalg::Matrix<T>&,const linalg::Matrix<T>&);
    };
}
