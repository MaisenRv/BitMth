#pragma once
#include <algorithm>

#include <BitMth/linalg/Matrix.hpp>
#include <BitMth/utils/Constants.hpp>

namespace BitMth::ia{
    enum class LossFunct: unsigned char {
        MSE,
        BINARY_CROSS_ENTROPY
        // CATEGORICAL_CROSS_ENTROPY
    };

    template <typename T>
    [[nodiscard]] T mse(const linalg::Matrix<T>& predict, const linalg::Matrix<T>& real){
        CHECK_ERROR_MATRIX(
            predict.getRows() != real.getRows() || predict.getCols() != real.getCols(),
            "Loss function (MSE)",
            "Matrix dimensions must match (rows = rows && cols == cols)"
        );
        linalg::Matrix<T> lossMatrix = linalg::Matrix<T>::matrixApplyFunction(
            predict, real, nullptr, [](T mA, T mB) {
                T diff = mA - mB;
                return diff * diff;
            }
        );
        return lossMatrix.reduceSumTotal() / static_cast<T>(real.size());
    }

    template <typename T>
    [[nodiscard]] linalg::Matrix<T> mseDerivative(const linalg::Matrix<T>& predict, const linalg::Matrix<T>& real){
        CHECK_ERROR_MATRIX(
            predict.getRows() != real.getRows() || predict.getCols() != real.getCols(),
            "Loss function (MSE derivative)",
            "Matrix dimensions must match (rows == rows && cols == cols)"
        );

        const T invN = static_cast<T>(1) / static_cast<T>(real.size());
        return linalg::Matrix<T>::matrixApplyFunction(
            predict, real, nullptr, [invN](T mA, T mB) {
                return (mA - mB) * invN;
            }
        );
    }

    template <typename T>
    [[nodiscard]] T bce(const linalg::Matrix<T>& predict, const linalg::Matrix<T>& real) {
        CHECK_ERROR_MATRIX(
            predict.getRows() != real.getRows() || predict.getCols() != real.getCols(),
            "Loss function (BCE)",
            "Matrix dimensions must match (rows == rows && cols == cols)"
        );
        const T one = 1;
        linalg::Matrix<T> lossMatrix = linalg::Matrix<T>::matrixApplyFunction(
            predict, real, nullptr, [one](T mA, T mB){
                T y_pred = std::clamp(mA, utils::EPSILON<T>, one - utils::EPSILON<T>);
                return  mB * std::log(y_pred) + (one - mB) * std::log(one - y_pred);
        });
        return (-lossMatrix.reduceSumTotal()) / T(real.size());
    }

    template <typename T>
    [[nodiscard]] linalg::Matrix<T> bceDerivative(const linalg::Matrix<T>& predict, const linalg::Matrix<T>& real) {
        CHECK_ERROR_MATRIX(
            predict.getRows() != real.getRows() || predict.getCols() != real.getCols(),
            "Loss function (BCE derivative)",
            "Matrix dimensions must match (rows == rows && cols == cols)"
        );
        const T one = static_cast<T>(1);
        const T N = one / static_cast<T>(real.size());

        return linalg::Matrix<T>::matrixApplyFunction(
            predict, real, nullptr, [one, N](T mA, T mB) {
                T y_pred = std::clamp(mA, utils::EPSILON<T>, one - utils::EPSILON<T>);
                T grad = (y_pred - mB) / (y_pred * (one - y_pred));
                return grad * N;
            }
        );
    }
}
