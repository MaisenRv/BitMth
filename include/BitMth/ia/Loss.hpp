#pragma once

#include <BitMth/linalg/Matrix.hpp>
#include <BitMth/utils/Constants.hpp>

namespace BitMth::ia{
    enum class LossFunct: unsigned char {
        MSE,
        BINARY_CROSS_ENTROPY
        // CATEGORICAL_CROSS_ENTROPY
    };

    template <typename T>
    T mse(const linalg::Matrix<T>& predict, const linalg::Matrix<T>& real){
        CHECK_ERROR_MATRIX(
            predict.rows != real.rows || predict.cols != real.cols,
            "Loss function (MSE)",
            "Matrix dimensions must match (rows = rows && cols == cols)"
        );
        return (predict - real).pow(2).reduceSumTotal() / static_cast<T>(real.numElements);
    }

    template <typename T>
    linalg::Matrix<T> mseDerivative(const linalg::Matrix<T>& predict, const linalg::Matrix<T>& real){
        return predict - real;
    }

    template <typename T>
    T bce(const linalg::Matrix<T>& predict, const linalg::Matrix<T>& real) {
        CHECK_ERROR_MATRIX(
            predict.rows != real.rows || predict.cols != real.cols,
            "Loss function (BCE)",
            "Matrix dimensions must match (rows == rows && cols == cols)"
        );

        linalg::Matrix<T> lossMatrix(predict.rows, predict.cols);

        for (size_t i = 0; i < predict.numElements; i++) {
            T y_pred = std::clamp(predict.m[i], utils::EPSILON<T>, T(1.0) - utils::EPSILON<T>);
            T y_real = real.m[i];
            lossMatrix.m[i] = y_real * std::log(y_pred) + (T(1.0) - y_real) * std::log(T(1.0) - y_pred);
        }
        return -lossMatrix.reduceSumTotal() / static_cast<T>(real.numElements);
    }

    template <typename T>
    linalg::Matrix<T> bceDerivative(const linalg::Matrix<T>& predict, const linalg::Matrix<T>& real) {
        CHECK_ERROR_MATRIX(
            predict.rows != real.rows || predict.cols != real.cols,
            "Loss function (BCE derivative)",
            "Matrix dimensions must match (rows == rows && cols == cols)"
        );
        
        linalg::Matrix<T> denominator = (T(1.0) - predict).hadamard(predict);
        return (predict - real) / (denominator + utils::EPSILON<T>) ;
    }
}
