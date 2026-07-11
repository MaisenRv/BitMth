#pragma once
#include <algorithm>

#include <BitMth/linalg/Matrix.hpp>
#include <BitMth/utils/Constants.hpp>

namespace BitMth::ia{
    enum class LossFunct: unsigned char {
        MSE
        // BINARY_CROSS_ENTROPY
        // CATEGORICAL_CROSS_ENTROPY
    };

    template <typename T>
    [[nodiscard]] T mse(const linalg::Matrix<T>& predict, const linalg::Matrix<T>& real){
        CHECK_ERROR_MATRIX(
            predict.getRows() != real.getRows() || predict.getCols() != real.getCols(),
            "Loss function (MSE)",
            "Matrix dimensions must match (rows = rows && cols == cols)"
        );
        return (predict - real).pow(2).reduceSumTotal() / static_cast<T>(real.size());
    }

    template <typename T>
    [[nodiscard]] linalg::Matrix<T> mseDerivative(const linalg::Matrix<T>& predict, const linalg::Matrix<T>& real){
        return predict - real;
    }

    // template <typename T>
    // [[nodiscard]] T bce(const linalg::Matrix<T>& predict, const linalg::Matrix<T>& real) {
    //     CHECK_ERROR_MATRIX(
    //         predict.getRows() != real.getRows() || predict.getCols() != real.getCols(),
    //         "Loss function (BCE)",
    //         "Matrix dimensions must match (rows == rows && cols == cols)"
    //     );

    //     linalg::Matrix<T> lossMatrix(predict.getRows(), predict.getCols());

    //     for (size_t i = 0; i < predict.size(); i++) {
    //         T y_pred = std::clamp(predict.getValues()[i], utils::EPSILON<T>, T(1.0) - utils::EPSILON<T>);
    //         T y_real = real.getValues()[i];
    //         lossMatrix.getValues()[i] = y_real * std::log(y_pred) + (T(1.0) - y_real) * std::log(T(1.0) - y_pred);
    //     }
    //     return -lossMatrix.reduceSumTotal() / static_cast<T>(real.size());
    // }

    // template <typename T>
    // [[nodiscard]] linalg::Matrix<T> bceDerivative(const linalg::Matrix<T>& predict, const linalg::Matrix<T>& real) {
    //     CHECK_ERROR_MATRIX(
    //         predict.getRows() != real.getRows() || predict.getCols() != real.getCols(),
    //         "Loss function (BCE derivative)",
    //         "Matrix dimensions must match (rows == rows && cols == cols)"
    //     );
        
    //     linalg::Matrix<T> denominator = (T(1.0) - predict).hadamardInPlace(predict);
    //     return (predict - real) / (denominator + utils::EPSILON<T>) ;
    // }
}
