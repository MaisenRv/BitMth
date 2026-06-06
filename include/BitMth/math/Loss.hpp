#pragma once

#include <BitMth/core/Matrix.hpp>

namespace BitMth{
    namespace Math{
        enum class LossFunct: unsigned char {
            MSE,
            BINARY_CROSS_ENTROPY
            // CATEGORICAL_CROSS_ENTROPY
        };

        template <typename T>
        T mse(const Matrix<T>& predict, const Matrix<T>& real){
            CHECK_ERROR_MATRIX(
                predict.rows != real.rows || predict.cols != real.cols,
                "Loss function (MSE)",
                "Matrix dimensions must match (rows = rows && cols == cols)"
            );
            return (predict - real).pow(2).reduceSumTotal() / static_cast<T>(real.numElements);
        }

        template <typename T>
        Matrix<T> mseDerivative(const Matrix<T>& predict, const Matrix<T>& real){
            return predict - real;
        }

        template <typename T>
        T bce(const Matrix<T>& predict, const Matrix<T>& real) {
            CHECK_ERROR_MATRIX(
                predict.rows != real.rows || predict.cols != real.cols,
                "Loss function (BCE)",
                "Matrix dimensions must match (rows == rows && cols == cols)"
            );

            Matrix<T> lossMatrix(predict.rows, predict.cols);
            T epsilon = T(1e-7);

            for (size_t i = 0; i < predict.numElements; i++) {
                T y_pred = std::clamp(predict.m[i], epsilon, T(1.0) - epsilon);
                T y_real = real.m[i];
                lossMatrix.m[i] = y_real * std::log(y_pred) + (T(1.0) - y_real) * std::log(T(1.0) - y_pred);
            }
            return -lossMatrix.reduceSumTotal() / static_cast<T>(real.numElements);
        }

        template <typename T>
        Matrix<T> bceDerivative(const Matrix<T>& predict, const Matrix<T>& real) {
            CHECK_ERROR_MATRIX(
                predict.rows != real.rows || predict.cols != real.cols,
                "Loss function (BCE derivative)",
                "Matrix dimensions must match (rows == rows && cols == cols)"
            );
            T epsilon = static_cast<T>(1e-7);

            Matrix<T> denominator = (T(1.0) - predict).hadamard(predict);
            return (predict - real) / (denominator + epsilon) ;
        }

    }

}
