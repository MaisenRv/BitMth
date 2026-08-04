#pragma once
#include <algorithm>

#include <BitMth/linalg/Matrix.hpp>
#include <BitMth/utils/Constants.hpp>
#include <BitMth/ia/types/LossTypes.hpp>

namespace BitMth::ia{

    template <typename T>
    struct [[nodiscard]] LossFunctions{
      private:
        using Matrix = linalg::Matrix<T>;
      public:
        static T mse(const Matrix& predict, const Matrix& real, core::Arena* targetArena = nullptr){
            CHECK_ERROR_MATRIX(
                predict.getRows() != real.getRows() || predict.getCols() != real.getCols(),
                "Loss function (MSE)",
                "Matrix dimensions must match (rows = rows && cols == cols)"
            );
            Matrix lossMatrix = Matrix::matrixApplyFunction(
                predict, real, targetArena, [](T mA, T mB) {
                    T diff = mA - mB;
                    return diff * diff;
                }
            );
            return lossMatrix.reduceSumTotal() / static_cast<T>(real.size());
        }

        static Matrix mseDerivative(const Matrix& predict, const Matrix& real, core::Arena* targetArena = nullptr){
            CHECK_ERROR_MATRIX(
                predict.getRows() != real.getRows() || predict.getCols() != real.getCols(),
                "Loss function (MSE derivative)",
                "Matrix dimensions must match (rows == rows && cols == cols)"
            );

            const T invN = static_cast<T>(1) / static_cast<T>(real.size());
            return Matrix::matrixApplyFunction(
                predict, real, targetArena, [invN](T mA, T mB) {
                    return (mA - mB) * invN;
                }
            );
        }

        static T bce(const Matrix& predict, const Matrix& real, core::Arena* targetArena = nullptr) {
            CHECK_ERROR_MATRIX(
                predict.getRows() != real.getRows() || predict.getCols() != real.getCols(),
                "Loss function (BCE)",
                "Matrix dimensions must match (rows == rows && cols == cols)"
            );
            const T one = 1;
            Matrix lossMatrix = Matrix::matrixApplyFunction(
                predict, real, targetArena, [one](T mA, T mB){
                    T y_pred = std::clamp(mA, utils::EPSILON<T>, one - utils::EPSILON<T>);
                    return  mB * std::log(y_pred) + (one - mB) * std::log(one - y_pred);
            });
            return (-lossMatrix.reduceSumTotal()) / T(real.size());
        }

        static Matrix bceDerivative(const Matrix& predict, const Matrix& real, core::Arena* targetArena = nullptr) {
            CHECK_ERROR_MATRIX(
                predict.getRows() != real.getRows() || predict.getCols() != real.getCols(),
                "Loss function (BCE derivative)",
                "Matrix dimensions must match (rows == rows && cols == cols)"
            );
            const T one = static_cast<T>(1);
            const T N = one / static_cast<T>(real.size());

            return Matrix::matrixApplyFunction(
                predict, real, targetArena, [one, N](T mA, T mB) {
                    T y_pred = std::clamp(mA, utils::EPSILON<T>, one - utils::EPSILON<T>);
                    T grad = (y_pred - mB) / (y_pred * (one - y_pred));
                    return grad * N;
                }
            );
        }
    };
}
