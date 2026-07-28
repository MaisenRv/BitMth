#pragma once
#include <algorithm>
#include <unordered_map>

#include <BitMth/linalg/Matrix.hpp>
#include <BitMth/utils/Constants.hpp>
#include <BitMth/ia/types/LossTypes.hpp>

namespace BitMth::ia{

    template <typename T>
    struct [[nodiscard]] Losses{
        static T mse(const linalg::Matrix<T>& predict, const linalg::Matrix<T>& real, core::Arena* targetArena = nullptr){
            CHECK_ERROR_MATRIX(
                predict.getRows() != real.getRows() || predict.getCols() != real.getCols(),
                "Loss function (MSE)",
                "Matrix dimensions must match (rows = rows && cols == cols)"
            );
            linalg::Matrix<T> lossMatrix = linalg::Matrix<T>::matrixApplyFunction(
                predict, real, targetArena, [](T mA, T mB) {
                    T diff = mA - mB;
                    return diff * diff;
                }
            );
            return lossMatrix.reduceSumTotal() / static_cast<T>(real.size());
        }

        static linalg::Matrix<T> mseDerivative(const linalg::Matrix<T>& predict, const linalg::Matrix<T>& real, core::Arena* targetArena = nullptr){
            CHECK_ERROR_MATRIX(
                predict.getRows() != real.getRows() || predict.getCols() != real.getCols(),
                "Loss function (MSE derivative)",
                "Matrix dimensions must match (rows == rows && cols == cols)"
            );

            const T invN = static_cast<T>(1) / static_cast<T>(real.size());
            return linalg::Matrix<T>::matrixApplyFunction(
                predict, real, targetArena, [invN](T mA, T mB) {
                    return (mA - mB) * invN;
                }
            );
        }

        static T bce(const linalg::Matrix<T>& predict, const linalg::Matrix<T>& real, core::Arena* targetArena = nullptr) {
            CHECK_ERROR_MATRIX(
                predict.getRows() != real.getRows() || predict.getCols() != real.getCols(),
                "Loss function (BCE)",
                "Matrix dimensions must match (rows == rows && cols == cols)"
            );
            const T one = 1;
            linalg::Matrix<T> lossMatrix = linalg::Matrix<T>::matrixApplyFunction(
                predict, real, targetArena, [one](T mA, T mB){
                    T y_pred = std::clamp(mA, utils::EPSILON<T>, one - utils::EPSILON<T>);
                    return  mB * std::log(y_pred) + (one - mB) * std::log(one - y_pred);
            });
            return (-lossMatrix.reduceSumTotal()) / T(real.size());
        }

        static linalg::Matrix<T> bceDerivative(const linalg::Matrix<T>& predict, const linalg::Matrix<T>& real, core::Arena* targetArena = nullptr) {
            CHECK_ERROR_MATRIX(
                predict.getRows() != real.getRows() || predict.getCols() != real.getCols(),
                "Loss function (BCE derivative)",
                "Matrix dimensions must match (rows == rows && cols == cols)"
            );
            const T one = static_cast<T>(1);
            const T N = one / static_cast<T>(real.size());

            return linalg::Matrix<T>::matrixApplyFunction(
                predict, real, targetArena, [one, N](T mA, T mB) {
                    T y_pred = std::clamp(mA, utils::EPSILON<T>, one - utils::EPSILON<T>);
                    T grad = (y_pred - mB) / (y_pred * (one - y_pred));
                    return grad * N;
                }
            );
        }
    };

    template<typename T>
    inline const ia::types::LossContent<T>& getLossFunction(const ia::types::LossFunctType type){
        static std::unordered_map<ia::types::LossFunctType, ia::types::LossContent<T>> functTable ={
            {ia::types::LossFunctType::MSE,                  {Losses<T>::mse, Losses<T>::mseDerivate}},
            {ia::types::LossFunctType::BINARY_CROSS_ENTROPY, {Losses<T>::bce, Losses<T>::bceDerivate}}
        };
        return functTable[type];
    } 
}
