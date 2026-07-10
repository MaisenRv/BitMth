#pragma once

#include <algorithm>
#include <cmath>
#include <unordered_map>

#include <BitMth/ia/types/ActivationTypes.hpp>

namespace BitMth::ia{

    template <typename T>
    struct Activations{

        [[nodiscard]] static linalg::Matrix<T> relu(const linalg::Matrix<T>& matrixZ){
            linalg::Matrix<T> result(matrixZ.getRows(), matrixZ.getCols());
            for (size_t i = 0; i < matrixZ.size(); i++) {
                result.getValues()[i] = std::max( T(0.0), matrixZ.getValues()[i]);
            }
            return result;
        };

        [[nodiscard]] static linalg::Matrix<T> reluDerivative(const linalg::Matrix<T>& matrixZ, const linalg::Matrix<T>& matrixA){
            linalg::Matrix<T> result(matrixZ.getRows(), matrixZ.getCols());
            for (size_t i = 0; i < matrixZ.size(); i++) {
                result.getValues()[i] = matrixZ.getValues()[i] > T(0.0) ? T(1.0) : T(0.0);
            }
            return result;
        };

        [[nodiscard]] static linalg::Matrix<T> sigmoid(const linalg::Matrix<T>& matrixZ){
            linalg::Matrix<T> result(matrixZ.getRows(), matrixZ.getCols());
            for (size_t i = 0; i < matrixZ.size(); i++) {
                result.getValues()[i] = T(1.0) / ( T(1.0) + std::exp(-matrixZ.getValues()[i]) );
            }
            return result;
        };

        [[nodiscard]] static linalg::Matrix<T> sigmoidDerivative(const linalg::Matrix<T>& matrixZ, const linalg::Matrix<T>& matrixA){
            linalg::Matrix<T> result(matrixA.getRows(), matrixA.getCols());
            for (size_t i = 0; i < matrixA.size(); i++) {
                result.getValues()[i] = matrixA.getValues()[i] * ( T(1.0) - matrixA.getValues()[i] );
            }
            return result;
        };

        [[nodiscard]] static linalg::Matrix<T> Tanh(const linalg::Matrix<T>& matrixZ){
            linalg::Matrix<T> result(matrixZ.getRows(), matrixZ.getCols());
            for (size_t i = 0; i < matrixZ.size(); i++) {
                result.getValues()[i] = std::tanh(matrixZ.getValues()[i]);
            }
            return result;
        };

        [[nodiscard]] static linalg::Matrix<T> TanhDerivative(const linalg::Matrix<T>& matrixZ, const linalg::Matrix<T>& matrixA){
            linalg::Matrix<T> result(matrixA.getRows(), matrixA.getCols());
            for (size_t i = 0; i < matrixA.size(); i++) {
                result.getValues()[i] = T(1.0) - (matrixA.getValues()[i] * matrixA.getValues()[i]);
            }
            return result;
        };

        [[nodiscard]] static linalg::Matrix<T> softmax(const linalg::Matrix<T>& matrixZ){
            linalg::Matrix<T> result(matrixZ.getRows(), matrixZ.getCols());
            for (size_t r = 0; r < matrixZ.getRows(); r++) {
                size_t rowOffset = r * matrixZ.getCols();

                T maxVal = matrixZ.getValues()[rowOffset];
                for (size_t c = 1; c < matrixZ.getCols(); c++) {
                    maxVal = std::max(maxVal, matrixZ.getValues()[rowOffset + c]);
                }
            
                T sum = T(0.0);
                for (size_t c = 0; c < matrixZ.getCols(); c++) {
                    size_t idx = rowOffset + c;
                    result.getValues()[idx] = std::exp(matrixZ.getValues()[idx] - maxVal);
                    sum += result.getValues()[idx];
                }
            
                for (size_t c = 0; c < matrixZ.getCols(); c++) {
                    result.getValues()[rowOffset + c] /= sum;
                }
            }
        
            return result;
        };
    };

    template<typename T>
    inline std::unordered_map<ia::types::ActivationFunct, ia::types::activationsFunct<T>>& getActivationFunctions(){
        static std::unordered_map<ia::types::ActivationFunct, ia::types::activationsFunct<T>> functTable ={
            {ia::types::ActivationFunct::RELU,    {Activations<T>::relu, Activations<T>::reluDerivative}},
            {ia::types::ActivationFunct::SIGMOID, {Activations<T>::sigmoid, Activations<T>::sigmoidDerivative}},
            {ia::types::ActivationFunct::TANH,    {Activations<T>::Tanh, Activations<T>::TanhDerivative}},
            {ia::types::ActivationFunct::SOFTMAX, {Activations<T>::softmax, nullptr}}
        };
        return functTable;
    }
}
