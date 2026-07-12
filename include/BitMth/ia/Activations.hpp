#pragma once

#include <algorithm>
#include <cmath>
#include <unordered_map>

#include <BitMth/ia/types/ActivationTypes.hpp>
#include <BitMth/core/Arena.hpp>

namespace BitMth::ia{

    template <typename T>
    struct Activations{

        static linalg::Matrix<T> relu(const linalg::Matrix<T>& matrixZ, core::Arena* targetArena = nullptr ){
            linalg::Matrix<T> result(matrixZ.getRows(), matrixZ.getCols(), targetArena, false);
            const T* const valuesZ = matrixZ.getValues();
            const size_t size = matrixZ.size();
            T* const valuesR = result.getValues();
            
            for (size_t i = 0; i < size; i++) {
                valuesR[i] = std::max(T(0.0), valuesZ[i]);
            }
            return result;
        };

        static linalg::Matrix<T> reluDerivative(const linalg::Matrix<T>& matrixZ, const linalg::Matrix<T>& matrixA, core::Arena* targetArena = nullptr){
            linalg::Matrix<T> result(matrixZ.getRows(), matrixZ.getCols(), targetArena, false);
            const T* const valuesZ = matrixZ.getValues();
            const size_t size = matrixZ.size();
            T* const valuesR = result.getValues();
            
            for (size_t i = 0; i < size; i++) {
                valuesR[i] = valuesZ[i] > T(0.0) ? T(1.0) : T(0.0);
            }
            return result;
        };

        static linalg::Matrix<T> sigmoid(const linalg::Matrix<T>& matrixZ, core::Arena* targetArena = nullptr){
            linalg::Matrix<T> result(matrixZ.getRows(), matrixZ.getCols(), targetArena, false);
            const T* const valuesZ = matrixZ.getValues();
            const size_t size = matrixZ.size();
            T* const valuesR = result.getValues();

            for (size_t i = 0; i < size; i++) {
                valuesR[i] = T(1.0) / ( T(1.0) + std::exp(-valuesZ[i]) );
            }
            return result;
        };

        static linalg::Matrix<T> sigmoidDerivative(const linalg::Matrix<T>& matrixZ, const linalg::Matrix<T>& matrixA, core::Arena* targetArena = nullptr){
            linalg::Matrix<T> result(matrixA.getRows(), matrixA.getCols(), targetArena, false);
            const T* const valuesA = matrixA.getValues();
            T* const valuesR = result.getValues();
            const size_t size = matrixA.size();

            for (size_t i = 0; i < size; i++) {
                valuesR[i] = valuesA[i] * (T(1.0) - valuesA[i]);
            }
            return result;

        };

        static linalg::Matrix<T> Tanh(const linalg::Matrix<T>& matrixZ, core::Arena* targetArena = nullptr){
            linalg::Matrix<T> result(matrixZ.getRows(), matrixZ.getCols(), targetArena, false);
            const T* const valuesZ = matrixZ.getValues();
            const size_t size = matrixZ.size();
            T* const valuesR = result.getValues();

            for (size_t i = 0; i < size; i++) {
                valuesR[i] = std::tanh(valuesZ[i]);
            }
            return result;
        };

        static linalg::Matrix<T> TanhDerivative(const linalg::Matrix<T>& matrixZ, const linalg::Matrix<T>& matrixA, core::Arena* targetArena = nullptr){
            linalg::Matrix<T> result(matrixA.getRows(), matrixA.getCols(), targetArena, false);
            const T* const valuesA = matrixA.getValues();
            T* const valuesR = result.getValues();
            const size_t size = matrixA.size();

            for (size_t i = 0; i < size; i++) {
                valuesR[i] = T(1.0) - (valuesA[i] * valuesA[i]);
            }
            return result;
        };

        static linalg::Matrix<T> softmax(const linalg::Matrix<T>& matrixZ, core::Arena* targetArena = nullptr){
            linalg::Matrix<T> result(matrixZ.getRows(), matrixZ.getCols(), targetArena, false);
    
            const size_t jumpRowIn  = matrixZ.getStrides()[0] / sizeof(T);
            const size_t jumpColIn  = matrixZ.getStrides()[1] / sizeof(T);
    
            const size_t jumpRowOut = result.getStrides()[0] / sizeof(T);

            for (size_t r = 0; r < matrixZ.getRows(); r++) {
                const T* const rowIn = &matrixZ.getValues()[r * jumpRowIn];
                T* const rowOut = &result.getValues()[r * jumpRowOut];

                T maxVal = rowIn[0 * jumpColIn];
                for (size_t c = 1; c < matrixZ.getCols(); c++) {
                    maxVal = std::max(maxVal, rowIn[c * jumpColIn]);
                }
    
                T sum = T(0.0);
                for (size_t c = 0; c < matrixZ.getCols(); c++) {
                    rowOut[c] = std::exp(rowIn[c * jumpColIn] - maxVal); 
                    sum += rowOut[c];
                }
    
                for (size_t c = 0; c < matrixZ.getCols(); c++) {
                    rowOut[c] /= sum;
                }
            }
            return result;
        };
    };

    template<typename T>
    inline const ia::types::ActivationContent<T>& getActivationFunction(const ia::types::ActivationFunctType type){
        static std::unordered_map<ia::types::ActivationFunctType, ia::types::ActivationContent<T>> functTable ={
            {ia::types::ActivationFunctType::RELU,    {Activations<T>::relu, Activations<T>::reluDerivative}},
            {ia::types::ActivationFunctType::SIGMOID, {Activations<T>::sigmoid, Activations<T>::sigmoidDerivative}},
            {ia::types::ActivationFunctType::TANH,    {Activations<T>::Tanh, Activations<T>::TanhDerivative}},
            {ia::types::ActivationFunctType::SOFTMAX, {Activations<T>::softmax, nullptr}}
        };
        return functTable[type];
    }
}
