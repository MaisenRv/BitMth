#pragma once

#include <algorithm>
#include <unordered_map>

#include <BitMth/ia/types/ActivationTypes.hpp>

namespace BitMth::ia{

    template <typename T>
    struct [[nodiscard]] Activations{

        static linalg::Matrix<T> relu(const linalg::Matrix<T>& matrixZ, core::Arena* targetArena = nullptr ){
            return linalg::Matrix<T>::scalarApplyFunction(matrixZ, T(0.0), targetArena,
                [](T valZ , T scalarZero ){
                    return valZ > scalarZero ? valZ : scalarZero;
                }
            );
        };

        static linalg::Matrix<T> reluDerivative(const linalg::Matrix<T>& matrixZ, const linalg::Matrix<T>& matrixA, core::Arena* targetArena = nullptr){
            return linalg::Matrix<T>::scalarApplyFunction(matrixZ, T(0.0), targetArena,
                [](T valZ , T scalarZero ){
                    return valZ > scalarZero ? T(1.0) : T(0.0);
                }
            );
        };

        static linalg::Matrix<T> sigmoid(const linalg::Matrix<T>& matrixZ, core::Arena* targetArena = nullptr){
            return linalg::Matrix<T>::scalarApplyFunction(matrixZ, T(1.0), targetArena,
                [](T valZ , T scalar ){
                    return scalar / ( scalar + std::exp(-valZ) );
                }
            );
        };

        static linalg::Matrix<T> sigmoidDerivative(const linalg::Matrix<T>& matrixZ, const linalg::Matrix<T>& matrixA, core::Arena* targetArena = nullptr){
            return linalg::Matrix<T>::scalarApplyFunction(matrixA, T(1.0), targetArena,
                [](T valA , T scalar ){
                    return valA * ( scalar - valA);
                }
            );
        };

        static linalg::Matrix<T> Tanh(const linalg::Matrix<T>& matrixZ, core::Arena* targetArena = nullptr){
            return linalg::Matrix<T>::scalarApplyFunction(matrixZ, T(0.0), targetArena,
                [](T valZ , T scalar ){
                    return std::tanh(valZ);
                }
            );
        };

        static linalg::Matrix<T> TanhDerivative(const linalg::Matrix<T>& matrixZ, const linalg::Matrix<T>& matrixA, core::Arena* targetArena = nullptr){
            return linalg::Matrix<T>::scalarApplyFunction(matrixA, T(1.0), targetArena,
                [](T valA , T scalar ){
                    return scalar - (valA * valA);
                }
            );
        };

        static linalg::Matrix<T> softmax(const linalg::Matrix<T>& matrixZ, core::Arena* targetArena = nullptr){
            size_t zRows = matrixZ.getRows();
            size_t zCols = matrixZ.getCols();
            if (zRows == 0 || zCols == 0) {
                return linalg::Matrix<T>(zRows, zCols, targetArena, false);
            }
            linalg::Matrix<T> result(zRows, zCols, targetArena, false);
    
            const size_t jumpRowIn  = matrixZ.getStrides()[0] / sizeof(T);
            const size_t jumpColIn  = matrixZ.getStrides()[1] / sizeof(T);
    
            const size_t jumpRowOut = result.getStrides()[0] / sizeof(T);

            for (size_t r = 0; r < zRows; r++) {
                const T* const rowIn = &matrixZ.getValues()[r * jumpRowIn];
                T* const rowOut = &result.getValues()[r * jumpRowOut];

                T maxVal = rowIn[0];
                for (size_t c = 1; c < zCols; c++) {
                    maxVal = std::max(maxVal, rowIn[c * jumpColIn]);
                }
    
                T sum = T(0.0);
                for (size_t c = 0; c < zCols; c++) {
                    rowOut[c] = std::exp(rowIn[c * jumpColIn] - maxVal); 
                    sum += rowOut[c];
                }
                T aux = T(1)/sum;
                for (size_t c = 0; c < zCols; c++) {
                    rowOut[c] *= aux;
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
