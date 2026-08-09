#pragma once

#include <algorithm>

#include <BitMth/ia/types/ActivationTypes.hpp>

namespace BitMth::ia{
    template <typename T>
    struct [[nodiscard]] ActivationFunctions{
        private:
            using Matrix = linalg::Matrix<T>;
        public:

        static Matrix relu(const Matrix& matrixZ, core::Arena* targetArena = nullptr ){
            return Matrix::scalarApplyFunction(matrixZ, T(0.0), targetArena,
                [](T valZ , T scalarZero ){
                    return valZ > scalarZero ? valZ : scalarZero;
                }
            );
        };

        static Matrix reluDerivative(const Matrix& matrixA, const Matrix& matrixGrad, core::Arena* targetArena = nullptr){
            return Matrix::matrixApplyFunction(matrixA, matrixGrad, targetArena,
                [](T valA, T valGrad) {
                    return valA > T(0.0) ? valGrad : T(0.0);
                }
            );
        };

        static Matrix sigmoid(const Matrix& matrixZ, core::Arena* targetArena = nullptr){
            return Matrix::scalarApplyFunction(matrixZ, T(1.0), targetArena,
                [](T valZ , T scalar ){
                    return scalar / ( scalar + std::exp(-valZ) );
                }
            );
        };

        static Matrix sigmoidDerivative(const Matrix& matrixA, const Matrix& matrixGrad, core::Arena* targetArena = nullptr){
            return Matrix::matrixApplyFunction(matrixA, matrixGrad, targetArena,
                [](T valA, T valGrad) {
                    return valGrad * valA * (T(1.0) - valA);
                }
            );
        };

        static Matrix tanh(const Matrix& matrixZ, core::Arena* targetArena = nullptr){
            return Matrix::scalarApplyFunction(matrixZ, T(0.0), targetArena,
                [](T valZ , T scalar ){
                    return std::tanh(valZ);
                }
            );
        };

        static Matrix tanhDerivative(const Matrix& matrixA, const Matrix& matrixGrad, core::Arena* targetArena = nullptr){
            return Matrix::matrixApplyFunction(matrixA, matrixGrad, targetArena,
                [](T valA, T valGrad) {
                    return valGrad * (T(1.0) - (valA * valA));
                }
            );
        };

        static Matrix softmax(const Matrix& matrixZ, core::Arena* targetArena = nullptr){
            size_t zRows = matrixZ.getRows();
            size_t zCols = matrixZ.getCols();
            if (zRows == 0 || zCols == 0) {
                return Matrix(zRows, zCols, targetArena, false);
            }
            Matrix result(zRows, zCols, targetArena, false);
    
            const size_t jumpRowIn  = matrixZ.getRowJump();
            const size_t jumpColIn  = matrixZ.getColJump();
    
            const size_t jumpRowOut = result.getRowJump();

            const T* const values = matrixZ.getValues();
            T* const valuesOut = result.getValues();

            for (size_t r = 0; r < zRows; r++) {
                const T* const rowIn = &values[r * jumpRowIn];
                T* const rowOut = &valuesOut[r * jumpRowOut];

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
        static Matrix softmaxDerivative(const Matrix& matrixA, const Matrix& matrixGrad, core::Arena* targetArena = nullptr) {
            size_t aRows = matrixA.getRows();
            size_t aCols = matrixA.getCols();
            if (aRows == 0 || aCols == 0) return Matrix(aRows, aCols, targetArena, false);

            Matrix result(aRows, aCols, targetArena, false);

            const size_t jumpRowA    = matrixA.getRowJump();
            const size_t jumpColA    = matrixA.getColJump();
            const size_t jumpRowGrad = matrixGrad.getRowJump();
            const size_t jumpColGrad = matrixGrad.getColJump();
            const size_t jumpRowOut  = result.getRowJump();

            const T* const valuesA    = matrixA.getValues();
            const T* const valuesGrad = matrixGrad.getValues();
            T* const valuesOut        = result.getValues();

            for (size_t r = 0; r < aRows; r++) {
                const T* const rowA    = &valuesA[r * jumpRowA];
                const T* const rowGrad = &valuesGrad[r * jumpRowGrad];
                T* const rowOut        = &valuesOut[r * jumpRowOut];

                T dotProduct = T(0.0);
                for (size_t c = 0; c < aCols; c++) {
                    dotProduct += rowGrad[c * jumpColGrad] * rowA[c * jumpColA];
                }

                for (size_t c = 0; c < aCols; c++) {
                    T s_val = rowA[c * jumpColA];
                    T g_val = rowGrad[c * jumpColGrad];
                    rowOut[c] = s_val * (g_val - dotProduct);
                }
            }
            return result;
        }
    };
}
