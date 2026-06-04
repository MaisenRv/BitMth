#pragma once

#include <algorithm>
#include <cmath>

#include <BitMth/core/Matrix.hpp>

namespace Bitmth{
    namespace Math{

        template <typename T>
        void relu(Matrix<T>& matrix){
            for (size_t i = 0; i < matrix.numElements; i++) {
                matrix.m[i] = std::max( T(0.0), matrix.m[i]);
            }
        };

        template <typename T>
        void reluDerivative(Matrix<T>& matrix){
            for (size_t i = 0; i < matrix.numElements; i++) {
                matrix.m[i] = matrix.m[i] > T(0.0) ? T(1.0) : T(0.0);
            }
        };

        template <typename T>
        void sigmoid(Matrix<T>& matrix){
            for (size_t i = 0; i < matrix.numElements; i++) {
                matrix.m[i] = T(1.0) / ( T(1.0) + std::exp(-matrix.m[i]) );
            }
        };

        template <typename T>
        void sigmoidDerivative(Matrix<T>& matrix){
            for (size_t i = 0; i < matrix.numElements; i++) {
                matrix.m[i] = matrix.m[i] * ( T(1.0) - matrix.m[i] );
            }
        };

        template <typename T>
        void Tanh(Matrix<T>& matrix){
            for (size_t i = 0; i < matrix.numElements; i++) {
                matrix.m[i] = std::tanh(matrix.m[i]);
            }
        };

        template <typename T>
        void TanhDerivative(Matrix<T>& matrix){
            for (size_t i = 0; i < matrix.numElements; i++) {
                matrix.m[i] = T(1.0) - (matrix.m[i] * matrix.m[i]);
            }
        };

        template <typename T>
        Matrix<T> softmax(const Matrix<T>& matrix){
            Matrix<T> result(matrix.rows, matrix.cols);
            for (size_t r = 0; r < matrix.rows; r++) {
                size_t rowOffset = r * matrix.cols;

                T maxVal = matrix.m[rowOffset];
                for (size_t c = 1; c < matrix.cols; c++) {
                    maxVal = std::max(maxVal, matrix.m[rowOffset + c]);
                }
                
                T sum = T(0.0);
                for (size_t c = 0; c < matrix.cols; c++) {
                    size_t idx = rowOffset + c;
                    result.m[idx] = std::exp(matrix.m[idx] - maxVal);
                    sum += result.m[idx];
                }
                
                for (size_t c = 0; c < matrix.cols; c++) {
                    result.m[rowOffset + c] /= sum;
                }
            }
            
            return result;
        };

    }
}
