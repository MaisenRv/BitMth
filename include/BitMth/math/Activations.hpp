#pragma once

#include <algorithm>
#include <cmath>

#include <BitMth/core/Matrix.hpp>

namespace BitMth{
    namespace Math{
        enum class ActivationFunct: unsigned char {
            RELU,
            SIGMOID,
            TANH,
            SOFTMAX
        };

        template <typename T>
        Matrix<T> relu(const Matrix<T>& matrixZ){
            Matrix<T> result(matrixZ.rows, matrixZ.cols);
            for (size_t i = 0; i < matrixZ.numElements; i++) {
                result.m[i] = std::max( T(0.0), matrixZ.m[i]);
            }
            return result;
        };

        template <typename T>
        Matrix<T> reluDerivative(const Matrix<T>& matrixZ, const Matrix<T>& matrixA){
            Matrix<T> result(matrixZ.rows, matrixZ.cols);
            for (size_t i = 0; i < matrixZ.numElements; i++) {
                result.m[i] = matrixZ.m[i] > T(0.0) ? T(1.0) : T(0.0);
            }
            return result;
        };

        template <typename T>
        Matrix<T> sigmoid(const Matrix<T>& matrixZ){
            Matrix<T> result(matrixZ.rows, matrixZ.cols);
            for (size_t i = 0; i < matrixZ.numElements; i++) {
                result.m[i] = T(1.0) / ( T(1.0) + std::exp(-matrixZ.m[i]) );
            }
            return result;
        };

        template <typename T>
        Matrix<T> sigmoidDerivative(const Matrix<T>& matrixZ, const Matrix<T>& matrixA){
            Matrix<T> result(matrixA.rows, matrixA.cols);
            for (size_t i = 0; i < matrixA.numElements; i++) {
                result.m[i] = matrixA.m[i] * ( T(1.0) - matrixA.m[i] );
            }
            return result;
        };

        template <typename T>
        Matrix<T> Tanh(const Matrix<T>& matrixZ){
            Matrix<T> result(matrixZ.rows, matrixZ.cols);
            for (size_t i = 0; i < matrixZ.numElements; i++) {
                result.m[i] = std::tanh(matrixZ.m[i]);
            }
            return result;
        };

        template <typename T>
        Matrix<T> TanhDerivative(const Matrix<T>& matrixZ, const Matrix<T>& matrixA){
            Matrix<T> result(matrixA.rows, matrixA.cols);
            for (size_t i = 0; i < matrixA.numElements; i++) {
                result.m[i] = T(1.0) - (matrixA.m[i] * matrixA.m[i]);
            }
            return result;
        };

        template <typename T>
        Matrix<T> softmax(const Matrix<T>& matrixZ){
            Matrix<T> result(matrixZ.rows, matrixZ.cols);
            for (size_t r = 0; r < matrixZ.rows; r++) {
                size_t rowOffset = r * matrixZ.cols;

                T maxVal = matrixZ.m[rowOffset];
                for (size_t c = 1; c < matrixZ.cols; c++) {
                    maxVal = std::max(maxVal, matrixZ.m[rowOffset + c]);
                }
                
                T sum = T(0.0);
                for (size_t c = 0; c < matrixZ.cols; c++) {
                    size_t idx = rowOffset + c;
                    result.m[idx] = std::exp(matrixZ.m[idx] - maxVal);
                    sum += result.m[idx];
                }
                
                for (size_t c = 0; c < matrixZ.cols; c++) {
                    result.m[rowOffset + c] /= sum;
                }
            }
            
            return result;
        };

    }
}
