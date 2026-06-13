#pragma once

#include <cstddef>
#include <iostream>
#include <cmath>

#include <BitMth/utils/Errors.hpp>
#include <BitMth/utils/Constants.hpp>

namespace BitMth::linalg{
    template <typename T>
    struct Matrix{
        size_t rows{0}, cols{0};
        size_t numElements{0};
        T *m{nullptr};

        Matrix(size_t rows, size_t cols):
            rows(rows), cols(cols), numElements(rows * cols), m(new T[numElements]) { clear(); }

        Matrix() = default;
        
        ~Matrix(){ delete[] m; }
        
        // Copy matrix methods ---------------------------
        Matrix(const Matrix& inMatrix):
            rows(inMatrix.rows), cols(inMatrix.cols),numElements(inMatrix.numElements), m(new T[numElements]){
                for (size_t i = 0; i < numElements; i++) m[i] = inMatrix.m[i];
            }
        
        Matrix( Matrix&& inMatrix) noexcept :
            rows(inMatrix.rows), cols(inMatrix.cols),numElements(inMatrix.numElements), m(inMatrix.m){
                inMatrix.rows = 0;
                inMatrix.cols = 0;
                inMatrix.numElements = 0;
                inMatrix.m = nullptr;
            }

        Matrix& operator=(const Matrix& inMatrix){
            if(this == &inMatrix) return *this;

            if(rows != inMatrix.rows || cols != inMatrix.cols){
                delete[] m;
                m = new T[inMatrix.numElements];
            }
            rows = inMatrix.rows;
            cols = inMatrix.cols;
            numElements = inMatrix.numElements;

            for (size_t i = 0; i < numElements; i++) m[i] = inMatrix.m[i];
            
            return *this;
        }

        Matrix& operator=(Matrix&& inMatrix) noexcept {
            if (this == &inMatrix) return *this;

            delete[] m; 

            rows = inMatrix.rows;
            cols = inMatrix.cols;
            numElements = inMatrix.numElements;
            m = inMatrix.m;

            inMatrix.rows = 0;
            inMatrix.cols = 0;
            inMatrix.numElements = 0;
            inMatrix.m = nullptr;

            return *this;
        }


        T& operator()(size_t r, size_t c) {
            CHECK_ERROR_MATRIX(
                r >= rows || c >= cols,
                "operator ()",
                "Outside range ( row: " << r << " , col: " << c << " )"
            );
            return m[r * cols + c]; 
        }
        const T& operator()(size_t r, size_t c) const { 
            CHECK_ERROR_MATRIX(
                r >= rows || c >= cols,
                "operator ()",
                "Outside range ( row: " << r << " , col: " << c << " )"
            );
            return m[r * cols + c]; 
        }
        
        // Operators Matrix - number ---------------------------
        [[nodiscard]] Matrix<T> operator+(T number) const {
            Matrix<T> result(*this);
            for (size_t i = 0; i < numElements; i++) result.m[i] += number;
            return result;
        }
        [[nodiscard]] friend Matrix<T> operator+(T scalar, const Matrix<T>& matrix) {
            return matrix + scalar;
        }
        Matrix& operator+=(T number) {
            for (size_t i = 0; i < numElements; i++) m[i] += number;
            return *this;
        }

        [[nodiscard]] Matrix<T> operator-(T number) const {
            Matrix<T> result(*this);
            for (size_t i = 0; i < numElements; i++) result.m[i] -= number;
            return result;
        }
        [[nodiscard]] friend Matrix<T> operator-(T scalar, const Matrix<T>& matrix) {
            return matrix - scalar;
        }
        Matrix& operator-=(T number) {
            for (size_t i = 0; i < numElements; i++) m[i] -= number;
            return *this;
        }

        [[nodiscard]] Matrix<T> operator*(T number) const {
            Matrix<T> result(*this);
            for (size_t i = 0; i < numElements; i++) result.m[i] *= number;
            return result;
        }
        [[nodiscard]] friend Matrix<T> operator*(T scalar, const Matrix<T>& matrix) {
            return matrix * scalar;
        }
        Matrix& operator*=(T number) {
            for (size_t i = 0; i < numElements; i++) m[i] *= number;
            return *this;
        }

        // Operators Matrix - Matrix ---------------------------
        [[nodiscard]] Matrix<T> operator+(const Matrix<T>& matrix) const {
            CHECK_ERROR_MATRIX(
                rows != matrix.rows || cols != matrix.cols,
                "matrix add (+)",
                "Matrix dimensions must match (rows = rows && cols == cols)"
            );
            Matrix<T> result(*this);
            for (size_t i = 0; i < numElements; i++) result.m[i] += matrix.m[i];
            return result;
        }
        Matrix& operator+=(const Matrix<T>& matrix) {
            CHECK_ERROR_MATRIX(
                rows != matrix.rows || cols != matrix.cols,
                "matrix add (+=)",
                "Matrix dimensions must match (rows = rows && cols == cols)"
            );
            for (size_t i = 0; i < numElements; i++) m[i] += matrix.m[i];
            return *this;
        }

        [[nodiscard]] Matrix<T> operator-(const Matrix<T>& matrix) const {
            CHECK_ERROR_MATRIX(
                rows != matrix.rows || cols != matrix.cols,
                "matrix subtract (-)",
                "Matrix dimensions must match (rows = rows && cols == cols)"
            );
            Matrix<T> result(*this);
            for (size_t i = 0; i < numElements; i++) result.m[i] -= matrix.m[i];
            return result;
        }
        Matrix& operator-=(const Matrix<T>& matrix) {
            CHECK_ERROR_MATRIX(
                rows != matrix.rows || cols != matrix.cols,
                "matrix subtract (-=)",
                "Matrix dimensions must match (rows = rows && cols == cols)"
            );
            for (size_t i = 0; i < numElements; i++) m[i] -= matrix.m[i];
            return *this;
        }

        [[nodiscard]] Matrix<T> operator*(const Matrix<T>& matrix) const {
            CHECK_ERROR_MATRIX(
                cols != matrix.rows,
                "matrix multiplication (*)",
                "Matrix dimensions must match rows = cols"
            );

            Matrix<T> result(rows, matrix.cols);

            for (size_t i = 0; i < rows; i++) {
                for (size_t k = 0; k < cols; k++) {
                    T factor = m[i * cols + k]; 
                    
                    for (size_t j = 0; j < matrix.cols; j++) {
                        result.m[i * matrix.cols + j] += factor * matrix.m[k * matrix.cols + j]; 
                    }
                }
            }
            return result;
        }

        Matrix<T>& operator*=(const Matrix<T>& matrix) {
            *this = *this * matrix;
            return *this;
        }

        [[nodiscard]] Matrix<T> t() const {
            Matrix<T> result(cols, rows);
            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) result.m[j * rows + i] = m[i * cols + j];
            }
            return result;
        }

        Matrix<T>& powInPlace(T exponent) {
            if (exponent == T(2)) {
                for (size_t i = 0; i < numElements; i++) m[i] = m[i] * m[i];
                return *this;
            }
            for (size_t i = 0; i < numElements; i++) m[i] = std::pow(m[i], exponent);
            return *this;
        }
        [[nodiscard]] Matrix<T> pow(T exponent) const {
            Matrix<T> result(*this);
            result.powInPlace(exponent);
            return result;
        }

        Matrix& hadamard(const Matrix<T>& matrix){
            CHECK_ERROR_MATRIX(
                rows != matrix.rows || cols != matrix.cols,
                "matrix hadamard product ( (·) )",
                "Matrix dimensions must match (rows = rows && cols == cols)"
            );   
            for (size_t i = 0; i < numElements; i++) m[i] *= matrix.m[i];
            return *this;
        }

        [[nodiscard]] Matrix<T> operator/(const Matrix<T>& other) const {
            CHECK_ERROR_MATRIX(
                rows != other.rows || cols != other.cols,
                "Matrix operator/",
                "Dimensions must match for element-wise division"
            );
            Matrix<T> result(*this);
            for (size_t i = 0; i < numElements; i++) {
                if (std::abs(other.m[i]) > utils::EPSILON<T>) {
                    result.m[i] = m[i] / other.m[i];
                    continue;
                } 
                result.m[i] = T(0); 
            }
            return result;
        }

        [[nodiscard]] Matrix<T> reduceSumCols() const{
            Matrix<T> result(rows, 1);
            for (size_t i = 0; i < rows; i++){
                for (size_t j = 0; j < cols; j++){
                    result.m[i] += m[i * cols + j];  
                }
            }
            return result;
        }

        [[nodiscard]] Matrix<T> reduceSumRows() const{
            Matrix<T> result(1, cols);
            for (size_t i = 0; i < rows; i++){
                for (size_t j = 0; j < cols; j++){
                    result.m[j] += m[i * cols + j];  
                }
            }
            return result;
        }

        [[nodiscard]] T reduceSumTotal() const {
            T result = 0;
            for (size_t i = 0; i < numElements; i++) result += m[i];
            return result;
        }


        [[nodiscard]] Matrix<T> addRowVector(const Matrix<T>& rowVector) const {
            CHECK_ERROR_MATRIX(
                rowVector.rows != 1 || rowVector.cols != cols,
                "Matrix operator (addRowVector)",
                "Dimensions mismatch"
            );
            Matrix<T> result(rows, cols);
            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                    size_t index = i * cols + j;
                    result.m[index] = m[index] + rowVector.m[j];
                }
            }
            return result;
        }

        // Utils
        void clear(){ for (size_t i = 0; i < numElements; i++) m[i] = 0; }
        
        void print() const {
            std::cout << "\n";
            for (size_t i = 0; i < rows; i++) {
                std::cout << "\t|";
                for (size_t j = 0; j < cols; j++) {
                    if (j != cols - 1 ){
                        std::cout << (*this)(i,j) << "\t";
                        continue;
                    }
                    std::cout << (*this)(i,j);
                }
                std::cout << "|" << std::endl;
            }
            std::cout << "\n";  
        }
    };
}
