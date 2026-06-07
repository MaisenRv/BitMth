#pragma once

#include <cstddef>
#include <iostream>
#include <cmath>

#include <BitMth/utils/Errors.hpp>

namespace BitMth{
    namespace linalg{
        template <typename T>
        struct Matrix{
            size_t rows, cols;
            size_t numElements;
            T *m;

            Matrix(size_t rows, size_t cols):
                rows(rows), cols(cols), numElements(rows * cols), m(new T[numElements]) { clear(); }

            Matrix() : rows(0), cols(0), numElements(0), m(nullptr) {}
            
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

                if(inMatrix.numElements != numElements){
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
            Matrix<T> operator+(T number) const {
                Matrix<T> result(*this);
                for (size_t i = 0; i < numElements; i++) result.m[i] += number;
                return result;
            }
            friend Matrix<T> operator+(T scalar, const Matrix<T>& matrix) {
                return matrix + scalar;
            }
            Matrix& operator+=(T number) {
                for (size_t i = 0; i < numElements; i++) m[i] += number;
                return *this;
            }

            Matrix<T> operator-(T number) const {
                Matrix<T> result(*this);
                for (size_t i = 0; i < numElements; i++) result.m[i] -= number;
                return result;
            }
            friend Matrix<T> operator-(T scalar, const Matrix<T>& matrix) {
                return matrix - scalar;
            }
            Matrix& operator-=(T number) {
                for (size_t i = 0; i < numElements; i++) m[i] -= number;
                return *this;
            }

            Matrix<T> operator*(T number) const {
                Matrix<T> result(*this);
                for (size_t i = 0; i < numElements; i++) result.m[i] *= number;
                return result;
            }
            friend Matrix<T> operator*(T scalar, const Matrix<T>& matrix) {
                return matrix * scalar;
            }
            Matrix& operator*=(T number) {
                for (size_t i = 0; i < numElements; i++) m[i] *= number;
                return *this;
            }

            // Operators Matrix - Matrix ---------------------------
            Matrix<T> operator+(const Matrix<T>& matrix) const {
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

            Matrix<T> operator-(const Matrix<T>& matrix) const {
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

            Matrix<T> operator*(const Matrix<T>& matrix) const {
                CHECK_ERROR_MATRIX(
                    cols != matrix.rows,
                    "matrix multiplication (*)",
                    "Matrix dimensions must match rows = cols"
                );

                Matrix<T> result(rows, matrix.cols);

                for (size_t i = 0; i < rows; i++) {
                    for (size_t k = 0; k < cols; k++) {
                        T factor = (*this)(i, k); 
                        
                        for (size_t j = 0; j < matrix.cols; j++) {
                            result(i, j) += factor * matrix(k, j); 
                        }
                    }
                }
                return result;
            }

            Matrix<T>& operator*=(const Matrix<T>& matrix) {
                CHECK_ERROR_MATRIX(
                    cols != matrix.rows,
                    "matrix multiplication (*=)",
                    "Matrix dimensions must match rows = cols"
                );

                Matrix<T> result(rows, matrix.cols);

                for (size_t i = 0; i < rows; i++) {
                    for (size_t k = 0; k < cols; k++) {
                        T factor = (*this)(i, k); 
                        
                        for (size_t j = 0; j < matrix.cols; j++) {
                            result(i, j) += factor * matrix(k, j); 
                        }
                    }
                }

                (*this) = result;
                return *this;
            }

            Matrix<T> t() const {
                Matrix<T> result(cols, rows);
                for (size_t i = 0; i < rows; i++) {
                    for (size_t j = 0; j < cols; j++) result(j,i) = (*this)(i,j);
                }
                return result;
            }

            Matrix<T>& powInPlace(T exponent) {
                for (size_t i = 0; i < numElements; i++) m[i] = std::pow(m[i], exponent);
                return *this;
            }
            Matrix<T> pow(T exponent) const {
                Matrix<T> result(*this);
                for (size_t i = 0; i < numElements; i++) result.m[i] = std::pow(result.m[i], exponent);
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

            Matrix<T> operator/(const Matrix<T>& other) const {
                CHECK_ERROR_MATRIX(
                    rows != other.rows || cols != other.cols,
                    "Matrix operator/",
                    "Dimensions must match for element-wise division"
                );
                Matrix<T> result(*this);
                for (size_t i = 0; i < numElements; i++) {
                    result.m[i] /= other.m[i];
                }
                return result;
            }

            Matrix<T> reduceSumCols() const{
                Matrix<T> result(rows, 1);
                for (size_t i = 0; i < rows; i++){
                    for (size_t j = 0; j < cols; j++){
                        result.m[i] += m[i * cols + j];  
                    }
                }
                return result;
            }

            Matrix<T> reduceSumRows() const{
                Matrix<T> result(1, cols);
                for (size_t i = 0; i < rows; i++){
                    for (size_t j = 0; j < cols; j++){
                        result.m[j] += m[i * cols + j];  
                    }
                }
                return result;
            }

            T reduceSumTotal() const {
                T result = 0;
                for (size_t i = 0; i < numElements; i++) result += m[i];
                return result;
            }


            Matrix<T> addRowVector(const Matrix<T>& rowVector) const {
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
}
