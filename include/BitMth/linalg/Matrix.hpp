#pragma once

#include <cstddef>
#include <iostream>
#include <cmath>
#include <utility>
#include <cstring>
#include <iomanip>

#include <BitMth/utils/Errors.hpp>
#include <BitMth/utils/Constants.hpp>
#include <BitMth/core/Arena.hpp>


namespace BitMth::linalg{
    template <typename T>
    class Matrix{
    private:
        Matrix(size_t rows, size_t cols, const size_t *customStrides, core::Arena* arena): Matrix(rows, cols, arena, false){
            stride[0] = customStrides[0];
            stride[1] = customStrides[1];
        }

        template< typename Op>
        [[nodiscard]] static Matrix<T> _scalarApplyFunction(const Matrix<T> &matrix,T scalar, core::Arena *arenaContainer, Op funct){
            Matrix<T> result(matrix.rows, matrix.cols, arenaContainer, false);
            for (size_t i = 0; i < matrix.numElements; i++){
                result.m[i] = funct(matrix.m[i], scalar);
            }
            return result;
        }

        template< typename Op>
        [[nodiscard]] static Matrix<T> _matrixApplyFunction(const Matrix<T> &matrixA, const Matrix<T> &matrixB, core::Arena *arenaContainer, Op funct){
            Matrix<T> result(matrixA.rows, matrixA.cols, arenaContainer, false);
            const size_t aJumpRow = matrixA.stride[0] / sizeof(T);
            const size_t aJumpCol = matrixA.stride[1] / sizeof(T);
            const size_t bJumpRow = matrixB.stride[0] / sizeof(T);
            const size_t bJumpCol = matrixB.stride[1] / sizeof(T);
            for (size_t i = 0; i < matrixA.rows; i++) {
                for (size_t j = 0; j < matrixA.cols; j++) {
                    T valA = matrixA.m[i * aJumpRow + j * aJumpCol];
                    T valB = matrixB.m[i * bJumpRow+ j * bJumpCol];
                    result.m[i * result.cols + j] = funct(valA, valB);
                }
            }

            return result;
        }

        size_t rows{0}, cols{0}, numElements{0};
        size_t stride[2]{};
        T *m{nullptr};
        core::Arena *arena{nullptr};
    public:
        Matrix(size_t rows, size_t cols, core::Arena *arenaContainer = nullptr, bool initializeData = true)
            :rows(rows), cols(cols), numElements(rows * cols), arena(arenaContainer){
            if (arena != nullptr){
                void *ptrData = arena->alloc(numElements * sizeof(T), alignof(T));
                m = reinterpret_cast<T*>(ptrData);
            }else{
                m = new T[numElements];
            }
            stride[0] = cols * sizeof(T);
            stride[1] = sizeof(T);
            if(initializeData){
                clear(); 
            }
        }

        Matrix() = default;
        
        ~Matrix(){
            if (arena == nullptr){
                delete[] m; 
            }
        }
        
        // Copy matrix methods ---------------------------
        Matrix(const Matrix& inMatrix):
            rows(inMatrix.rows), cols(inMatrix.cols),numElements(inMatrix.numElements), m(new T[numElements]){
                stride[0] = inMatrix.stride[0];
                stride[1] = inMatrix.stride[1];
                for (size_t i = 0; i < rows; i++){
                    for (size_t j = 0; j < cols; j++){
                        (*this)(i,j) = inMatrix(i,j); 
                    }
                }
            }

        [[nodiscard]] Matrix<T> clone(core::Arena *arenaContainer) const {
            Matrix<T> copy(rows,cols,arenaContainer,false);
            const size_t srcRowJump = stride[0] / sizeof(T);
            if (srcRowJump == cols) {
                std::memcpy(copy.m, m, numElements * sizeof(T));
            } else {
                for (size_t i = 0; i < rows; ++i) {
                    std::memcpy(&copy.m[i * cols], &m[i * srcRowJump], cols * sizeof(T));
                }
            }
            return copy;
        }
        
        Matrix( Matrix&& inMatrix) noexcept :
            rows(inMatrix.rows), cols(inMatrix.cols),numElements(inMatrix.numElements), m(inMatrix.m), arena(inMatrix.arena){
                stride[0] = inMatrix.stride[0];
                stride[1] = inMatrix.stride[1];
                inMatrix.stride[0] = 0;
                inMatrix.stride[1] = 0;
                inMatrix.rows = 0;
                inMatrix.cols = 0;
                inMatrix.numElements = 0;
                inMatrix.m = nullptr;
                inMatrix.arena = nullptr;
            }

        Matrix& operator=(const Matrix& inMatrix){
            if(this == &inMatrix) return *this;

            if(rows != inMatrix.rows || cols != inMatrix.cols){
                if(arena == nullptr){
                    delete[] m;
                }
                arena = nullptr;
                m = new T[inMatrix.numElements];
                
                rows = inMatrix.rows;
                cols = inMatrix.cols;
                numElements = inMatrix.numElements;
            }
            
            stride[0] = inMatrix.stride[0];
            stride[1] = inMatrix.stride[1];
            for (size_t i = 0; i < rows; i++){
                for (size_t j = 0; j < cols; j++){
                    (*this)(i,j) = inMatrix(i,j); 
                }
            }
            return *this;
        }

        Matrix& operator=(Matrix&& inMatrix) noexcept {
            if (this == &inMatrix) return *this;

            if (arena == nullptr) delete[] m;

            arena = inMatrix.arena;
            rows = inMatrix.rows;
            cols = inMatrix.cols;
            numElements = inMatrix.numElements;
            m = inMatrix.m;
            stride[0] = inMatrix.stride[0];
            stride[1] = inMatrix.stride[1];

            inMatrix.rows = 0;
            inMatrix.cols = 0;
            inMatrix.numElements = 0;
            inMatrix.m = nullptr;
            inMatrix.arena = nullptr;
            inMatrix.stride[0] = 0;
            inMatrix.stride[1] = 0;

            return *this;
        }

        T& operator()(size_t r, size_t c) {
            CHECK_ERROR_MATRIX(
                r >= rows || c >= cols,
                "operator ()",
                "Outside range ( row: " << r << " , col: " << c << " )"
            );
            uint8_t *ptr = reinterpret_cast<uint8_t*>(m);
            return *reinterpret_cast<T*>(ptr + (r * stride[0] + c * stride[1]));
        }

        const T& operator()(size_t r, size_t c) const { 
            CHECK_ERROR_MATRIX(
                r >= rows || c >= cols,
                "operator ()",
                "Outside range ( row: " << r << " , col: " << c << " )"
            );
            const uint8_t *ptr = reinterpret_cast<const uint8_t*>(m);
            return *reinterpret_cast<const T*>(ptr + (r * stride[0] + c * stride[1]));
        }
        
        // MATRIX - SCALAR

        // ----------- ADDITION OPERATOR -----------
        Matrix& operator+=(T scalar) {
            for (size_t i = 0; i < numElements; i++) m[i] += scalar;
            return *this;
        }
        [[nodiscard]] Matrix<T> operator+(T scalar) const {
            return Matrix<T>::_scalarApplyFunction(*this, scalar, nullptr, [](T element, T sca){ return element + sca; });
        }
        [[nodiscard]] friend Matrix<T> operator+(T scalar, const Matrix<T>& matrix) {
            return matrix + scalar;
        }
        [[nodiscard]] static Matrix<T> add(const Matrix<T>& matrix, T scalar, core::Arena* targetArena) {
            return Matrix<T>::_scalarApplyFunction(matrix, scalar, targetArena, [](T element, T sca){ return element + sca; });
        }
        [[nodiscard]] static Matrix<T> add(T scalar, const Matrix<T>& matrix, core::Arena* targetArena) {
            return Matrix<T>::add(matrix, scalar, targetArena);
        }
        // ------------------------------------------

        // ----------- SUBTRACTION OPERATOR -----------
        Matrix& operator-=(T scalar) {
            for (size_t i = 0; i < numElements; i++) m[i] -= scalar;
            return *this;
        }
        [[nodiscard]] Matrix<T> operator-(T scalar) const {
            return Matrix<T>::_scalarApplyFunction(*this, scalar, nullptr, [](T element, T sca){ return element - sca; });
        }
        [[nodiscard]] friend Matrix<T> operator-(T scalar, const Matrix<T>& matrix) {
            return Matrix<T>::_scalarApplyFunction(matrix, scalar, nullptr, [](T element, T sca){ return sca - element; });
        }
        [[nodiscard]] static Matrix<T> sub(const Matrix<T>& matrix, T scalar, core::Arena* targetArena) {
            return Matrix<T>::_scalarApplyFunction(matrix, scalar, targetArena, [](T element, T sca){ return element - sca; });
        }
        [[nodiscard]] static Matrix<T> sub(T scalar,const Matrix<T>& matrix, core::Arena* targetArena) {
            return Matrix<T>::_scalarApplyFunction(matrix, scalar, targetArena, [](T element, T sca){ return sca - element; });
        }
        // ------------------------------------------

        // ----------- MULTIPLICATION OPERATOR -----------
        Matrix& operator*=(T scalar) {
            for (size_t i = 0; i < numElements; i++) m[i] *= scalar;
            return *this;
        }
        [[nodiscard]] Matrix<T> operator*(T scalar) const {
            return Matrix<T>::_scalarApplyFunction(*this, scalar, nullptr, [](T element, T sca){ return element * sca; });
        }
        [[nodiscard]] friend Matrix<T> operator*(T scalar, const Matrix<T>& matrix) {
            return matrix * scalar;
        }
        [[nodiscard]] static Matrix<T> mul(const Matrix<T>& matrix, T scalar, core::Arena* targetArena) {
            return Matrix<T>::_scalarApplyFunction(matrix, scalar, targetArena, [](T element, T sca){ return element * sca; });
        }
        [[nodiscard]] static Matrix<T> mul(T scalar, const Matrix<T>& matrix, core::Arena* targetArena) {
            return Matrix<T>::mul(matrix, scalar, targetArena);
        }
        // ------------------------------------------

        // ----------- DIVISION OPERATOR -----------
        Matrix& operator/=(T scalar) {
            CHECK_ERROR_MATRIX(
                scalar < utils::EPSILON<T> && scalar > -utils::EPSILON<T>,
                "Matrix operator/= scalar",
                "Division by zero"
            );
            T aux = T(1) / scalar;
            return *this *= aux;
        }
        [[nodiscard]] Matrix<T> operator/(T scalar) const {
            CHECK_ERROR_MATRIX(
                scalar < utils::EPSILON<T> && scalar > -utils::EPSILON<T>,
                "Matrix operator/ scalar",
                "Division by zero"
            );
            T aux = T(1) / scalar;
            return *this * aux;
        }
        [[nodiscard]] static Matrix<T> div(const Matrix<T>& matrix, T scalar, core::Arena* targetArena) {
            CHECK_ERROR_MATRIX(
                scalar < utils::EPSILON<T> && scalar > -utils::EPSILON<T>,
                "Matrix operator/ scalar",
                "Division by zero"
            );
            T aux = T(1) / scalar;
            return Matrix<T>::mul(matrix,aux,targetArena);
        }
        // ------------------------------------------

        // ----------- POW OPERATION -----------
        Matrix<T>& powInPlace(T exponent) {
            if(exponent < utils::EPSILON<T> && exponent > -utils::EPSILON<T>){ 
                setOne();
            }else if (exponent == T(2)) {
                for (size_t i = 0; i < numElements; i++) m[i] *= m[i];
            }else{
                for (size_t i = 0; i < numElements; i++) m[i] = std::pow(m[i], exponent);
            }
            return *this;
        }
        [[nodiscard]] Matrix<T> pow(T exponent, core::Arena* targetArena = nullptr) const {
            if(exponent < utils::EPSILON<T> && exponent > -utils::EPSILON<T>){
                Matrix<T> result(rows, cols, stride, targetArena);
                result.setOne();
                return result;
            }
            if (exponent == T(2)) {
                return Matrix<T>::_scalarApplyFunction(*this, exponent, targetArena, [](T element, T exp){ return element * element; });
            }
            return Matrix<T>::_scalarApplyFunction(*this, exponent, targetArena, [](T element, T exp){ return std::pow(element, exp); });
        }
        // ------------------------------------------

        // MATRIX - MATRIX
        
        // ----------- ADDITION OPERATOR -----------
        Matrix& operator+=(const Matrix<T>& matrix) {
            CHECK_ERROR_MATRIX(
                rows != matrix.rows || cols != matrix.cols,
                "matrix add (+=)",
                "Matrix dimensions must match (rows = rows && cols == cols)"
            );
            const size_t rowJumpThis = stride[0] / sizeof(T);
            const size_t colJumpThis = stride[1] / sizeof(T);
    
            const size_t rowJumpOther = matrix.stride[0] / sizeof(T);
            const size_t colJumpOther = matrix.stride[1] / sizeof(T);

            for (size_t i = 0; i < rows; i++) {
                T* const rowThis = &m[i * rowJumpThis];
                const T* const rowOther = &matrix.m[i * rowJumpOther];

                for (size_t j = 0; j < cols; j++) {
                    rowThis[j * colJumpThis] += rowOther[j * colJumpOther];
                }
            }

            return *this;
        }
        [[nodiscard]] Matrix<T> operator+(const Matrix<T>& matrix) const {
            CHECK_ERROR_MATRIX(
                rows != matrix.rows || cols != matrix.cols,
                "matrix add (+)",
                "Matrix dimensions must match (rows = rows && cols == cols)"
            );
            return Matrix<T>::_matrixApplyFunction(*this, matrix,nullptr, [](T mA, T mB){ return mA + mB; });
        }
        [[nodiscard]] static Matrix<T> add(const Matrix<T>& matrixA, const Matrix<T>& matrixB, core::Arena* targetArena){
            CHECK_ERROR_MATRIX(
                matrixA.rows != matrixB.rows || matrixA.cols != matrixB.cols,
                "Matrix::add",
                "Matrix dimensions must match (rows = rows && cols == cols)"
            );
            return Matrix<T>::_matrixApplyFunction(matrixA, matrixB, targetArena, [](T mA, T mB){ return mA + mB; });
        }
        // ------------------------------------------

         // ----------- SUBTRACTION OPERATOR -----------
        Matrix& operator-=(const Matrix<T>& matrix) {
            CHECK_ERROR_MATRIX(
                rows != matrix.rows || cols != matrix.cols,
                "matrix sub (-=)",
                "Matrix dimensions must match (rows == rows && cols == cols)"
            );
            const size_t rowJumpThis = stride[0] / sizeof(T);
            const size_t colJumpThis = stride[1] / sizeof(T);
    
            const size_t rowJumpOther = matrix.stride[0] / sizeof(T);
            const size_t colJumpOther = matrix.stride[1] / sizeof(T);

            for (size_t i = 0; i < rows; i++) {
                T* const rowThis = &m[i * rowJumpThis];
                const T* const rowOther = &matrix.m[i * rowJumpOther];

                for (size_t j = 0; j < cols; j++) {
                    rowThis[j * colJumpThis] -= rowOther[j * colJumpOther];
                }
            }

            return *this;
        }
        [[nodiscard]] Matrix<T> operator-(const Matrix<T>& matrix) const {
            CHECK_ERROR_MATRIX(
                rows != matrix.rows || cols != matrix.cols,
                "matrix sub (-)",
                "Matrix dimensions must match (rows == rows && cols == cols)"
            );
            return Matrix<T>::_matrixApplyFunction(*this, matrix, nullptr, [](T mA, T mB){ return mA - mB; });
        }
        [[nodiscard]] static Matrix<T> sub(const Matrix<T>& matrixA, const Matrix<T>& matrixB, core::Arena* targetArena){
            CHECK_ERROR_MATRIX(
                matrixA.rows != matrixB.rows || matrixA.cols != matrixB.cols,
                "Matrix::sub",
                "Matrix dimensions must match (rows == rows && cols == cols)"
            );
            return Matrix<T>::_matrixApplyFunction(matrixA, matrixB, targetArena, [](T mA, T mB){ return mA - mB; });
        }
        // ------------------------------------------

        // ----------- MULTIPLICATION OPERATOR -----------
        Matrix<T>& operator*=(const Matrix<T>& matrix) {
            *this = *this * matrix;
            return *this;
        }
        [[nodiscard]] Matrix<T> operator*(const Matrix<T>& matrix) const {
            CHECK_ERROR_MATRIX(
                cols != matrix.rows,
                "matrix multiplication (*)",
                "Matrix dimensions must match rows = cols"
            );

            Matrix<T> result(rows, matrix.cols);
            const size_t lhsRowStride = stride[0] / sizeof(T);
            const size_t lhsColStride = stride[1] / sizeof(T);

            const size_t rhsRowStride = matrix.stride[0] / sizeof(T);
            const size_t rhsColStride = matrix.stride[1] / sizeof(T);

            for (size_t i = 0; i < rows; ++i) {
                T* const resRow = &result.m[i * result.cols]; 

                for (size_t k = 0; k < cols; ++k) {
                    const T factor = m[i * lhsRowStride + k * lhsColStride];
        
                    const T* const rhsRowPtr = &matrix.m[k * rhsRowStride];

                    for (size_t j = 0; j < matrix.cols; ++j) {
                        resRow[j] += factor * rhsRowPtr[j * rhsColStride];
                    }
                }
            }

            return result;
        }
        [[nodiscard]] static Matrix<T> mul(const Matrix<T>& matrixA, const Matrix<T>& matrixB, core::Arena* targetArena) {
            CHECK_ERROR_MATRIX(
                matrixA.cols != matrixB.rows,
                "matrix::mul",
                "Matrix dimensions must match rows = cols"
            );

            Matrix<T> result(matrixA.rows, matrixB.cols, targetArena);
            const size_t aRowStride = matrixA.stride[0] / sizeof(T);
            const size_t aColStride = matrixA.stride[1] / sizeof(T);
    
            const size_t bRowStride = matrixB.stride[0] / sizeof(T);
            const size_t bColStride = matrixB.stride[1] / sizeof(T);

            for (size_t i = 0; i < matrixA.rows; ++i) {
                T* const resRow = &result.m[i * matrixB.cols];

                for (size_t k = 0; k < matrixA.cols; ++k) {
                    const T factor = matrixA.m[i * aRowStride + k * aColStride];
            
                    const T* const bRowPtr = &matrixB.m[k * bRowStride];

                    for (size_t j = 0; j < matrixB.cols; ++j) {
                        resRow[j] += factor * bRowPtr[j * bColStride];
                    }
                }
            }

            return result;
        }
        Matrix& hadamardInPlace(const Matrix<T>& matrix){
            CHECK_ERROR_MATRIX(
                rows != matrix.rows || cols != matrix.cols,
                "matrix hadamard product ( (·) )",
                "Matrix dimensions must match (rows = rows && cols == cols)"
            );
            const size_t rowJumpThis = stride[0] / sizeof(T);
            const size_t colJumpThis = stride[1] / sizeof(T);
            
            const size_t rowJumpOther = matrix.stride[0] / sizeof(T);
            const size_t colJumpOther = matrix.stride[1] / sizeof(T);

            for (size_t i = 0; i < rows; i++) {
                T* const rowThis = &m[i * rowJumpThis];
                const T* const rowOther = &matrix.m[i * rowJumpOther];

                for (size_t j = 0; j < cols; j++) {
                    rowThis[j * colJumpThis] *= rowOther[j * colJumpOther];
                }
            }
            return *this;
        }
        [[nodiscard]] static Matrix<T> hadamard(const Matrix<T>& matrixA, const Matrix<T>& matrixB, core::Arena* targetArena){
            CHECK_ERROR_MATRIX(
                matrixA.rows != matrixB.rows || matrixA.cols != matrixB.cols,
                "matrix hadamard product ( (·) )",
                "Matrix dimensions must match (rows = rows && cols == cols)"
            );
            return Matrix<T>::_matrixApplyFunction(matrixA, matrixB, targetArena, [](T mA, T mB){ return mA * mB; });
        }
        // ------------------------------------------

        Matrix& tInPlace(){
            std::swap(rows,cols);
            std::swap(stride[0],stride[1]);
            return *this;
        } 

        [[nodiscard]] static Matrix<T> t(const Matrix<T>& matrix, core::Arena* targetArena = nullptr) {
            Matrix<T> result = matrix.clone(targetArena);
            result.tInPlace();
            return result;
        }

        [[nodiscard]] Matrix<T> operator/(const Matrix<T>& matrix) const {
            CHECK_ERROR_MATRIX(
                rows != matrix.rows || cols != matrix.cols,
                "Matrix operator/",
                "Dimensions must match for element-wise division"
            );

            for (size_t i = 0; i < matrix.numElements; i++) {
                CHECK_ERROR_MATRIX(
                    matrix.m[i] < utils::EPSILON<T> && matrix.m[i] > -utils::EPSILON<T>,
                    "Matrix operator/ (element-wise)",
                    "Division by zero: One or more elements in the divisor matrix are zero"
                );
            }
            return Matrix<T>::_matrixApplyFunction(*this, matrix, nullptr, [](T mA, T mB){ 
                return mA / mB; 
            });
        }

        [[nodiscard]] bool operator==(const Matrix<T>& matrix) const {
            if(rows != matrix.getRows() || 
               matrix.getCols() != cols || 
               matrix.size() != numElements ||
               matrix.getStrides()[0] != stride[0] ||
               matrix.getStrides()[1] != stride[1]
            ){ return false; }

            for (size_t i = 0; i < rows; i++){
                for (size_t j = 0; j < cols; j++){
                    if((*this)(i,j) != matrix(i,j)) return false;
                }
            }
            return true;
        }

        [[nodiscard]] bool isApprox(const Matrix<T>& matrix, T margin = utils::EPSILON<T>) const {
            if (rows != matrix.rows || cols != matrix.cols) return false;
            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                    T diff = (*this)(i, j) - matrix(i, j);
                    if (diff < 0) diff = -diff;
                    if (diff > margin) return false;
                }
            }
            return true;
        }

        [[nodiscard]] Matrix<T> reduceSumCols() const{
            Matrix<T> result(rows, 1);
            const size_t rowJump = stride[0] / sizeof(T);

            for (size_t i = 0; i < rows; i++){
                T sum = 0;
                const T* const currentRow = &m[i * rowJump];

                for (size_t j = 0; j < cols; j++){
                    sum += currentRow[j];  
                }
                result.m[i] = sum;
            }
            return result;
        }
        [[nodiscard]] Matrix<T> reduceSumRows() const{
            Matrix<T> result(1, cols);
            const size_t rowJump = stride[0] / sizeof(T);

            for (size_t i = 0; i < rows; i++){
                const T* const currentRow = &m[i * rowJump];
                for (size_t j = 0; j < cols; j++){
                    result.m[j] += currentRow[j];
                }
            }
            return result;
        }
        [[nodiscard]] T reduceSumTotal() const {
            T result = 0;
            for (size_t i = 0; i < numElements; i++) result += m[i];
            return result;
        }

        [[nodiscard]] Matrix<T> addRowVector(const Matrix<T>& rowVector, core::Arena* targetArena = nullptr) const {
            CHECK_ERROR_MATRIX(
                rowVector.rows != 1 || rowVector.cols != cols,
                "Matrix operator (addRowVector)",
                "Dimensions mismatch"
            );
            Matrix<T> result(rows, cols, stride, targetArena);
            const size_t rowJump = stride[0] / sizeof(T);
            const size_t resRowJump = result.stride[0] / sizeof(T);

            for (size_t i = 0; i < rows; i++) {
                const T* const currentRow = &m[i * rowJump];
                T* const resCurrentRow = &result.m[i * resRowJump];

                for (size_t j = 0; j < cols; j++) {
                    resCurrentRow[j] = currentRow[j] + rowVector.m[j];
                }
            }
            return result;
        }

        // Utils
        void clear(){ std::fill_n(m, numElements, T(0)); }
        void setOne(){ std::fill_n(m, numElements, T(1)); }
        void setWith(T number){ std::fill_n(m, numElements, number); }
        void setIdentity(){
            CHECK_ERROR_MATRIX(
                rows != cols,
                "setIdentity",
                "Matrix dimensions must match (rows == cols)"
            );
            clear();
            const size_t rowJump = stride[0] / sizeof(T);
            for(size_t i = 0; i < rows; i++){
                T* const currentRow = &m[i * rowJump];
                currentRow[i] = 1;
            }
        }

        friend std::ostream& operator<<(std::ostream& os, const Matrix<T>& matrix) {
            std::ios_base::fmtflags f(os.flags());
            const int precision = 4;
            const int width = 8;

            os << "\n  Matrix [" << matrix.rows << "x" << matrix.cols << "]\n";
            os << "  ┌" << std::string(matrix.cols * (width + 1) + 1, ' ') << "┐\n";

            for (size_t i = 0; i < matrix.rows; i++) {
                os << "  │ ";
                for (size_t j = 0; j < matrix.cols; j++) {
                    os << std::fixed << std::setprecision(precision) << std::setw(width) << matrix(i, j);
                    if (j < matrix.cols - 1) { os << " "; }
                }
                os << " │\n";
            }

            os << "  └" << std::string(matrix.cols * (width + 1) + 1, ' ') << "┘\n";
            os.flags(f); 
            return os;
        }

        [[nodiscard]] bool hasNaN() const {
            const size_t rowJump = stride[0] / sizeof(T);
            const size_t colJump = stride[1] / sizeof(T);

            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                    // std::isnan devuelve true si el número es NaN
                    if (std::isnan(m[i * rowJump + j * colJump])) {
                        return true; 
                    }
                }
            }
            return false;
        }

        void print(int precision = 4, int width = 8) const {
            std::ios_base::fmtflags f(std::cout.flags());
            
            std::cout << "\n  Matrix [" << rows << "x" << cols << "]\n";
            std::cout << "  ┌" << std::string(cols * (width + 1) + 1, ' ') << "┐\n";

            for (size_t i = 0; i < rows; i++) {
                std::cout << "  │ ";
                for (size_t j = 0; j < cols; j++) {
                    std::cout << std::fixed << std::setprecision(precision) << std::setw(width) << (*this)(i, j);
                    
                    if (j < cols - 1) { std::cout << " "; }
                }
                std::cout << " │\n";
            }

            std::cout << "  └" << std::string(cols * (width + 1) + 1, ' ') << "┘\n\n";
            std::cout.flags(f); 
        }

        // GETTERS - SETTERS
        inline size_t             getRows()     const noexcept { return rows; }
        inline size_t             getCols()     const noexcept { return cols; }
        inline const size_t*      getStrides()  const noexcept { return stride; }
        inline const core::Arena* getArena()    const noexcept { return arena; }
        inline const T*           getValues()   const noexcept { return m; }
        inline T*                 getValues()   noexcept       { return m; }
        inline size_t             size()        const noexcept { return numElements; }
    };
}
