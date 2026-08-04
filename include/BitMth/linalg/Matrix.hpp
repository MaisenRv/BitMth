#pragma once

#include <climits>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <utility>
#include <cstring>
#include <iomanip>
#include <vector>

#include <BitMth/utils/Errors.hpp>
#include <BitMth/utils/Constants.hpp>
#include <BitMth/core/Arena.hpp>
#include <BitMth/core/ParallelExecutor.hpp>
#include <BitMth/ia/autograd/Node.hpp>

namespace BitMth::linalg{
    template <typename T>
    class [[nodiscard]] Matrix{
    private:
        Matrix(size_t rows, size_t cols, const size_t *customStrides, core::Arena* arena): Matrix(rows, cols, arena, false){
            stride[0] = customStrides[0];
            stride[1] = customStrides[1];
            _updateStrideJumps();
        }

        size_t rows{0}, cols{0}, numElements{0};
        size_t stride[2]{};
        T *m{nullptr};
        core::Arena *arena{nullptr};
        ia::Node<Matrix<T>>* autogradNode{nullptr};
        bool requiresGrad{false};

        size_t rowJumpThis;
        size_t colJumpThis;
        void _updateStrideJumps(){
            rowJumpThis = stride[0] / sizeof(T);
            colJumpThis = stride[1] / sizeof(T);
        }
    public:
        template< typename Op>
        static Matrix<T> scalarApplyFunction(const Matrix<T> &matrix,T scalar, core::Arena *arenaContainer, Op funct){
            Matrix<T> result(matrix.rows, matrix.cols, arenaContainer, false);

            if (matrix.numElements < core::config::PARALLEL_THRESHOLD_SIMPLE) {
                for (size_t i = 0; i < matrix.numElements; i++) result.m[i] = funct(matrix.m[i], scalar);
                return result;
            }

            core::getParallelExecutor().execute(matrix, scalar, [&result,funct](Matrix<T>& mat, T sc, unsigned int start, unsigned int end){
                for (size_t i = start; i < end; i++) result.m[i] = funct(mat.m[i], sc);
            });
            return result;
        }

        template<typename Op>
        Matrix<T>& scalarApplyFunctionInPlace(T scalar, Op funct) {
            if(numElements < core::config::PARALLEL_THRESHOLD_SIMPLE){
                for (size_t i = 0; i < numElements; i++) funct(this->m[i], scalar);
                return *this;
            }

            core::getParallelExecutor().execute(*this, scalar, [this, funct](Matrix<T>&, T sc, unsigned int start, unsigned int end){
                for (size_t i = start; i < end; i++) funct(this->m[i], sc);
            });
            return *this; 
        }

        template< typename Op>
        static Matrix<T> matrixApplyFunction(const Matrix<T> &matrixA, const Matrix<T> &matrixB, core::Arena *arenaContainer, Op funct){
            Matrix<T> result(matrixA.rows, matrixA.cols, arenaContainer, false);
            const size_t aJumpRow = matrixA.getRowJump();
            const size_t aJumpCol = matrixA.getColJump();
            const size_t bJumpRow = matrixB.getRowJump();
            const size_t bJumpCol = matrixB.getColJump();

            if (matrixA.size() < core::config::PARALLEL_THRESHOLD_COMPLEX) {
                for (size_t i = 0; i < matrixA.rows; i++) {
                    for (size_t j = 0; j < matrixA.cols; j++) {
                        T valA = matrixA.m[i * aJumpRow + j * aJumpCol];
                        T valB = matrixB.m[i * bJumpRow+ j * bJumpCol];
                        result.m[i * result.cols + j] = funct(valA, valB);
                    }
                }
                return result;
            }
            core::getParallelExecutor().execute(matrixA, matrixB,
                [&result,funct,aJumpRow, aJumpCol, bJumpRow, bJumpCol](Matrix<T>& matA, Matrix<T>& matB, unsigned int startR, unsigned int endR, unsigned int startC, unsigned int endC){
                    for (size_t i = startR; i < endR; i++) {
                        for (size_t j = startC; j < endC; j++) {
                            T valA = matA.m[i * aJumpRow + j * aJumpCol];
                            T valB = matB.m[i * bJumpRow+ j * bJumpCol];
                            result.m[i * result.cols + j] = funct(valA, valB);
                        }
                    }
            });
            return result;
        }

        template< typename Op>
        Matrix<T>& matrixApplyFunctionInPlace(const Matrix<T>& matrix,Op funct) {
            const size_t rowJumpThis = getRowJump();
            const size_t colJumpThis = getColJump();
            const size_t rowJumpOther = matrix.getRowJump();
            const size_t colJumpOther = matrix.getColJump();

            if (size() < core::config::PARALLEL_THRESHOLD_COMPLEX) {
                for (size_t i = 0; i < rows; i++) {
                    T* const rowThis = &m[i * rowJumpThis];
                    const T* const rowOther = &matrix.m[i * rowJumpOther];
                    for (size_t j = 0; j < cols; j++) {
                        funct(rowThis[j * colJumpThis], rowOther[j * colJumpOther]);
                    }
                }
                return *this;
            }
            core::getParallelExecutor().execute(*this, const_cast<linalg::Matrix<T>&>(matrix),
                [funct,rowJumpThis ,colJumpThis , rowJumpOther , colJumpOther ](Matrix<T>& matA, Matrix<T>& matB, unsigned int startR, unsigned int endR, unsigned int startC, unsigned int endC){
                    for (size_t i = startR; i < endR; i++) {
                        for (size_t j = startC; j < endC; j++) {
                            T& valA = matA.m[i * rowJumpThis + j * colJumpThis ];
                            T valB = matB.m[i * rowJumpOther + j * colJumpOther ];
                            funct(valA, valB);
                        }
                    }
            });
            return *this;
        }

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
            _updateStrideJumps();
            autogradNode->conteiner = this;
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
                _updateStrideJumps();
                arena = nullptr;
                autogradNode = nullptr;
                requiresGrad = inMatrix.requiresGrad;
                std::memcpy(m,inMatrix.m,numElements * sizeof(T));
            }

        Matrix<T> clone(core::Arena *arenaContainer) const {
            Matrix<T> copy(rows,cols,stride,arenaContainer);
            std::memcpy(copy.m, m, numElements * sizeof(T));
            return copy;
        }
        
        Matrix( Matrix&& inMatrix) noexcept :
            rows(inMatrix.rows), cols(inMatrix.cols),numElements(inMatrix.numElements), m(inMatrix.m), arena(inMatrix.arena){
                stride[0] = inMatrix.stride[0];
                stride[1] = inMatrix.stride[1];
                autogradNode = inMatrix.autogradNode;
                if(autogradNode != nullptr){
                    autogradNode->container = this; 
                }
                _updateStrideJumps();
                inMatrix.stride[0] = 0;
                inMatrix.stride[1] = 0;
                inMatrix.rowJumpThis = 0;
                inMatrix.colJumpThis = 0;
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
            _updateStrideJumps();
            autogradNode = nullptr;
            std::memcpy(m, inMatrix.m, numElements * sizeof(T));
            return *this;
        }

        Matrix& operator=(Matrix&& inMatrix) noexcept {
            if (this == &inMatrix) return *this;

            if (arena == nullptr) delete[] m;

            arena = inMatrix.arena;
            rows = inMatrix.rows;
            cols = inMatrix.cols;
            autogradNode = inMatrix.autogradNode;
            if(autogradNode != nullptr){
                autogradNode->container = this; 
            }
            numElements = inMatrix.numElements;
            m = inMatrix.m;
            stride[0] = inMatrix.stride[0];
            stride[1] = inMatrix.stride[1];
            _updateStrideJumps();

            inMatrix.rows = 0;
            inMatrix.cols = 0;
            inMatrix.numElements = 0;
            inMatrix.m = nullptr;
            inMatrix.arena = nullptr;
            inMatrix.stride[0] = 0;
            inMatrix.stride[1] = 0;
            inMatrix.rowJumpThis = 0;
            inMatrix.colJumpThis = 0;
            inMatrix.autogradNode = nullptr;

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
            return scalarApplyFunctionInPlace(scalar, [](T& element, T sca) { element += sca; });
        }
        Matrix<T> operator+(T scalar) const {
            return scalarApplyFunction(*this, scalar, nullptr, [](T element, T sca){ return element + sca; });
        }
        friend Matrix<T> operator+(T scalar, const Matrix<T>& matrix) {
            return matrix + scalar;
        }
        static Matrix<T> add(const Matrix<T>& matrix, T scalar, core::Arena* targetArena) {
            return scalarApplyFunction(matrix, scalar, targetArena, [](T element, T sca){ return element + sca; });
        }
        static Matrix<T> add(T scalar, const Matrix<T>& matrix, core::Arena* targetArena) {
            return add(matrix, scalar, targetArena);
        }
        // ------------------------------------------

        // ----------- SUBTRACTION OPERATOR -----------
        Matrix& operator-=(T scalar) {
            return scalarApplyFunctionInPlace(scalar, [](T& element, T sca) { element -= sca; });
        }
        Matrix<T> operator-(T scalar) const {
            return scalarApplyFunction(*this, scalar, nullptr, [](T element, T sca){ return element - sca; });
        }
        friend Matrix<T> operator-(T scalar, const Matrix<T>& matrix) {
            return scalarApplyFunction(matrix, scalar, nullptr, [](T element, T sca){ return sca - element; });
        }
        static Matrix<T> sub(const Matrix<T>& matrix, T scalar, core::Arena* targetArena) {
            return scalarApplyFunction(matrix, scalar, targetArena, [](T element, T sca){ return element - sca; });
        }
        static Matrix<T> sub(T scalar,const Matrix<T>& matrix, core::Arena* targetArena) {
            return scalarApplyFunction(matrix, scalar, targetArena, [](T element, T sca){ return sca - element; });
        }
        // ------------------------------------------

        // ----------- MULTIPLICATION OPERATOR -----------
        Matrix& operator*=(T scalar) {
            return scalarApplyFunctionInPlace(scalar, [](T& element, T sca) { element *= sca; });           
        }
        Matrix<T> operator*(T scalar) const {
            return scalarApplyFunction(*this, scalar, nullptr, [](T element, T sca){ return element * sca; });
        }
        friend Matrix<T> operator*(T scalar, const Matrix<T>& matrix) {
            return matrix * scalar;
        }
        static Matrix<T> mul(const Matrix<T>& matrix, T scalar, core::Arena* targetArena) {
            return scalarApplyFunction(matrix, scalar, targetArena, [](T element, T sca){ return element * sca; });
        }
        static Matrix<T> mul(T scalar, const Matrix<T>& matrix, core::Arena* targetArena) {
            return mul(matrix, scalar, targetArena);
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
        Matrix<T> operator/(T scalar) const {
            CHECK_ERROR_MATRIX(
                scalar < utils::EPSILON<T> && scalar > -utils::EPSILON<T>,
                "Matrix operator/ scalar",
                "Division by zero"
            );
            T aux = T(1) / scalar;
            return *this * aux;
        }
        static Matrix<T> div(const Matrix<T>& matrix, T scalar, core::Arena* targetArena) {
            CHECK_ERROR_MATRIX(
                scalar < utils::EPSILON<T> && scalar > -utils::EPSILON<T>,
                "Matrix operator/ scalar",
                "Division by zero"
            );
            T aux = T(1) / scalar;
            return mul(matrix,aux,targetArena);
        }
        // ------------------------------------------

        // ----------- POW OPERATION -----------
        Matrix<T>& powInPlace(T exponent) {
            if(exponent < utils::EPSILON<T> && exponent > -utils::EPSILON<T>){ 
                setOne();
                return *this;
            }
            if (exponent == T(2)) {
                return scalarApplyFunctionInPlace(exponent, [](T& element, T exp) { element *= element; });
            } 
            return scalarApplyFunctionInPlace(exponent, [](T& element, T exp) { element = std::pow(element, exp); });
        }
        Matrix<T> pow(T exponent, core::Arena* targetArena = nullptr) const {
            if(exponent < utils::EPSILON<T> && exponent > -utils::EPSILON<T>){
                Matrix<T> result(rows, cols, stride, targetArena);
                result.setOne();
                return result;
            }
            if (exponent == T(2)) {
                return scalarApplyFunction(*this, exponent, targetArena, [](T element, T exp){ return element * element; });
            }
            return scalarApplyFunction(*this, exponent, targetArena, [](T element, T exp){ return std::pow(element, exp); });
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
            return matrixApplyFunctionInPlace(matrix, [](T& valThis, T valOther) { valThis += valOther; });
        }
        Matrix<T> operator+(const Matrix<T>& matrix) const {
            CHECK_ERROR_MATRIX(
                rows != matrix.rows || cols != matrix.cols,
                "matrix add (+)",
                "Matrix dimensions must match (rows = rows && cols == cols)"
            );
            return matrixApplyFunction(*this, matrix,nullptr, [](T mA, T mB){ return mA + mB; });
        }
        static Matrix<T> add(const Matrix<T>& matrixA, const Matrix<T>& matrixB, core::Arena* targetArena){
            CHECK_ERROR_MATRIX(
                matrixA.rows != matrixB.rows || matrixA.cols != matrixB.cols,
                "Matrix::add",
                "Matrix dimensions must match (rows = rows && cols == cols)"
            );
            return matrixApplyFunction(matrixA, matrixB, targetArena, [](T mA, T mB){ return mA + mB; });
        }
        // ------------------------------------------

         // ----------- SUBTRACTION OPERATOR -----------
        Matrix& operator-=(const Matrix<T>& matrix) {
            CHECK_ERROR_MATRIX(
                rows != matrix.rows || cols != matrix.cols,
                "matrix sub (-=)",
                "Matrix dimensions must match (rows == rows && cols == cols)"
            );
            return matrixApplyFunctionInPlace(matrix, [](T& valThis, T valOther) { valThis -= valOther; });
        }
        Matrix<T> operator-(const Matrix<T>& matrix) const {
            CHECK_ERROR_MATRIX(
                rows != matrix.rows || cols != matrix.cols,
                "matrix sub (-)",
                "Matrix dimensions must match (rows == rows && cols == cols)"
            );
            return matrixApplyFunction(*this, matrix, nullptr, [](T mA, T mB){ return mA - mB; });
        }
        static Matrix<T> sub(const Matrix<T>& matrixA, const Matrix<T>& matrixB, core::Arena* targetArena){
            CHECK_ERROR_MATRIX(
                matrixA.rows != matrixB.rows || matrixA.cols != matrixB.cols,
                "Matrix::sub",
                "Matrix dimensions must match (rows == rows && cols == cols)"
            );
            return matrixApplyFunction(matrixA, matrixB, targetArena, [](T mA, T mB){ return mA - mB; });
        }
        // ------------------------------------------

        // ----------- MULTIPLICATION OPERATOR -----------
        Matrix<T>& operator*=(const Matrix<T>& matrix) {
            *this = *this * matrix;
            return *this;
        }
        Matrix<T> operator*(const Matrix<T>& matrix) const {
            CHECK_ERROR_MATRIX(
                cols != matrix.rows,
                "matrix multiplication (*)",
                "Matrix dimensions must match rows = cols"
            );

            Matrix<T> result(rows, matrix.cols, nullptr, true);
            const size_t lhsRowStride = getRowJump();
            const size_t lhsColStride = getColJump();
            const size_t rhsRowStride = matrix.getRowJump();
            const size_t rhsColStride = matrix.getColJump();

            const size_t totalOps = rows * matrix.cols * cols;
            if (totalOps < core::config::PARALLEL_THRESHOLD_COMPLEX) {
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

            core::getParallelExecutor().execute(*this, matrix,
                [&result, lhsRowStride, lhsColStride, rhsRowStride, rhsColStride, this]
                (const Matrix<T>& matA, const Matrix<T>& matB, unsigned int startR, unsigned int endR, unsigned int startC, unsigned int endC) {
                    for (size_t i = startR; i < endR; i++) {
                        T* const resRow = &result.m[i * result.cols];
                
                        for (size_t k = 0; k < matA.cols; k++) {
                            const T factor = matA.m[i * lhsRowStride + k * lhsColStride];
                            const T* const rhsRowPtr = &matB.m[k * rhsRowStride];

                            for (size_t j = startC; j < endC; j++) {
                                resRow[j] += factor * rhsRowPtr[j * rhsColStride];
                            }
                        }
                    }
                }
            );

            return result;
        }
        static Matrix<T> mul(const Matrix<T>& matrixA, const Matrix<T>& matrixB, core::Arena* targetArena) {
            CHECK_ERROR_MATRIX(
                matrixA.cols != matrixB.rows,
                "matrix::mul",
                "Matrix dimensions must match rows = cols"
            );

            Matrix<T> result(matrixA.rows, matrixB.cols, targetArena, true);
            const size_t aRowStride = matrixA.getRowJump();
            const size_t aColStride = matrixA.getColJump();
            const size_t bRowStride = matrixB.getRowJump();
            const size_t bColStride = matrixB.getColJump();

            const size_t totalOps = matrixA.rows * matrixB.cols * matrixA.cols;

            if (totalOps < core::config::PARALLEL_THRESHOLD_COMPLEX) {
                for (size_t i = 0; i < matrixA.rows; ++i) {
                    T* const resRow = &result.m[i * result.cols];

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

            core::getParallelExecutor().execute(matrixA, matrixB,
                [&result, aRowStride, aColStride, bRowStride, bColStride]
                (const Matrix<T>& matA, const Matrix<T>& matB, unsigned int startR, unsigned int endR, unsigned int startC, unsigned int endC) {
                    
                    for (size_t i = startR; i < endR; ++i) {
                        T* const resRow = &result.m[i * result.cols];

                        for (size_t k = 0; k < matA.cols; ++k) {
                            const T factor = matA.m[i * aRowStride + k * aColStride];
                            const T* const bRowPtr = &matB.m[k * bRowStride];

                            for (size_t j = startC; j < endC; ++j) {
                                resRow[j] += factor * bRowPtr[j * bColStride];
                            }
                        }
                    }
                }
            );

            return result;
        }
        Matrix& hadamardInPlace(const Matrix<T>& matrix){
            CHECK_ERROR_MATRIX(
                rows != matrix.rows || cols != matrix.cols,
                "matrix hadamard product ( (·) )",
                "Matrix dimensions must match (rows = rows && cols == cols)"
            );
            return matrixApplyFunctionInPlace(matrix, [](T& valThis, T valOther) { valThis *= valOther; });
        }
        static Matrix<T> hadamard(const Matrix<T>& matrixA, const Matrix<T>& matrixB, core::Arena* targetArena){
            CHECK_ERROR_MATRIX(
                matrixA.rows != matrixB.rows || matrixA.cols != matrixB.cols,
                "matrix hadamard product ( (·) )",
                "Matrix dimensions must match (rows = rows && cols == cols)"
            );
            return matrixApplyFunction(matrixA, matrixB, targetArena, [](T mA, T mB){ return mA * mB; });
        }
        // ------------------------------------------

        Matrix& tInPlace(){
            std::swap(rows,cols);
            std::swap(stride[0],stride[1]);
            _updateStrideJumps();
            return *this;
        } 

        static Matrix<T> t(const Matrix<T>& matrix, core::Arena* targetArena = nullptr) {
            Matrix<T> result = matrix.clone(targetArena);
            result.tInPlace();
            return result;
        }

        Matrix<T> operator/(const Matrix<T>& matrix) const {
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
            return matrixApplyFunction(*this, matrix, nullptr, [](T mA, T mB){ 
                return mA / mB; 
            });
        }

        bool operator==(const Matrix<T>& matrix) const {
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

        bool isApprox(const Matrix<T>& matrix, T margin = utils::EPSILON<T>) const {
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

        Matrix<T> reduceSumCols(core::Arena* targetArena = nullptr) const{
            Matrix<T> result(rows, 1, targetArena, true);

            for (size_t i = 0; i < rows; i++){
                const T* const currentRow = &m[i * rowJumpThis];

                for (size_t j = 0; j < cols; j++){
                    result.m[i] += currentRow[j];  
                }
            }
            return result;
        }
        Matrix<T> reduceSumRows(core::Arena* targetArena = nullptr) const{
            Matrix<T> result(1, cols, targetArena, true);

            for (size_t i = 0; i < rows; i++){
                const T* const currentRow = &m[i * rowJumpThis];
                for (size_t j = 0; j < cols; j++){
                    result.m[j] += currentRow[j];
                }
            }
            return result;
        }
        T reduceSumTotal() const {
            T result = 0;
            for (size_t i = 0; i < numElements; i++) result += m[i];
            return result;
        }

        Matrix<T> addRowVector(const Matrix<T>& rowVector, core::Arena* targetArena = nullptr) const {
            CHECK_ERROR_MATRIX(
                rowVector.rows != 1 || rowVector.cols != cols,
                "Matrix operator (addRowVector)",
                "Dimensions mismatch"
            );
            Matrix<T> result(rows, cols, stride, targetArena);
            const size_t resRowJump = result.getRowJump();

            for (size_t i = 0; i < rows; i++) {
                const T* const currentRow = &m[i * rowJumpThis];
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
            for(size_t i = 0; i < rows; i++){
                T* const currentRow = &m[i * rowJumpThis];
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

        bool hasNaN() const {
            for (size_t i = 0; i < numElements; i++) {
                if (std::isnan(m[i])) return true;
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

        Matrix<T> getRowsByIndices(const std::vector<size_t>& indices, size_t offset, size_t count) const {
            size_t actualCount = std::min(count, indices.size() - offset);
            Matrix batch(actualCount, cols, nullptr, false);

            T* dest = batch.getValues();
            const size_t bytesToCopy = stride[0];

            for (size_t i = 0; i < actualCount; ++i) {
                size_t originalRow = indices[offset + i];
                const T* const rowThis = &m[originalRow * rowJumpThis];

                std::memcpy(
                    dest + (i * cols), 
                    rowThis, 
                    bytesToCopy
                );
            }
            return batch;
        }

        // GETTERS - SETTERS
        inline size_t               getRows()     const noexcept { return rows; }
        inline size_t               getCols()     const noexcept { return cols; }
        inline const size_t*        getStrides()  const noexcept { return stride; }
        inline const core::Arena*   getArena()    const noexcept { return arena; }
        inline const T*             getValues()   const noexcept { return m; }
        inline T*                   getValues()   noexcept       { return m; }
        inline size_t               size()        const noexcept { return numElements; }
        inline const size_t         getRowJump()  const noexcept { return rowJumpThis; }
        inline const size_t         getColJump()  const noexcept { return colJumpThis; }

        inline ia::Node<Matrix<T>>* getAutogradNode() const noexcept           { return autogradNode; }
        inline void                 setAutogradNode(ia::Node<Matrix<T>>* node) { autogradNode = node; }  
        inline void                 setRequiresGrad(bool req) noexcept         { requiresGrad = req; }
        inline bool                 getRequiresGrad() const noexcept           {return (requiresGrad || (autogradNode != nullptr && autogradNode->requiresGrad)); }

    };
}
