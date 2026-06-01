#pragma once

#include <random>
#include <cmath>

#include <BitMth/core/Matrix.hpp>

namespace Bitmth{
    namespace Random{
        inline std::random_device rd;
        inline std::mt19937 gen(rd());

        template <typename T>
        class MatrixRandom {            
            public:

                static void uniform(Matrix<T>& matrix, T min, T max){
                    std::uniform_real_distribution<T> dist(min, max);
                    for (size_t i = 0; i < matrix.numElements; i++) matrix.m[i] = dist(gen);
                }

                static void normal(Matrix<T>& matrix, T mean, T stddev){
                    std::normal_distribution<T> dist(mean, stddev);
                    for (size_t i = 0; i < matrix.numElements; i++) matrix.m[i] = dist(gen);
                }

                static void xavierUniform(Matrix<T>& matrix, size_t fanIn, size_t fanOut){
                    T d = sqrt( T(6.0) /(fanIn + fanOut) );
                    uniform( matrix, -d ,d );
                }

                static void xavierNormal(Matrix<T>& matrix, size_t fanIn, size_t fanOut){
                    T stddev = sqrt( T(2.0) / (fanIn + fanOut) );
                    normal( matrix, T(0.0), stddev);
                }

                static void heUniform(Matrix<T>& matrix, size_t fanIn){
                    T d = sqrt( T(6.0) / fanIn );
                    uniform( matrix, -d ,d );
                }

                static void heNormal(Matrix<T>& matrix, size_t fanIn){
                    T stddev = sqrt( T(2.0) / fanIn );
                    normal( matrix, T(0.0), stddev);
                }
        };
    }
}