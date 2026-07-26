#pragma once

#include <random>

#include <BitMth/linalg/Matrix.hpp>

namespace BitMth::random{
    inline thread_local std::mt19937 gen(std::random_device{}());

    template <typename T>
    void uniform(linalg::Matrix<T>& matrix, T min, T max){
        using DistType = std::conditional_t<
            std::is_integral_v<T>,
            std::uniform_int_distribution<T>,
            std::uniform_real_distribution<T>
        >;

        DistType dist(min, max);
        T unusedScalar = T(0);
        matrix.scalarApplyFunctionInPlace(unusedScalar, [&dist](T& element, T) {
            element = dist(gen);
        });
    }

    template <typename T>
    void normal(linalg::Matrix<T>& matrix, T mean, T stddev){
        static_assert(std::is_floating_point_v<T>,"normal only supports floating-point types (float, double, etc.).");

        std::normal_distribution<T> dist(mean, stddev);
        T unusedScalar = static_cast<T>(0);

        matrix.scalarApplyFunctionInPlace(unusedScalar, [&dist](T& element, T) {
            element = dist(gen);
        });
    }

    template <typename T>
    void xavierUniform(linalg::Matrix<T>& matrix, size_t fanIn, size_t fanOut){
        T d = sqrt( T(6.0) /(fanIn + fanOut) );
        uniform( matrix, -d ,d );
    }

    template <typename T>
    void xavierNormal(linalg::Matrix<T>& matrix, size_t fanIn, size_t fanOut){
        T stddev = sqrt( T(2.0) / (fanIn + fanOut) );
        normal( matrix, T(0.0), stddev);
    }

    template <typename T>
    void heUniform(linalg::Matrix<T>& matrix, size_t fanIn){
        T d = sqrt( T(6.0) / fanIn );
        uniform( matrix, -d ,d );
    }

    template <typename T>
    void heNormal(linalg::Matrix<T>& matrix, size_t fanIn){
        T stddev = sqrt( T(2.0) / fanIn );
        normal( matrix, T(0.0), stddev);
    }
}
