#pragma once

namespace BitMth{
    namespace Utils{
        template <typename T>
        inline constexpr T EPSILON = T(1e-6);

        template <typename T>
        inline constexpr T GRAVITY = T(9.80665);

        template <typename T>
        inline constexpr T PI = T(3.14159265358979323846);
    } 
}