#pragma once

namespace BitMth::utils{
    template <typename T>
    inline constexpr T EPSILON = T(1e-6);

    template <typename T>
    inline constexpr T GRAVITY = T(9.80665);

    template <typename T>
    inline constexpr T PI = T(3.14159265358979323846);
    template <typename T>
    inline constexpr T TWO_PI = PI<T> * static_cast<T>(2);
    template <typename T>
    inline constexpr T HALF_PI = PI<T> * static_cast<T>(0.5);

    
    template <typename T>
    inline constexpr T MOMENTUM_FACTOR = static_cast<T>(0.9);
    template <typename T>
    inline constexpr T MOMENTUM_FACTOR2 = static_cast<T>(0.999);
    template <typename T>
    inline constexpr T WEIGHT_DECAY = static_cast<T>(0.01);
}
