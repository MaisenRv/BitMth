#pragma once

#include <BitMth/signal/types/Complex.hpp>

namespace BitMth::signal::types{
    template <typename T>
    struct FourierData{
        Complex<T> numbers;
        T frecuency;
        T amplitude;
        T phase;
    };
}