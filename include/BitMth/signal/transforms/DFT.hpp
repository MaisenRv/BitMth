#pragma once

#include <vector>

#include <BitMth/signal/types/FourierData.hpp>
#include <BitMth/utils/Constants.hpp>

namespace BitMth::signal{
    template <typename T>
    [[nodiscard]] std::vector<types::FourierData<T>> dft(const std::vector<types::Complex<T>> &data) {
        std::vector<types::FourierData<T>> fourierInfo;
        const size_t N = data.size();
        fourierInfo.reserve(N);

        for (size_t k = 0; k < N; k++) {
            types::Complex<T> sum;

            for (size_t n = 0; n < N; n++) {
                T angle = ( T(2) * utils::PI<T> * T(k * n) ) / T(N);
                types::Complex<T> result = data[n] * types::Complex<T>::fromPolar(T(1), -angle);
                sum += result;
            }
            sum /= T(N);
            T frecuency = k;
            T amplitude = sum.magnitude();
            T phase = sum.phase();

            types::FourierData<T> fourierElement{sum, frecuency, amplitude, phase};
            fourierInfo.emplace_back(fourierElement);
        }
        return fourierInfo;
    }
}