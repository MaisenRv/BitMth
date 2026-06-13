#pragma once

#include <iostream>

#include <BitMth/utils/Errors.hpp>

namespace BitMth::signal::types{
    template <typename T>
    struct Complex {
        T re{}, im{};
        
        Complex(T re, T im): re(re), im(im){}
        Complex() = default;

        // Copy
        Complex(const Complex& complex) = default;
        Complex( Complex&& complex) noexcept = default;
        Complex& operator=( const Complex& complex) = default;
        Complex& operator=( Complex&& complex) noexcept = default;
        ~Complex() = default;

        // Operators Complex - number
        [[nodiscard]] Complex<T> operator+(T number) const { return  Complex<T>(re + number, im); }
        [[nodiscard]] friend Complex<T> operator+(T number, const Complex<T>& complex) { return  complex + number; }
        Complex<T>& operator+=(T number) { 
            re += number;
            return *this; 
        }

        [[nodiscard]] Complex<T> operator-(T number) const { return  Complex<T>(re - number, im); }
        [[nodiscard]] friend Complex<T> operator-(T number, const Complex<T>& complex) { return Complex<T>(number - complex.re, -complex.im); }
        Complex<T>& operator-=(T number) { 
            re -= number;
            return *this; 
        }

        [[nodiscard]] Complex<T> operator*(T number) const { return  Complex<T>(re * number, im * number); }
        [[nodiscard]] friend Complex<T> operator*(T number, const Complex<T>& complex) { return  complex * number; }
        Complex<T>& operator*=(T number) { 
            re *= number;
            im *= number;
            return *this; 
        }

        [[nodiscard]] Complex<T> operator/(T number) const { 
            CHECK_ERROR_COMPLEX(
                number == T(0),
                "Complex division (/)",
                "Division for zero"
            );
            T inv = T(1) / number;
            return  Complex<T>(re * inv, im * inv); 
        }
        [[nodiscard]] friend Complex<T> operator/(T number, const Complex<T>& complex) {
            T denom = complex.re * complex.re + complex.im * complex.im;
            CHECK_ERROR_COMPLEX(
                denom == T(0),
                "Complex division inverse (scalar / complex)",
                "Division by zero"
            );
            return Complex<T>((number * complex.re) / denom, (-number * complex.im) / denom);
        }
        Complex<T>& operator/=(T number) {
            CHECK_ERROR_COMPLEX(
                number == T(0),
                "Complex division (/)",
                "Division for zero"
            ); 
            T inv = T(1) / number;
            re *= inv;
            im *= inv;
            return *this; 
        }

        // Operators Complex - Complex
        [[nodiscard]] Complex<T> operator+(const Complex<T>& complex) const { return Complex<T>(re + complex.re, im + complex.im); }
        Complex<T>& operator+=(const Complex<T>& complex)  { 
            re += complex.re;
            im += complex.im;
            return *this;
        }

        [[nodiscard]] Complex<T> operator-(const Complex<T>& complex) const { return Complex<T>(re - complex.re, im - complex.im); }
        Complex<T>& operator-=(const Complex<T>& complex)  { 
            re -= complex.re;
            im -= complex.im;
            return *this;
        }

        [[nodiscard]] Complex<T> operator*(const Complex<T>& complex) const { 
            return Complex<T>(
                re * complex.re - im * complex.im, 
                re * complex.im + im * complex.re
            ); 
        }
        Complex<T>& operator*=(const Complex<T>& complex)  { 
            T r = re * complex.re - im * complex.im;
            im = re * complex.im + im * complex.re;
            re = r;
            return *this;
        }

        [[nodiscard]] Complex<T> operator/(const Complex<T>& complex) const {
            T denom = complex.re * complex.re + complex.im * complex.im;
            CHECK_ERROR_COMPLEX(
                denom == T(0),
                "Complex/Complex division (/)",
                "Division by zero (Denominator magnitude is zero)"
            );
            T invDenom = T(1) / denom;
            return Complex<T>(
                (re * complex.re + im * complex.im) * invDenom,
                (im * complex.re - re * complex.im) * invDenom
            );
        }
        Complex<T>& operator/=(const Complex<T>& complex) {
            T denom = complex.re * complex.re + complex.im * complex.im;
            CHECK_ERROR_COMPLEX(
                denom == T(0),
                "Complex/Complex division (/=)",
                "Division by zero (Denominator magnitude is zero)"
            );
            T invDenom = T(1) / denom;
            T r = (re * complex.re + im * complex.im) * invDenom;
            im = (im * complex.re - re * complex.im) * invDenom;
            re = r;
            return *this;
        }

        [[nodiscard]] bool operator==(const Complex<T>& complex) const = default;

        // Utils
        void clear(){ re = T(0); im = T(0);}
        void print() const { std::cout << "\t" << re << " " << im << "i" << std::endl; }
    };
}