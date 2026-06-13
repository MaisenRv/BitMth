#pragma once

#include <iostream>
#include <cmath>

#include <BitMth/utils/Errors.hpp>
#include <BitMth/utils/Constants.hpp>

namespace BitMth::linalg{
    template <typename T>
    struct Vec2{
        T x{}, y{};

        Vec2(T x, T y): x(x), y(y){}
        Vec2() = default;

        // Copy
        Vec2(const Vec2& vector) = default;
        Vec2( Vec2&& vector) noexcept = default;
        Vec2& operator=( const Vec2& vector) = default;
        Vec2& operator=( Vec2&& vector) noexcept = default;
        ~Vec2() = default;

        // Operators Vector - number
        [[nodiscard]] Vec2<T> operator+(T number) const { return  Vec2<T>(x + number, y + number); }
        [[nodiscard]] friend Vec2<T> operator+(T number, const Vec2<T>& vector) { return  vector + number; }
        Vec2<T>& operator+=(T number) { 
            x += number;
            y += number;
            return *this; 
        }

        [[nodiscard]] Vec2<T> operator-(T number) const { return  Vec2<T>(x - number, y - number); }
        [[nodiscard]] friend Vec2<T> operator-(T number, const Vec2<T>& vector) { return  vector - number; }
        Vec2<T>& operator-=(T number) { 
            x -= number;
            y -= number;
            return *this; 
        }

        [[nodiscard]] Vec2<T> operator*(T number) const { return  Vec2<T>(x * number, y * number); }
        [[nodiscard]] friend Vec2<T> operator*(T number, const Vec2<T>& vector) { return  vector * number; }
        Vec2<T>& operator*=(T number) { 
            x *= number;
            y *= number;
            return *this; 
        }

        [[nodiscard]] Vec2<T> operator/(T number){ 
            CHECK_ERROR_VECTOR( number == T(0), "Vector division (/)", "Division for zero", "2d");
            T inv = T(1) / number;
            return  Vec2<T>(x * inv, y * inv);
        }
        Vec2<T>& operator/=(T number) { 
            CHECK_ERROR_VECTOR( number == T(0), "Vector division (/)", "Division for zero", "2d");
            T inv = T(1) / number;
            x *= inv;
            y *= inv;
            return *this; 
        }

        // Operators Vector - Vector
        [[nodiscard]] Vec2<T> operator+(const Vec2<T>& vector) const { return Vec2<T>(x + vector.x, y + vector.y); }
        Vec2<T>& operator+=(const Vec2<T>& vector)  { 
            x += vector.x;
            y += vector.y;
            return *this;
        }

        [[nodiscard]] Vec2<T> operator-(const Vec2<T>& vector) const { return Vec2<T>(x - vector.x, y - vector.y); }
        Vec2<T>& operator-=(const Vec2<T>& vector)  { 
            x -= vector.x;
            y -= vector.y;
            return *this;
        } 

        [[nodiscard]] T dot(const Vec2<T>& vector)            const { return (x * vector.x) + (y * vector.y);}
        [[nodiscard]] T cross (const Vec2<T>& vector)         const { return (x * vector.y) - (y * vector.x);}
        [[nodiscard]] Vec2<T> hadamard(const Vec2<T>& vector) const { return Vec2<T>(x * vector.x, y * vector.y); }
        Vec2<T>& hadamardInPlace(const Vec2<T>& vector){
            x *= vector.x;
            y *= vector.y;
            return *this;
        }

        // Others
        [[nodiscard]] T distanceSquared(const Vec2<T>& vector) const { 
            T dx = vector.x - x;
            T dy = vector.y - y;
            return (dx * dx) + (dy * dy);
        }
        [[nodiscard]] T distance(const Vec2<T>& vector) const {
            return std::sqrt(distanceSquared(vector));
        }

        [[nodiscard]] T magnitudeSquared() const { return (x * x) + (y * y); }
        [[nodiscard]] T magnitude() const { return std::sqrt(magnitudeSquared()); }

        Vec2<T>& normalizeInPlance(){
            T mag = magnitude();
            if (mag > utils::EPSILON<T>) {
                *this /= mag;
                return *this;
            } 
            x = T(0); y = T(0);
            return *this;
        }
        [[nodiscard]] Vec2<T> normalize() const {
            T mag = magnitude();
            if (mag > utils::EPSILON<T>) return *this / mag; 
            return Vec2<T>(T(0), T(0));
        }

        [[nodiscard]] T angleTo(const Vec2<T>& vector) const {
            T dx = vector.x - x;
            T dy = vector.y - y;
            return std::atan2(dy, dx);
        }

        [[nodiscard]] T angleBetween(const Vec2<T>& vector) const {
            return std::atan2(cross(vector), dot(vector));
        }

        [[nodiscard]] Vec2<T> midpoint(const Vec2<T>& vector) const {
            return (*this + vector) * T(0.5);
        }

        // Utils
        void clear(){ x = T(0); y = T(0); }
        void print() const { std::cout << "\tX: " << x << "\tY: " << y << std::endl; }
    };       
}