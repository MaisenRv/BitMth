#pragma once

#include <iostream>
#include <cmath>

#include <BitMth/utils/Errors.hpp>

namespace BitMth{
    namespace linalg{
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

            // Operators Vector - number
            Vec2<T> operator+(T number) const { return  Vec2<T>(x + number, y + number); }
            friend Vec2<T> operator+(T number, const Vec2<T>& vector) { return  vector + number; }
            Vec2<T>& operator+=(T number) { 
                x += number;
                y += number;
                return *this; 
            }

            Vec2<T> operator-(T number) const { return  Vec2<T>(x - number, y - number); }
            friend Vec2<T> operator-(T number, const Vec2<T>& vector) { return  vector - number; }
            Vec2<T>& operator-=(T number) { 
                x -= number;
                y -= number;
                return *this; 
            }

            Vec2<T> operator*(T number) const { return  Vec2<T>(x * number, y * number); }
            friend Vec2<T> operator*(T number, const Vec2<T>& vector) { return  vector * number; }
            Vec2<T>& operator*=(T number) { 
                x *= number;
                y *= number;
                return *this; 
            }

            Vec2<T> operator/(T number){ 
                CHECK_ERROR_VECTOR( number == 0, "Vector division (/)", "Division for zero", "2d");
                T inv = T(1) / number;
                return  Vec2<T>(x * inv, y * inv);
            }
            Vec2<T>& operator/=(T number) { 
                CHECK_ERROR_VECTOR( number == 0, "Vector division (/)", "Division for zero", "2d");
                T inv = T(1) / number;
                x *= inv;
                y *= inv;
                return *this; 
            }

            // Operators Vector - Vector
            Vec2<T> operator+(const Vec2<T>& vector) const { return Vec2<T>(x + vector.x, y + vector.y); }
            Vec2<T>& operator+=(const Vec2<T>& vector)  { 
                x += vector.x;
                y += vector.y;
                return *this;
            }

            Vec2<T> operator-(const Vec2<T>& vector) const { return Vec2<T>(x - vector.x, y - vector.y); }
            Vec2<T>& operator-=(const Vec2<T>& vector)  { 
                x -= vector.x;
                y -= vector.y;
                return *this;
            } 

            T dot(const Vec2<T>& vector)            const { return (x * vector.x) + (y * vector.y);}
            T cross (const Vec2<T>& vector)         const { return (x * vector.y) - (y * vector.x);}
            Vec2<T> hadamard(const Vec2<T>& vector) const { return Vec2<T>(x * vector.x, y * vector.y); }
            Vec2<T>& hadamardInPlace(const Vec2<T>& vector){
                x *= vector.x;
                y *= vector.y;
                return *this;
            }

            // Others
            T distanceSquared(const Vec2<T>& vector) const { 
                T dx = vector.x - x;
                T dy = vector.y - y;
                return (dx * dx) + (dy * dy);
            }
            T distance(const Vec2<T>& vector) const {
                return std::sqrt(distanceSquared(vector));
            }

            T magnitudeSquared() const { return (x * x) + (y * y); }
            T magnitude() const { return std::sqrt(magnitudeSquared()); }

            Vec2<T>& normalizeInPlance(){
                T mag = magnitude();
                if (mag > T(0)) {
                    *this /= mag;
                    return *this;
                } 
                x = T(0); y = T(0);
                return *this;
            }
            Vec2<T> normalize() const {
                T mag = magnitude();
                if (mag > T(0)) return *this / mag; 
                return Vec2<T>(T(0), T(0));
            }

            // Utils
            void clear(){ x = T(0); y = T(0); }
            void print() const { std::cout << "\tX: " << x << "\tY: " << y << std::endl; }
        };       
    }
}