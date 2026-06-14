#pragma once

#include <vector>

#include <BitMth/linalg/Vec2.hpp>
#include <BitMth/geometry/types/LineAttachmentTypes.hpp>
#include <BitMth/utils/Constants.hpp>

namespace BitMth::geometry{
    template <typename T>
    class GeometryFactory {
    private:
        using vertexList = std::vector<linalg::Vec2<T>>;
        
        static void _calculateCornerRoundedRectangle(vertexList& list, linalg::Vec2<T> dimension, 
                                                    T delta, int cornerNumberVertex, T radius, int quadrant) noexcept{
            linalg::Vec2<T> quadrantV = {T(0), T(0)};
            T angle = T(0);
            T end = utils::HALF_PI<T>;

            if (quadrant == 1) {
                quadrantV = {T(1), T(-1)};
            } else if (quadrant == 2) {
                quadrantV = {T(-1), T(-1)};
                angle = utils::HALF_PI<T>;
                end = -utils::PI<T> * T(1);
            } else if (quadrant == 3) {
                quadrantV = {T(-1), T(1)};
                angle = -utils::PI<T>;
                end = -utils::HALF_PI<T>;
            } else {
                quadrantV = {T(1), T(1)};
                angle = -utils::HALF_PI<T>;   
                end = T(0);
            }

            T x_ = dimension.x * quadrantV.x;
            T y_ = dimension.y * quadrantV.y;

            for (int i = 0; i <= cornerNumberVertex; i++) {
                list.emplace_back(
                    x_ + (radius * std::cos(angle)),
                    y_ - (radius * std::sin(angle))
                );
                angle += delta;
            }
            
            list.emplace_back(
                x_ + (radius * std::cos(end)),
                y_ - (radius * std::sin(end))
            );
            
            if (quadrant == 4) {
                list.emplace_back(
                    x_ + (radius * std::cos(end)),
                    -y_ - (radius * std::sin(end))
                );
            }
        }

        static void _buildCircleInternal(vertexList& result ,T radius, int segments) noexcept{
            result.clear();
            result.reserve( size_t(segments) + 2);
            result.emplace_back(T(0), T(0));

            T delta = ( T(2) * utils::PI<T> ) / segments;
            for (int i = 0; i <= segments; i++) {
                T angle = i * delta;
                result.emplace_back( radius * std::cos(angle), radius * std::sin(angle) );
            }
        }

        static void _buildSquareInternal(vertexList& result,T width, T height) noexcept{
            result.clear();
            result.reserve( size_t(6));
            result.emplace_back(T(0), T(0));
            T hWidth = width / T(2);
            T hHeight = height / T(2);

            result.emplace_back(hWidth,hHeight);
            result.emplace_back(-hWidth,hHeight);
            result.emplace_back(-hWidth,-hHeight);
            result.emplace_back(hWidth,-hHeight);
            result.emplace_back(hWidth,hHeight);
        }

        static void _buildBorderSquareInternal(vertexList& result,T width, T height) noexcept{
            result.clear();
            result.reserve(size_t(5));
            result.emplace_back(T(0), T(0));
            result.emplace_back(T(0),height);
            result.emplace_back(width,height);
            result.emplace_back(width,T(0));
            result.emplace_back(T(0),T(0));
        }

        static void _buildLineInternal(vertexList& result, types::LineAttachmentTypes type) noexcept{
            result.clear();
            result.reserve(size_t(2));
            switch (type) {
            case types::LineAttachmentTypes::ATTACH_P0:
                result.emplace_back(T(0),T(0));
                result.emplace_back(T(1),T(0));
                break;
            case types::LineAttachmentTypes::ATTACH_P1:
                result.emplace_back(T(-1),T(0));
                result.emplace_back(T(0),T(0));
                break;
            case types::LineAttachmentTypes::ATTACH_CENTER:
                result.emplace_back(T(-0.5),T(0));
                result.emplace_back(T(0.5),T(0));
                break;
            default:
                result.emplace_back(T(0),T(0));
                result.emplace_back(T(1),T(0));
                break;
            }
        }

        static void _buildRoundedRectangleInternal(vertexList& result,T width, T height, T radius, int resolution) {
            result.clear();
            
            if (resolution % 4 != 0) {
                resolution += 4 - (resolution % 4);
            }
            
            int cornerNumberVertex = (resolution / 4) + 1;
            T cornerDelta = (utils::PI<T> / T(2)) / T(cornerNumberVertex);

            size_t totalVertices = (4 * (size_t(cornerNumberVertex) + 2)) + 2;
            result.reserve(totalVertices);
            result.emplace_back(T(0), T(0)); 

            T hWidth = width / T(2);
            T hHeight = height / T(2);

            for (int i = 1; i < 5; i++) {
                _calculateCornerRoundedRectangle(
                    result,
                    {hWidth, hHeight},
                    cornerDelta,
                    cornerNumberVertex,
                    radius,
                    i
                );
            }
        }
        
    public:
        GeometryFactory() = delete;
        ~GeometryFactory() = delete;
        GeometryFactory(const GeometryFactory&) = delete;
        GeometryFactory& operator=(const GeometryFactory&) = delete;
        
        // Fill shapes
        static void fillCircle(vertexList& vectorList ,T radius, int segments) noexcept{
            _buildCircleInternal(vectorList,radius,segments);
        }

        static void fillSquare(vertexList& vectorList,T width, T height) noexcept{
            _buildSquareInternal(vectorList,width,height);
        }

        static void fillBorderSquare(vertexList& vectorList,T width, T height) noexcept{
            _buildBorderSquareInternal(vectorList,width,height);
        }

        static void fillLine(vertexList& vectorList, types::LineAttachmentTypes type = types::LineAttachmentTypes::ATTACH_P0) noexcept{
            _buildLineInternal(vectorList,type);
        }

        static void fillRoundedRectangle(vertexList& vectorList,T width, T height, T radius, int resolution) noexcept {
            _buildRoundedRectangleInternal(vectorList,width,height,radius,resolution);
        }

        // Make shapes
        [[nodiscard]] static vertexList makeCircle(T radius, int segments) noexcept{
            vertexList newVector;
            _buildCircleInternal(newVector,radius,segments);
            return newVector;
        }

        [[nodiscard]] static vertexList makeSquare(T width, T height) noexcept{
            vertexList newVector;
            _buildSquareInternal(newVector,width,height);
            return newVector;
        }

        [[nodiscard]] static vertexList makeBorderSquare(T width, T height) noexcept{
            vertexList newVector;
            _buildBorderSquareInternal(newVector,width,height);
            return newVector;
        }

        [[nodiscard]] static vertexList makeLine(types::LineAttachmentTypes type = types::LineAttachmentTypes::ATTACH_P0) noexcept{
            vertexList newVector;
            _buildLineInternal(newVector,type);
            return newVector;
        }

        [[nodiscard]] static vertexList makeRoundedRectangle(T width, T height, T radius, int resolution) {
            vertexList newVector;
            _buildRoundedRectangleInternal(newVector,width,height,radius,resolution);
            return newVector;
        }

    };   
} 
