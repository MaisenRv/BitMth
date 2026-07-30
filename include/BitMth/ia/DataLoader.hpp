#pragma once

#include <vector>
#include <random>
#include <numeric>
#include <algorithm>

#include <BitMth/linalg/Matrix.hpp>

namespace BitMth::ia{
  template<typename T>
  class DataLoader {
    private:
        using Matrix = linalg::Matrix<T>;

        const Matrix& X;
        const Matrix& Y;
        size_t batchSize;
        std::vector<size_t> indices;
        size_t currentIndex = 0;

    public:
        DataLoader(const Matrix& xData, const Matrix& yData, size_t batch)
            : X(xData), Y(yData), batchSize(batch), indices(xData.getRows()) {
            std::iota(indices.begin(), indices.end(), 0);
        }

        void reset(std::mt19937& gen) {
            std::shuffle(indices.begin(), indices.end(), gen);
            currentIndex = 0;
        }

        bool hasNext() const {
            return currentIndex < X.getRows();
        }

        std::pair<Matrix, Matrix> getNextBatch() {
            size_t count = std::min(batchSize, X.getRows() - currentIndex);

            Matrix xBatch = X.getRowsByIndices(indices, currentIndex, count);
            Matrix yBatch = Y.getRowsByIndices(indices, currentIndex, count);

            currentIndex += count;
            return {xBatch, yBatch};
        }
    };
 
}
