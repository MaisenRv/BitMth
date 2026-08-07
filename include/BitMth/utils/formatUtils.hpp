#pragma once

#include <cstddef>
#include <ostream>
#include <iomanip>

#include <BitMth/linalg/Matrix.hpp>

namespace BitMth::utils {
  template<typename T>
  std::ostream& show(std::ostream& os, const BitMth::linalg::Matrix<T>& matrix,int precision, int width){
    std::ios_base::fmtflags f(os.flags());
    size_t rows = matrix.getRows();
    size_t cols = matrix.getCols();
    
    os << "\n Matrix [" << rows << "x" << cols << "]\n";
    os << " ┌" << std::string(cols * (width + 1) + 1, ' ') << "┐\n";

    for (size_t i = 0; i < rows; i++) {
        os << " │ ";
        for (size_t j = 0; j < cols; j++) {
            os << std::fixed << std::setprecision(precision) << std::setw(width) << matrix(i, j);
            if (j < cols - 1) { os << " "; }
        }
        os << " │\n";
    }

    os << " └" << std::string(cols * (width + 1) + 1, ' ') << "┘\n";
    os.flags(f); 
    return os;
  }
}
