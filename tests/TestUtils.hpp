#pragma once
#include <BitMth/linalg/Matrix.hpp>
#include <BitTest/BitTest.hpp>

namespace BitMth::tests {

  template <typename T>
  inline void assertMatrixIsDefaultInitialized(const BitMth::linalg::Matrix<T>& m) {
      BIT_ASSERT_EQ(0, m.getRows());
      BIT_ASSERT_EQ(0, m.getCols());
      BIT_ASSERT_EQ(0, m.size());
      BIT_ASSERT_EQ(0, m.getStrides()[0]);
      BIT_ASSERT_EQ(0, m.getStrides()[1]);
      BIT_ASSERT_EQ(nullptr, m.getArena());
      BIT_ASSERT_EQ(nullptr, m.getValues());
      BIT_ASSERT_EQ(nullptr, m.getAutogradNode());
      BIT_ASSERT_FALSE(m.getRequiresGrad());
      BIT_ASSERT_EQ(0, m.getRowJump());
      BIT_ASSERT_EQ(0, m.getColJump());
  }

} 
