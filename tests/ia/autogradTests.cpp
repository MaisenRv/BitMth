#include <BitTest/BitTest.hpp>
#include "../TestUtils.hpp"

BIT_GROUP_BEGIN(autograd)

BIT_TEST_CASE(NodeInitializationDefault) {
  using Matrix = BitMth::linalg::Matrix<float>;
  using Node = BitMth::ia::Node<Matrix>;
  Node node;
  BIT_ASSERT_EQ(nullptr, node.container);
  BIT_ASSERT_FALSE(bool(node.backward_fn));
  BIT_ASSERT_FALSE(node.requiresGrad);
  BIT_ASSERT_EQ(
    size_t(BitMth::ia::types::OpType::NONE),
    size_t(node.operation));
  BIT_ASSERT_EQ(0, node.parents.size());

  BitMth::tests::assertMatrixIsDefaultInitialized<float>(node.grad);
}

BIT_GROUP_END()
