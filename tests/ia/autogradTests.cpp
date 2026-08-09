#include "checks/Assert.hpp"
#include "ia/autograd/types/OpTypes.hpp"
#include <BitTest/BitTest.hpp>
#include <BitMth/linalg/Matrix.hpp>
#include <BitMth/ia/autograd/Node.hpp>
#include <BitMth/ia/autograd/ComputationGraph.hpp>
#include <BitMth/ia/autograd/Ops.hpp>
#include <type_traits>

using Matrix = BitMth::linalg::Matrix<double>;
using Arena = BitMth::core::Arena;
using Node = BitMth::ia::Node<Matrix>;
using Graph = BitMth::ia::ComputationGraph<Matrix>;
using RecordGraph = BitMth::ia::RecordGraphGuard<Matrix>;

BIT_GROUP_BEGIN(autograd)

BIT_TEST_CASE(NodeInitializationDefault) {
  Node node;
  BIT_ASSERT_EQ(nullptr, node.grad);
  BIT_ASSERT_EQ(nullptr, node.values);
  BIT_ASSERT_EQ(0, node.parents.size());
  BIT_ASSERT_FALSE(node.requiresGrad);
  BIT_ASSERT_EQ(
    size_t(BitMth::ia::types::OpType::NONE),
    size_t(node.operation));
  BIT_ASSERT_FALSE(bool(node.backward_fn));
}

BIT_TEST_CASE(ComputationGraphInitializationDefault) {
  BIT_ASSERT_EQ(nullptr, Graph::getComputationGraph());
  BIT_ASSERT_FALSE(Graph::isCreateGraph());
  Graph graph;
  BIT_ASSERT_EQ(0, graph.getNodes().size());
  BIT_ASSERT_FALSE(std::is_copy_constructible_v<Graph>);
  BIT_ASSERT_FALSE(std::is_copy_assignable_v<Graph>);
}

BIT_TEST_CASE(ComputationGraphNodeManager) {
  Graph graph;
  Node* node1 = graph.createNode();
  BIT_ASSERT_EQ(1, graph.getNodes().size());
  graph.clear();
  BIT_ASSERT_EQ(0, graph.getNodes().size());
}

BIT_TEST_CASE(RecordGraphGuard) {
  Graph graph;
  BIT_ASSERT_EQ(nullptr, Graph::getComputationGraph());
  RecordGraph rGraph(graph);
  BIT_ASSERT_EQ(&graph, Graph::getComputationGraph());
  BIT_ASSERT_FALSE(std::is_copy_constructible_v<RecordGraph>);
  BIT_ASSERT_FALSE(std::is_copy_assignable_v<RecordGraph>);
}

BIT_TEST_CASE(OpsAdd) {
  Graph graph;
  RecordGraph rGraph(graph);
  Arena arena(Arena::MB(2));

  Node* node1 = graph.createNode();
  node1->values = new Matrix(3, 3, &arena);
  node1->values->setWith(1.0);
  node1->grad = new Matrix(3, 3, &arena);
  node1->requiresGrad = true;

  Node* node2 = graph.createNode();
  node2->values = new Matrix(3, 3, &arena);
  node2->values->setWith(2.0);
  node2->grad = new Matrix(3, 3, &arena);
  node2->requiresGrad = true;

  Node* node3 = graph.createNode();
  node3->values = new Matrix(3, 3, &arena);
  node3->values->setWith(3.0);
  node3->grad = new Matrix(3, 3, &arena);
  node3->requiresGrad = true;

  Node* nodeR = BitMth::ia::Ops<double>::add({node1, node2, node3}, &arena);

  Matrix resultForward(3, 3, &arena);
  resultForward.setWith(6.0);

  BIT_ASSERT_TRUE(resultForward.isApprox(*(nodeR->values)));
  BIT_ASSERT_EQ(size_t(3), nodeR->parents.size());
  BIT_ASSERT_EQ(node1, nodeR->parents[0]);
  BIT_ASSERT_EQ(node2, nodeR->parents[1]);
  BIT_ASSERT_EQ(node3, nodeR->parents[2]);
  BIT_ASSERT_TRUE(nodeR->requiresGrad);
  BIT_ASSERT_EQ(
    size_t(BitMth::ia::types::OpType::ADD),
    size_t(nodeR->operation)
  );

  nodeR->grad->setWith(1.5);
  nodeR->backward_fn(nodeR);

  Matrix expectedGrad(3, 3, &arena);
  expectedGrad.setWith(1.5);

  BIT_ASSERT_TRUE(expectedGrad.isApprox(*(node1->grad)));
  BIT_ASSERT_TRUE(expectedGrad.isApprox(*(node2->grad)));
  BIT_ASSERT_TRUE(expectedGrad.isApprox(*(node3->grad)));

  nodeR->grad->setWith(0.5);
  nodeR->backward_fn(nodeR);

  Matrix expectedAccumGrad(3, 3, &arena);
  expectedAccumGrad.setWith(2.0);

  BIT_ASSERT_TRUE(expectedAccumGrad.isApprox(*(node1->grad)));
  BIT_ASSERT_TRUE(expectedAccumGrad.isApprox(*(node2->grad)));
  BIT_ASSERT_TRUE(expectedAccumGrad.isApprox(*(node3->grad)));

  Node* nodeGrad = graph.createNode();
  nodeGrad->values = new Matrix(2, 2, &arena);
  nodeGrad->values->setWith(4.0);
  nodeGrad->grad = new Matrix(2, 2, &arena);
  nodeGrad->requiresGrad = true;

  Node* nodeNoGrad = graph.createNode();
  nodeNoGrad->values = new Matrix(2, 2, &arena);
  nodeNoGrad->values->setWith(1.0);
  nodeNoGrad->grad = nullptr; // Constante/Entrada sin gradiente
  nodeNoGrad->requiresGrad = false;

  Node* nodeMixed = BitMth::ia::Ops<double>::add({nodeGrad, nodeNoGrad}, &arena);

  BIT_ASSERT_TRUE(nodeMixed->requiresGrad);

  Matrix expectedMixedForward(2, 2, &arena);
  expectedMixedForward.setWith(5.0);
  BIT_ASSERT_TRUE(expectedMixedForward.isApprox(*(nodeMixed->values)));

  nodeMixed->grad->setWith(3.0);
  nodeMixed->backward_fn(nodeMixed);

  Matrix expectedMixedGrad(2, 2, &arena);
  expectedMixedGrad.setWith(3.0);

  BIT_ASSERT_TRUE(expectedMixedGrad.isApprox(*(nodeGrad->grad)));
  BIT_ASSERT_EQ(nullptr, nodeNoGrad->grad);
}

BIT_TEST_CASE(OpsSub) {
  Graph graph;
  RecordGraph rGraph(graph);
  Arena arena(Arena::MB(2));

  Node* node1 = graph.createNode();
  node1->values = new Matrix(3, 3, &arena);
  node1->values->setWith(10.0);
  node1->grad = new Matrix(3, 3, &arena);
  node1->requiresGrad = true;

  Node* node2 = graph.createNode();
  node2->values = new Matrix(3, 3, &arena);
  node2->values->setWith(2.0);
  node2->grad = new Matrix(3, 3, &arena);
  node2->requiresGrad = true;

  Node* node3 = graph.createNode();
  node3->values = new Matrix(3, 3, &arena);
  node3->values->setWith(3.0);
  node3->grad = new Matrix(3, 3, &arena);
  node3->requiresGrad = true;

  Node* nodeR = BitMth::ia::Ops<double>::sub({node1, node2, node3}, &arena);

  Matrix resultForward(3, 3, &arena);
  resultForward.setWith(5.0);

  BIT_ASSERT_TRUE(resultForward.isApprox(*(nodeR->values)));
  BIT_ASSERT_EQ(size_t(3), nodeR->parents.size());
  BIT_ASSERT_EQ(node1, nodeR->parents[0]);
  BIT_ASSERT_EQ(node2, nodeR->parents[1]);
  BIT_ASSERT_EQ(node3, nodeR->parents[2]);
  BIT_ASSERT_TRUE(nodeR->requiresGrad);
  BIT_ASSERT_EQ(
    size_t(BitMth::ia::types::OpType::SUB),
    size_t(nodeR->operation)
  );

  nodeR->grad->setWith(1.5);
  nodeR->backward_fn(nodeR);

  Matrix expectedGradPos(3, 3, &arena);
  expectedGradPos.setWith(1.5);

  Matrix expectedGradNeg(3, 3, &arena);
  expectedGradNeg.setWith(-1.5);

  BIT_ASSERT_TRUE(expectedGradPos.isApprox(*(node1->grad)));
  BIT_ASSERT_TRUE(expectedGradNeg.isApprox(*(node2->grad)));
  BIT_ASSERT_TRUE(expectedGradNeg.isApprox(*(node3->grad)));

  nodeR->grad->setWith(0.5);
  nodeR->backward_fn(nodeR);

  Matrix expectedAccumPos(3, 3, &arena);
  expectedAccumPos.setWith(2.0); // 1.5 + 0.5

  Matrix expectedAccumNeg(3, 3, &arena);
  expectedAccumNeg.setWith(-2.0); // -1.5 - 0.5

  BIT_ASSERT_TRUE(expectedAccumPos.isApprox(*(node1->grad)));
  BIT_ASSERT_TRUE(expectedAccumNeg.isApprox(*(node2->grad)));
  BIT_ASSERT_TRUE(expectedAccumNeg.isApprox(*(node3->grad)));

  Node* nodeGrad = graph.createNode();
  nodeGrad->values = new Matrix(2, 2, &arena);
  nodeGrad->values->setWith(10.0);
  nodeGrad->grad = new Matrix(2, 2, &arena);
  nodeGrad->requiresGrad = true;

  Node* nodeNoGrad = graph.createNode();
  nodeNoGrad->values = new Matrix(2, 2, &arena);
  nodeNoGrad->values->setWith(3.0);
  nodeNoGrad->grad = nullptr; // Constante/Entrada sin gradiente
  nodeNoGrad->requiresGrad = false;

  Node* nodeMixed = BitMth::ia::Ops<double>::sub({nodeGrad, nodeNoGrad}, &arena);

  BIT_ASSERT_TRUE(nodeMixed->requiresGrad);

  Matrix expectedMixedForward(2, 2, &arena);
  expectedMixedForward.setWith(7.0);
  BIT_ASSERT_TRUE(expectedMixedForward.isApprox(*(nodeMixed->values)));

  nodeMixed->grad->setWith(3.0);
  nodeMixed->backward_fn(nodeMixed);

  Matrix expectedMixedGrad(2, 2, &arena);
  expectedMixedGrad.setWith(3.0);

  BIT_ASSERT_TRUE(expectedMixedGrad.isApprox(*(nodeGrad->grad)));
  BIT_ASSERT_EQ(nullptr, nodeNoGrad->grad);
}

BIT_TEST_CASE(OpsHadamard) {
  Graph graph;
  RecordGraph rGraph(graph);
  Arena arena(Arena::MB(2));

  Node* node1 = graph.createNode();
  node1->values = new Matrix(3, 3, &arena);
  node1->values->setWith(2.0);
  node1->grad = new Matrix(3, 3, &arena);
  node1->requiresGrad = true;

  Node* node2 = graph.createNode();
  node2->values = new Matrix(3, 3, &arena);
  node2->values->setWith(3.0);
  node2->grad = new Matrix(3, 3, &arena);
  node2->requiresGrad = true;

  Node* node3 = graph.createNode();
  node3->values = new Matrix(3, 3, &arena);
  node3->values->setWith(4.0);
  node3->grad = new Matrix(3, 3, &arena);
  node3->requiresGrad = true;

  Node* nodeR = BitMth::ia::Ops<double>::hadamard({node1, node2, node3}, &arena);

  // Forward: 2.0 * 3.0 * 4.0 = 24.0
  Matrix resultForward(3, 3, &arena);
  resultForward.setWith(24.0);

  BIT_ASSERT_TRUE(resultForward.isApprox(*(nodeR->values)));
  BIT_ASSERT_EQ(size_t(3), nodeR->parents.size());
  BIT_ASSERT_EQ(node1, nodeR->parents[0]);
  BIT_ASSERT_EQ(node2, nodeR->parents[1]);
  BIT_ASSERT_EQ(node3, nodeR->parents[2]);
  BIT_ASSERT_TRUE(nodeR->requiresGrad);
  BIT_ASSERT_EQ(
    size_t(BitMth::ia::types::OpType::HADAMARD),
    size_t(nodeR->operation)
  );

  // dL/dY = 1.5
  // grad(X0) = 1.5 * (X1 * X2) = 1.5 * (3 * 4) = 18.0
  // grad(X1) = 1.5 * (X0 * X2) = 1.5 * (2 * 4) = 12.0
  // grad(X2) = 1.5 * (X0 * X1) = 1.5 * (2 * 3) = 9.0
  nodeR->grad->setWith(1.5);
  nodeR->backward_fn(nodeR);

  Matrix expectedGrad1(3, 3, &arena);
  expectedGrad1.setWith(18.0);

  Matrix expectedGrad2(3, 3, &arena);
  expectedGrad2.setWith(12.0);

  Matrix expectedGrad3(3, 3, &arena);
  expectedGrad3.setWith(9.0);

  BIT_ASSERT_TRUE(expectedGrad1.isApprox(*(node1->grad)));
  BIT_ASSERT_TRUE(expectedGrad2.isApprox(*(node2->grad)));
  BIT_ASSERT_TRUE(expectedGrad3.isApprox(*(node3->grad)));

  // dL/dY nuevo = 0.5
  // grad_extra(X0) = 0.5 * 12 = 6.0  ==> Total = 18.0 + 6.0 = 24.0
  // grad_extra(X1) = 0.5 * 8  = 4.0  ==> Total = 12.0 + 4.0 = 16.0
  // grad_extra(X2) = 0.5 * 6  = 3.0  ==> Total = 9.0  + 3.0 = 12.0
  nodeR->grad->setWith(0.5);
  nodeR->backward_fn(nodeR);

  Matrix expectedAccum1(3, 3, &arena);
  expectedAccum1.setWith(24.0);

  Matrix expectedAccum2(3, 3, &arena);
  expectedAccum2.setWith(16.0);

  Matrix expectedAccum3(3, 3, &arena);
  expectedAccum3.setWith(12.0);

  BIT_ASSERT_TRUE(expectedAccum1.isApprox(*(node1->grad)));
  BIT_ASSERT_TRUE(expectedAccum2.isApprox(*(node2->grad)));
  BIT_ASSERT_TRUE(expectedAccum3.isApprox(*(node3->grad)));

  Node* nodeGrad = graph.createNode();
  nodeGrad->values = new Matrix(2, 2, &arena);
  nodeGrad->values->setWith(5.0);
  nodeGrad->grad = new Matrix(2, 2, &arena);
  nodeGrad->requiresGrad = true;

  Node* nodeNoGrad = graph.createNode();
  nodeNoGrad->values = new Matrix(2, 2, &arena);
  nodeNoGrad->values->setWith(3.0);
  nodeNoGrad->grad = nullptr; // Constante
  nodeNoGrad->requiresGrad = false;

  Node* nodeMixed = BitMth::ia::Ops<double>::hadamard({nodeGrad, nodeNoGrad}, &arena);

  BIT_ASSERT_TRUE(nodeMixed->requiresGrad);

  // Forward mixto: 5.0 * 3.0 = 15.0
  Matrix expectedMixedForward(2, 2, &arena);
  expectedMixedForward.setWith(15.0);
  BIT_ASSERT_TRUE(expectedMixedForward.isApprox(*(nodeMixed->values)));

  // Backward mixto: self->grad = 2.0
  // grad(nodeGrad) = 2.0 * 3.0 = 6.0
  nodeMixed->grad->setWith(2.0);
  nodeMixed->backward_fn(nodeMixed);

  Matrix expectedMixedGrad(2, 2, &arena);
  expectedMixedGrad.setWith(6.0);

  BIT_ASSERT_TRUE(expectedMixedGrad.isApprox(*(nodeGrad->grad)));
  BIT_ASSERT_EQ(nullptr, nodeNoGrad->grad);

  // Y = A * A = A^2  ==>  dY/dA = 2 * A
  Node* nodeA = graph.createNode();
  nodeA->values = new Matrix(2, 2, &arena);
  nodeA->values->setWith(3.0);
  nodeA->grad = new Matrix(2, 2, &arena);
  nodeA->requiresGrad = true;

  Node* nodeSquare = BitMth::ia::Ops<double>::hadamard({nodeA, nodeA}, &arena);

  // Forward: 3.0 * 3.0 = 9.0
  Matrix expectedSquareForward(2, 2, &arena);
  expectedSquareForward.setWith(9.0);
  BIT_ASSERT_TRUE(expectedSquareForward.isApprox(*(nodeSquare->values)));

  // Backward: self->grad = 1.0
  // grad(A) debe ser 2 * A * 1.0 = 2 * 3.0 = 6.0
  nodeSquare->grad->setWith(1.0);
  nodeSquare->backward_fn(nodeSquare);

  Matrix expectedSquareGrad(2, 2, &arena);
  expectedSquareGrad.setWith(6.0);

  BIT_ASSERT_TRUE(expectedSquareGrad.isApprox(*(nodeA->grad)));
}

BIT_TEST_CASE(OpsMul) {
  Graph graph;
  RecordGraph rGraph(graph);
  Arena arena(Arena::MB(2));

  // =============================================================
  // 1. Forward Pass con Matrices No Cuadradas
  // A: (2 x 3) llena de 2.0
  // B: (3 x 4) llena de 3.0
  // Salida Y = A * B -> Dimensión (2 x 4)
  // Cada elemento Y_ij = sum_k(2.0 * 3.0) = 3 * 6.0 = 18.0
  // =============================================================
  Node* nodeA = graph.createNode();
  nodeA->values = new Matrix(2, 3, &arena);
  nodeA->values->setWith(2.0);
  nodeA->grad = new Matrix(2, 3, &arena);
  nodeA->requiresGrad = true;

  Node* nodeB = graph.createNode();
  nodeB->values = new Matrix(3, 4, &arena);
  nodeB->values->setWith(3.0);
  nodeB->grad = new Matrix(3, 4, &arena);
  nodeB->requiresGrad = true;

  Node* nodeR = BitMth::ia::Ops<double>::mul(nodeA, nodeB, &arena);

  // Verificación de dimensiones y Forward
  Matrix resultForward(2, 4, &arena);
  resultForward.setWith(18.0);

  BIT_ASSERT_TRUE(resultForward.isApprox(*(nodeR->values)));
  BIT_ASSERT_EQ(size_t(2), nodeR->parents.size());
  BIT_ASSERT_EQ(nodeA, nodeR->parents[0]);
  BIT_ASSERT_EQ(nodeB, nodeR->parents[1]);
  BIT_ASSERT_TRUE(nodeR->requiresGrad);
  BIT_ASSERT_EQ(
    size_t(BitMth::ia::types::OpType::MUL),
    size_t(nodeR->operation)
  );

  // =============================================================
  // 2. Backward Pass Básico
  // Supongamos dL/dY = 0.5 (Matriz 2 x 4 llena de 0.5)
  //
  // dA = dY * B^T -> (2 x 4) * (4 x 3) = (2 x 3)
  // Cada elemento dA_ij = 4 * (0.5 * 3.0) = 6.0
  //
  // dB = A^T * dY -> (3 x 2) * (2 x 4) = (3 x 4)
  // Cada elemento dB_ij = 2 * (2.0 * 0.5) = 2.0
  // =============================================================
  nodeR->grad->setWith(0.5);
  nodeR->backward_fn(nodeR);

  Matrix expectedGradA(2, 3, &arena);
  expectedGradA.setWith(6.0);

  Matrix expectedGradB(3, 4, &arena);
  expectedGradB.setWith(2.0);

  BIT_ASSERT_TRUE(expectedGradA.isApprox(*(nodeA->grad)));
  BIT_ASSERT_TRUE(expectedGradB.isApprox(*(nodeB->grad)));

  // =============================================================
  // 3. Acumulación de Gradientes (Segunda Pasada Backward)
  // Nuevo dL/dY = 1.0
  // Delta dA = 4 * (1.0 * 3.0) = 12.0  ==> Total A = 6.0 + 12.0 = 18.0
  // Delta dB = 2 * (2.0 * 1.0) = 4.0   ==> Total B = 2.0 + 4.0  = 6.0
  // =============================================================
  nodeR->grad->setWith(1.0);
  nodeR->backward_fn(nodeR);

  Matrix expectedAccumGradA(2, 3, &arena);
  expectedAccumGradA.setWith(18.0);

  Matrix expectedAccumGradB(3, 4, &arena);
  expectedAccumGradB.setWith(6.0);

  BIT_ASSERT_TRUE(expectedAccumGradA.isApprox(*(nodeA->grad)));
  BIT_ASSERT_TRUE(expectedAccumGradB.isApprox(*(nodeB->grad)));

  // =============================================================
  // 4. Test con Entradas Mixtas (requiresGrad = true vs false)
  // =============================================================
  Node* nodeGrad = graph.createNode();
  nodeGrad->values = new Matrix(2, 2, &arena);
  nodeGrad->values->setWith(2.0);
  nodeGrad->grad = new Matrix(2, 2, &arena);
  nodeGrad->requiresGrad = true;

  Node* nodeNoGrad = graph.createNode();
  nodeNoGrad->values = new Matrix(2, 2, &arena);
  nodeNoGrad->values->setWith(4.0);
  nodeNoGrad->grad = nullptr; // Constante
  nodeNoGrad->requiresGrad = false;

  Node* nodeMixed = BitMth::ia::Ops<double>::mul(nodeGrad, nodeNoGrad, &arena);

  BIT_ASSERT_TRUE(nodeMixed->requiresGrad);

  // Forward mixto: (2x2) * (2x2) = 2 * (2.0 * 4.0) = 16.0
  Matrix expectedMixedForward(2, 2, &arena);
  expectedMixedForward.setWith(16.0);
  BIT_ASSERT_TRUE(expectedMixedForward.isApprox(*(nodeMixed->values)));

  // Backward mixto con dL/dY = 1.0
  // dA = dY * B^T = (2x2) * (2x2) = 2 * (1.0 * 4.0) = 8.0
  nodeMixed->grad->setWith(1.0);
  nodeMixed->backward_fn(nodeMixed);

  Matrix expectedMixedGradA(2, 2, &arena);
  expectedMixedGradA.setWith(8.0);

  BIT_ASSERT_TRUE(expectedMixedGradA.isApprox(*(nodeGrad->grad)));
  BIT_ASSERT_EQ(nullptr, nodeNoGrad->grad);
}

BIT_TEST_CASE(OpsRelu) {
    Graph graph;
    RecordGraph rGraph(graph);
    Arena arena(Arena::MB(2));

    // =============================================================
    // 1. Forward Pass
    // A: (2 x 3) con valores mixtos positivos, negativos y cero
    // =============================================================
    Node* nodeA = graph.createNode();
    nodeA->values = new Matrix(2, 3, &arena);
    nodeA->values->getValues()[0] = -1.5; nodeA->values->getValues()[1] =  0.0; nodeA->values->getValues()[2] = 2.0;
    nodeA->values->getValues()[3] =  3.0; nodeA->values->getValues()[4] = -0.5; nodeA->values->getValues()[5] = 0.0;
    nodeA->grad = new Matrix(2, 3, &arena);
    nodeA->requiresGrad = true;

    Node* nodeR = BitMth::ia::Ops<double>::relu(nodeA, &arena);

    // Verificación Forward
    Matrix expectedForward(2, 3, &arena);
    expectedForward.getValues()[0] = 0.0; expectedForward.getValues()[1] = 0.0; expectedForward.getValues()[2] = 2.0;
    expectedForward.getValues()[3] = 3.0; expectedForward.getValues()[4] = 0.0; expectedForward.getValues()[5] = 0.0;

    BIT_ASSERT_TRUE(expectedForward.isApprox(*(nodeR->values)));
    BIT_ASSERT_EQ(size_t(1), nodeR->parents.size());
    BIT_ASSERT_EQ(nodeA, nodeR->parents[0]);
    BIT_ASSERT_TRUE(nodeR->requiresGrad);
    BIT_ASSERT_EQ(size_t(BitMth::ia::types::OpType::RELU), size_t(nodeR->operation));

    // =============================================================
    // 2. Backward Pass Básico
    // dL/dY = Matriz de unos (1.0)
    // dA = dY * (A > 0 ? 1 : 0)
    // =============================================================
    nodeR->grad->setWith(1.0);
    nodeR->backward_fn(nodeR);

    Matrix expectedGradA(2, 3, &arena);
    expectedGradA.getValues()[0] = 0.0; expectedGradA.getValues()[1] = 0.0; expectedGradA.getValues()[2] = 1.0;
    expectedGradA.getValues()[3] = 1.0; expectedGradA.getValues()[4] = 0.0; expectedGradA.getValues()[5] = 0.0;

    BIT_ASSERT_TRUE(expectedGradA.isApprox(*(nodeA->grad)));

    // =============================================================
    // 3. Acumulación de Gradientes
    // dL/dY = 2.0 -> Nuevo delta dA = 2.0 * (A > 0)
    // Acumulado = Grad1 + Grad2
    // =============================================================
    nodeR->grad->setWith(2.0);
    nodeR->backward_fn(nodeR);

    Matrix expectedAccumGradA(2, 3, &arena);
    expectedAccumGradA.getValues()[0] = 0.0; expectedAccumGradA.getValues()[1] = 0.0; expectedAccumGradA.getValues()[2] = 3.0;
    expectedAccumGradA.getValues()[3] = 3.0; expectedAccumGradA.getValues()[4] = 0.0; expectedAccumGradA.getValues()[5] = 0.0;

    BIT_ASSERT_TRUE(expectedAccumGradA.isApprox(*(nodeA->grad)));

    // =============================================================
    // 4. Entradas sin Gradiente (requiresGrad = false)
    // =============================================================
    Node* nodeNoGrad = graph.createNode();
    nodeNoGrad->values = new Matrix(1, 2, &arena);
    nodeNoGrad->values->setWith(1.0);
    nodeNoGrad->requiresGrad = false;

    Node* nodeNoGradResult = BitMth::ia::Ops<double>::relu(nodeNoGrad, &arena);
    BIT_ASSERT_FALSE(nodeNoGradResult->requiresGrad);
}

BIT_TEST_CASE(OpsSigmoid) {
    Graph graph;
    RecordGraph rGraph(graph);
    Arena arena(Arena::MB(2));

    Node* nodeA = graph.createNode();
    nodeA->values = new Matrix(1, 2, &arena);
    nodeA->values->getValues()[0] = 0.0;  // sigmoid(0) = 0.5
    nodeA->values->getValues()[1] = 2.0;  // sigmoid(2) = 0.8807970779778823
    nodeA->grad = new Matrix(1, 2, &arena);
    nodeA->requiresGrad = true;

    Node* nodeR = BitMth::ia::Ops<double>::sigmoid(nodeA, &arena);

    // Forward
    double sig0 = 0.5;
    double sig2 = 1.0 / (1.0 + std::exp(-2.0));
    Matrix expectedForward(1, 2, &arena);
    expectedForward.getValues()[0] = sig0;
    expectedForward.getValues()[1] = sig2;

    BIT_ASSERT_TRUE(expectedForward.isApprox(*(nodeR->values)));
    BIT_ASSERT_EQ(size_t(BitMth::ia::types::OpType::SIGMOID), size_t(nodeR->operation));

    // Backward: dZ = grad * A * (1 - A)
    nodeR->grad->setWith(1.0);
    nodeR->backward_fn(nodeR);

    Matrix expectedGradA(1, 2, &arena);
    expectedGradA.getValues()[0] = 1.0 * sig0 * (1.0 - sig0); // 0.25
    expectedGradA.getValues()[1] = 1.0 * sig2 * (1.0 - sig2);

    BIT_ASSERT_TRUE(expectedGradA.isApprox(*(nodeA->grad)));

    // Acumulación
    nodeR->backward_fn(nodeR); // Acumula el mismo gradiente una segunda vez
    Matrix expectedAccumGradA(1, 2, &arena);
    expectedAccumGradA.getValues()[0] = 2.0 * (sig0 * (1.0 - sig0));
    expectedAccumGradA.getValues()[1] = 2.0 * (sig2 * (1.0 - sig2));

    BIT_ASSERT_TRUE(expectedAccumGradA.isApprox(*(nodeA->grad)));
}

BIT_TEST_CASE(OpsTanh) {
    Graph graph;
    RecordGraph rGraph(graph);
    Arena arena(Arena::MB(2));

    Node* nodeA = graph.createNode();
    nodeA->values = new Matrix(1, 2, &arena);
    nodeA->values->getValues()[0] = 0.0; // tanh(0) = 0.0
    nodeA->values->getValues()[1] = 1.0; // tanh(1) = 0.7615941559557649
    nodeA->grad = new Matrix(1, 2, &arena);
    nodeA->requiresGrad = true;

    Node* nodeR = BitMth::ia::Ops<double>::tanH(nodeA, &arena);

    // Forward
    double tanh0 = 0.0;
    double tanh1 = std::tanh(1.0);
    Matrix expectedForward(1, 2, &arena);
    expectedForward.getValues()[0] = tanh0;
    expectedForward.getValues()[1] = tanh1;

    BIT_ASSERT_TRUE(expectedForward.isApprox(*(nodeR->values)));
    BIT_ASSERT_EQ(size_t(BitMth::ia::types::OpType::TANH), size_t(nodeR->operation));

    // Backward: dZ = grad * (1 - A^2)
    nodeR->grad->setWith(1.0);
    nodeR->backward_fn(nodeR);

    Matrix expectedGradA(1, 2, &arena);
    expectedGradA.getValues()[0] = 1.0 * (1.0 - tanh0 * tanh0); // 1.0
    expectedGradA.getValues()[1] = 1.0 * (1.0 - tanh1 * tanh1);

    BIT_ASSERT_TRUE(expectedGradA.isApprox(*(nodeA->grad)));

    // Acumulación
    nodeR->backward_fn(nodeR);
    Matrix expectedAccumGradA(1, 2, &arena);
    expectedAccumGradA.getValues()[0] = 2.0 * (1.0 - tanh0 * tanh0);
    expectedAccumGradA.getValues()[1] = 2.0 * (1.0 - tanh1 * tanh1);

    BIT_ASSERT_TRUE(expectedAccumGradA.isApprox(*(nodeA->grad)));
}

BIT_TEST_CASE(OpsSoftmax) {
    Graph graph;
    RecordGraph rGraph(graph);
    Arena arena(Arena::MB(2));

    Node* nodeA = graph.createNode();
    nodeA->values = new Matrix(1, 3, &arena);
    nodeA->values->getValues()[0] = 1.0;
    nodeA->values->getValues()[1] = 2.0;
    nodeA->values->getValues()[2] = 3.0;
    nodeA->grad = new Matrix(1, 3, &arena);
    nodeA->requiresGrad = true;

    Node* nodeR = BitMth::ia::Ops<double>::softmax(nodeA, &arena);

    // Forward calculado manualmente
    double exp1 = std::exp(1.0 - 3.0);
    double exp2 = std::exp(2.0 - 3.0);
    double exp3 = std::exp(3.0 - 3.0);
    double sum = exp1 + exp2 + exp3;

    Matrix expectedForward(1, 3, &arena);
    expectedForward.getValues()[0] = exp1 / sum;
    expectedForward.getValues()[1] = exp2 / sum;
    expectedForward.getValues()[2] = exp3 / sum;

    BIT_ASSERT_TRUE(expectedForward.isApprox(*(nodeR->values)));
    BIT_ASSERT_EQ(size_t(BitMth::ia::types::OpType::SOFTMAX), size_t(nodeR->operation));

    // Backward Pass con gradOut = [1.0, 0.0, 0.0]
    // dZ_i = A_i * (grad_i - dotProduct) donde dotProduct = grad * A = 1.0 * A_0
    nodeR->grad->getValues()[0] = 1.0;
    nodeR->grad->getValues()[1] = 0.0;
    nodeR->grad->getValues()[2] = 0.0;
    nodeR->backward_fn(nodeR);

    double a0 = expectedForward.getValues()[0];
    double a1 = expectedForward.getValues()[1];
    double a2 = expectedForward.getValues()[2];
    double dotProduct = 1.0 * a0;

    Matrix expectedGradA(1, 3, &arena);
    expectedGradA.getValues()[0] = a0 * (1.0 - dotProduct);
    expectedGradA.getValues()[1] = a1 * (0.0 - dotProduct);
    expectedGradA.getValues()[2] = a2 * (0.0 - dotProduct);

    BIT_ASSERT_TRUE(expectedGradA.isApprox(*(nodeA->grad)));
}

BIT_TEST_CASE(OpsMSELoss) {
    Graph graph;
    RecordGraph rGraph(graph);
    Arena arena(Arena::MB(2));

    // Predicción (1x3) y Real (1x3)
    Node* nodePred = graph.createNode();
    nodePred->values = new Matrix(1, 3, &arena);
    nodePred->values->getValues()[0] = 2.0;
    nodePred->values->getValues()[1] = 3.0;
    nodePred->values->getValues()[2] = 4.0;
    nodePred->grad = new Matrix(1, 3, &arena);
    nodePred->requiresGrad = true;

    Node* nodeReal = graph.createNode();
    nodeReal->values = new Matrix(1, 3, &arena);
    nodeReal->values->getValues()[0] = 1.0;
    nodeReal->values->getValues()[1] = 1.0;
    nodeReal->values->getValues()[2] = 1.0;

    // Forward Pass
    Node* nodeR = BitMth::ia::Ops<double>::mse(nodePred, nodeReal, &arena);

    // Forward calculado manualmente
    // Diff = [1.0, 2.0, 3.0] -> Cuadrados = [1.0, 4.0, 9.0] -> Suma = 14.0 -> MSE (N=3) = 14.0 / 3.0
    Matrix expectedForward(1, 1, &arena);
    expectedForward.getValues()[0] = 14.0 / 3.0;

    BIT_ASSERT_TRUE(expectedForward.isApprox(*(nodeR->values)));
    BIT_ASSERT_EQ(size_t(BitMth::ia::types::OpType::MSE_LOSS), size_t(nodeR->operation));

    // Backward Pass con gradOut = [1.0] (Semilla del gradiente en la raíz 1x1)
    // dMSE / dPred = (2 / N) * (pred - real) * gradScal
    nodeR->grad = new Matrix(1, 1, &arena);
    nodeR->grad->getValues()[0] = 1.0;
    nodeR->backward_fn(nodeR);

    double invN = (2.0 * 1.0) / 3.0; // (2 * gradScal) / N
    Matrix expectedGradPred(1, 3, &arena);
    expectedGradPred.getValues()[0] = (2.0 - 1.0) * invN; // 2/3
    expectedGradPred.getValues()[1] = (3.0 - 1.0) * invN; // 4/3
    expectedGradPred.getValues()[2] = (4.0 - 1.0) * invN; // 6/3 = 2.0

    BIT_ASSERT_TRUE(expectedGradPred.isApprox(*(nodePred->grad)));
}

BIT_TEST_CASE(OpsBCELoss) {
    Graph graph;
    RecordGraph rGraph(graph);
    Arena arena(Arena::MB(5));

    // Predicción (1x2) y Real (1x2)
    Node* nodePred = graph.createNode();
    nodePred->values = new Matrix(1, 2, &arena);
    nodePred->values->getValues()[0] = 0.8;
    nodePred->values->getValues()[1] = 0.2;
    nodePred->grad = new Matrix(1, 2, &arena);
    nodePred->requiresGrad = true;

    Node* nodeReal = graph.createNode();
    nodeReal->values = new Matrix(1, 2, &arena);
    nodeReal->values->getValues()[0] = 1.0;
    nodeReal->values->getValues()[1] = 0.0;

    // Forward Pass
    Node* nodeR = BitMth::ia::Ops<double>::bce(nodePred, nodeReal, &arena);

    // Forward calculado manualmente
    // Pos 0: 1.0 * ln(0.8)
    // Pos 1: (1.0 - 0.0) * ln(1.0 - 0.2) = ln(0.8)
    // Loss = -(ln(0.8) + ln(0.8)) / 2 = -ln(0.8)
    Matrix expectedForward(1, 1, &arena);
    expectedForward.getValues()[0] = -std::log(0.8);

    BIT_ASSERT_TRUE(expectedForward.isApprox(*(nodeR->values)));
    BIT_ASSERT_EQ(size_t(BitMth::ia::types::OpType::BCE_LOSS), size_t(nodeR->operation));

    // Backward Pass con gradOut = [1.0]
    // dBCE / dPred = ((pred - real) / (pred * (1 - pred))) * (gradScal / N)
    nodeR->grad = new Matrix(1, 1, &arena);
    nodeR->grad->getValues()[0] = 1.0;
    nodeR->backward_fn(nodeR);

    double scale = 1.0 / 2.0; // gradScal / N (donde N = 2)
    Matrix expectedGradPred(1, 2, &arena);
    expectedGradPred.getValues()[0] = ((0.8 - 1.0) / (0.8 * 0.2)) * scale; // -0.625
    expectedGradPred.getValues()[1] = ((0.2 - 0.0) / (0.2 * 0.8)) * scale; //  0.625

    BIT_ASSERT_TRUE(expectedGradPred.isApprox(*(nodePred->grad)));
}

BIT_GROUP_END()
