#pragma once

#include "BitMth/core/Arena.hpp"
#include <BitMth/ia/autograd/Node.hpp>
#include <BitMth/ia/autograd/ComputationGraph.hpp>
#include <BitMth/ia/autograd/types/OpTypes.hpp>
#include <BitMth/ia/ActivationFunctions.hpp>
#include <BitMth/ia/LossFunctions.hpp>

namespace BitMth::ia{
  template<typename T>
  struct [[nodiscard]] Ops{
    private:
      using Matrix = linalg::Matrix<T>;

      static Node<Matrix>* _getOrCreateNode(ComputationGraph<Matrix>* graph, Matrix& matrix, core::Arena *arena){
        if (matrix.getAutogradNode() != nullptr) {
            auto* node = matrix.getAutogradNode();
            node->container = &matrix;
            return node;
        }

        Node<Matrix>* node = graph->createNode();
        node->requiresGrad = matrix.getRequiresGrad();
        node->container = &matrix;
        node->grad = Matrix(matrix.getRows(),matrix.getCols(),arena);
        matrix.setAutogradNode(node);
        return node;
      }

      static Node<Matrix>* _createNodeHelper(Matrix& A,Matrix& B,Matrix& out,core::Arena *arena, ComputationGraph<Matrix>* graph, types::OpType type){
        Node<Matrix>* nodeA = _getOrCreateNode(graph, A, arena);
        Node<Matrix>* nodeB = _getOrCreateNode(graph, B, arena);

        Node<Matrix>* nodeResult = graph->createNode();
        nodeResult->requiresGrad = true;
        nodeResult->parents = {nodeA, nodeB};
        nodeResult->operation = type;
        nodeResult->container = &out;
        nodeResult->grad = Matrix(out.getRows(),out.getCols(),arena);
        return nodeResult;
      }
      static Node<Matrix>* _createUnaryNodeHelper(const Matrix& in, Matrix& out,core::Arena *arena, ComputationGraph<Matrix>* graph, types::OpType type){
        Node<Matrix>* nodeIn = _getOrCreateNode(graph, const_cast<Matrix&>(in),arena);

        Node<Matrix>* nodeResult = graph->createNode();
        nodeResult->requiresGrad = true;
        nodeResult->parents = {nodeIn};
        nodeResult->operation = type;
        nodeResult->container = &out;
        nodeResult->grad = Matrix(out.getRows(),out.getCols(),arena);
        return nodeResult;
      }
    public:

    static inline Matrix add(Matrix& A, Matrix& B, core::Arena* targetArena = nullptr) {
        const bool isRowVectorBias = (B.getRows() == 1 && A.getRows() > 1 && A.getCols() == B.getCols());
        Matrix result;
        if (isRowVectorBias) {
            result = A.addRowVector(B, targetArena);
        } else {
            result = Matrix::add(A, B, targetArena);
        }

        ComputationGraph<Matrix>* graph = ComputationGraph<Matrix>::getComputationGraph();
        if (graph == nullptr) return result;

        if (A.getRequiresGrad() || B.getRequiresGrad()) {
            Node<Matrix>* nodeResult = _createNodeHelper(A, B, result,targetArena, graph, types::OpType::ADD);
            nodeResult->backward_fn = [isRowVectorBias, targetArena](Node<Matrix>* self) {
                Node<Matrix>* nodeA = self->parents[0];
                Node<Matrix>* nodeB = self->parents[1];

                if (nodeA != nullptr) {
                    nodeA->grad += self->grad;
                }

                if (nodeB != nullptr) {
                    if (isRowVectorBias) {
                        nodeB->grad += self->grad.reduceSumRows(targetArena);
                    } else {
                        nodeB->grad += self->grad;
                    }
                }
            };

            result.setRequiresGrad(true);
            result.setAutogradNode(nodeResult);
        }
        return result;
    }
    
    
    static inline Matrix sub(Matrix& A,Matrix& B,core::Arena *targetArena = nullptr ){
      Matrix result = Matrix::sub(A, B, targetArena);
      ComputationGraph<Matrix>* graph = ComputationGraph<Matrix>::getComputationGraph();

      if(graph == nullptr) return result;

      if(A.getRequiresGrad() || B.getRequiresGrad()){

          Node<Matrix>* nodeResult = _createNodeHelper(A, B,result,targetArena, graph, types::OpType::SUB);
          nodeResult->backward_fn = [](Node<Matrix>* self){
            Node<Matrix>* nodeA = self->parents[0];
            Node<Matrix>* nodeB = self->parents[1];
            if(nodeA != nullptr){
              nodeA->grad += self->grad;
            }
            if(nodeB != nullptr){
              nodeB->grad -= self->grad;
            }
          };
          result.setRequiresGrad(true);
          result.setAutogradNode(nodeResult);
      }
      return result;
    }

    static inline Matrix hadamard(Matrix& A,Matrix& B,core::Arena *targetArena = nullptr ){
      Matrix result = Matrix::hadamard(A, B, targetArena);
      ComputationGraph<Matrix>* graph = ComputationGraph<Matrix>::getComputationGraph();

      if(graph == nullptr) return result;

      if(A.getRequiresGrad() || B.getRequiresGrad()){

          Node<Matrix>* nodeResult = _createNodeHelper(A, B,result,targetArena, graph, types::OpType::HADAMARD);
          nodeResult->backward_fn = [targetArena](Node<Matrix>* self){
            Node<Matrix>* nodeA = self->parents[0];
            Node<Matrix>* nodeB = self->parents[1];
            if(nodeA != nullptr){
              nodeA->grad += Matrix::hadamard(self->grad, *nodeB->container,targetArena);
            }
            if(nodeB != nullptr){
              nodeB->grad += Matrix::hadamard(self->grad, *nodeA->container,targetArena);
            }
          };
          result.setRequiresGrad(true);
          result.setAutogradNode(nodeResult);
      }
      return result;
    }

    static inline Matrix mul(Matrix& A,Matrix& B,core::Arena *targetArena = nullptr ){
      Matrix result = Matrix::mul(A, B, targetArena);
      ComputationGraph<Matrix>* graph = ComputationGraph<Matrix>::getComputationGraph();

      if (graph == nullptr) return result;

      if (A.getRequiresGrad() || B.getRequiresGrad()) {
          Node<Matrix>* nodeA = _getOrCreateNode(graph, A, targetArena);
          Node<Matrix>* nodeB = _getOrCreateNode(graph, B, targetArena);

          Node<Matrix>* nodeResult = graph->createNode();
          nodeResult->requiresGrad = true;
          nodeResult->parents = {nodeA, nodeB};
          nodeResult->operation = types::OpType::MUL;
          
          result.setRequiresGrad(true);
          result.setAutogradNode(nodeResult);
          nodeResult->container = nullptr;
          nodeResult->grad = Matrix(result.getRows(), result.getCols(), targetArena);
          nodeResult->grad.fill(0);

          Matrix* ptrA = &A;
          Matrix* ptrB = &B;

          nodeResult->backward_fn = [nodeA, nodeB, ptrA, ptrB, targetArena](Node<Matrix>* self) {
            if (nodeA != nullptr && nodeA->requiresGrad) {
              Matrix bT = Matrix::t(*ptrB, targetArena);
              nodeA->grad += Matrix::mul(self->grad, bT, targetArena);
            }
            if (nodeB != nullptr && nodeB->requiresGrad) {
              Matrix aT = Matrix::t(*ptrA, targetArena);
              nodeB->grad += Matrix::mul(aT, self->grad, targetArena);
            }
          };
      }
      return result;

    }

    static inline Matrix relu(const Matrix& Z,core::Arena *targetArena = nullptr ){
      Matrix result = ActivationFunctions<T>::relu(Z, targetArena);
      ComputationGraph<Matrix>* graph = ComputationGraph<Matrix>::getComputationGraph();

      if(graph == nullptr) return result;

      if(Z.getRequiresGrad()){

          Node<Matrix>* nodeResult = _createUnaryNodeHelper(Z, result,targetArena, graph, types::OpType::RELU);
          nodeResult->backward_fn = [targetArena](Node<Matrix>* self){
            Node<Matrix>* nodeZ = self->parents[0];
            if(nodeZ != nullptr){
              Matrix dZ = ActivationFunctions<T>::reluDerivative(*nodeZ->container, *self->container,targetArena);
              nodeZ->grad += Ops<T>::hadamard(self->grad, dZ, targetArena);
            }
          };
          result.setRequiresGrad(true);
          result.setAutogradNode(nodeResult);
      }
      return result;
    }

    static inline Matrix sigmoid(const Matrix& Z,core::Arena *targetArena = nullptr ){
      Matrix result = ActivationFunctions<T>::sigmoid(Z, targetArena);
      ComputationGraph<Matrix>* graph = ComputationGraph<Matrix>::getComputationGraph();

      if(graph == nullptr) return result;

      if(Z.getRequiresGrad()){

          Node<Matrix>* nodeResult = _createUnaryNodeHelper(Z, result,targetArena, graph, types::OpType::SIGMOID);
          nodeResult->backward_fn = [targetArena](Node<Matrix>* self){
            Node<Matrix>* nodeZ = self->parents[0];
            if(nodeZ != nullptr){
              Matrix dA = ActivationFunctions<T>::sigmoidDerivative(*nodeZ->container, *self->container,targetArena);
              nodeZ->grad += Ops<T>::hadamard(self->grad, dA, targetArena);
            }
          };
          result.setRequiresGrad(true);
          result.setAutogradNode(nodeResult);
      }
      return result;
    }

    static inline Matrix tanH(const Matrix& Z,core::Arena *targetArena = nullptr ){
      Matrix result = ActivationFunctions<T>::tanh(Z, targetArena);
      ComputationGraph<Matrix>* graph = ComputationGraph<Matrix>::getComputationGraph();

      if(graph == nullptr) return result;

      if(Z.getRequiresGrad()){

          Node<Matrix>* nodeResult = _createUnaryNodeHelper(Z, result,targetArena, graph, types::OpType::TANH);
          nodeResult->backward_fn = [targetArena](Node<Matrix>* self){
            Node<Matrix>* nodeZ = self->parents[0];
            if(nodeZ != nullptr){
              Matrix dA = ActivationFunctions<T>::tanhDerivative(*nodeZ->container, *self->container,targetArena);
              nodeZ->grad += Ops<T>::hadamard(self->grad, dA, targetArena);
            }
          };
          result.setRequiresGrad(true);
          result.setAutogradNode(nodeResult);
      }
      return result;
    }

    static inline Matrix softmax(const Matrix& Z, core::Arena* targetArena = nullptr) {
        Matrix result = ActivationFunctions<T>::softmax(Z, targetArena);

        ComputationGraph<Matrix>* graph = ComputationGraph<Matrix>::getComputationGraph();
        if (graph == nullptr) return result;

        if (Z.getRequiresGrad()) {
            Node<Matrix>* nodeResult = _createUnaryNodeHelper(Z, result,targetArena, graph, types::OpType::SOFTMAX);

            nodeResult->backward_fn = [targetArena](Node<Matrix>* self) {
                Node<Matrix>* nodeZ = self->parents[0];
                if (nodeZ != nullptr) {
                    Matrix dZ = ActivationFunctions<T>::softmaxDerivative(
                        *self->container, 
                        self->grad, 
                        targetArena
                    );
                    nodeZ->grad += dZ;
                }
            };
            result.setRequiresGrad(true);
            result.setAutogradNode(nodeResult);
        }
        return result;
    }

    static Matrix mse(const Matrix& predict, const Matrix& real, core::Arena* targetArena = nullptr) {
      Matrix lossMatrix = LossFunctions<T>::mse(predict, real, targetArena);

      ComputationGraph<Matrix>* graph = ComputationGraph<Matrix>::getComputationGraph();
      if (graph != nullptr && predict.getRequiresGrad()) {
          Matrix& nonConstPred = const_cast<Matrix&>(predict);
          Node<Matrix>* nodePred = _getOrCreateNode(graph, nonConstPred, targetArena);

          Node<Matrix>* nodeLoss = graph->createNode();
          nodeLoss->requiresGrad = true;
          nodeLoss->parents = {nodePred};
          nodeLoss->operation = types::OpType::MSE_LOSS;
          nodeLoss->grad = Matrix(lossMatrix.getRows(), lossMatrix.getCols(), targetArena);
          nodeLoss->grad.fill(0);

          lossMatrix.setRequiresGrad(true);
          lossMatrix.setAutogradNode(nodeLoss);

          nodeLoss->backward_fn = [nonConstPred, real, targetArena](Node<Matrix>* self) {
              Node<Matrix>* nodeP = self->parents[0];
              if (nodeP != nullptr && nodeP->requiresGrad) {
                  Matrix gradMatrix = LossFunctions<T>::mseDerivative(nonConstPred, real, targetArena);
                  nodeP->grad += gradMatrix;
              }
          };
      }

      return lossMatrix;

    }

    static Matrix bce(const Matrix& predict, const Matrix& real, core::Arena* targetArena = nullptr) {
      Matrix lossMatrix = LossFunctions<T>::bce(predict, real, targetArena);

      ComputationGraph<Matrix>* graph = ComputationGraph<Matrix>::getComputationGraph();
      if (graph != nullptr && predict.getRequiresGrad()) {
          Matrix& nonConstPred = const_cast<Matrix&>(predict);
          Node<Matrix>* nodePred = _getOrCreateNode(graph, nonConstPred, targetArena);

          Node<Matrix>* nodeLoss = graph->createNode();
          nodeLoss->requiresGrad = true;
          nodeLoss->parents = {nodePred};
          nodeLoss->operation = types::OpType::BCE_LOSS;
        
          nodeLoss->grad = Matrix(lossMatrix.getRows(), lossMatrix.getCols(), targetArena);
          nodeLoss->grad.fill(0);

          lossMatrix.setRequiresGrad(true);
          lossMatrix.setAutogradNode(nodeLoss);

          nodeLoss->backward_fn = [predict, real, targetArena](Node<Matrix>* self) {
              Node<Matrix>* nodeP = self->parents[0];
              if (nodeP != nullptr && nodeP->requiresGrad) {
                  Matrix gradMatrix = LossFunctions<T>::bceDerivative(predict, real, targetArena);
                  nodeP->grad += gradMatrix;
              }
          };
      }
      return lossMatrix;

    }
  };
}
