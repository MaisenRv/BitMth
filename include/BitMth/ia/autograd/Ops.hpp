#pragma once

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

      static Node<Matrix>* _getOrCreateNode(ComputationGraph<Matrix>* graph,Matrix& matrix){
        if(!matrix.getRequiresGrad())           return nullptr;
        if(matrix.getAutogradNode() != nullptr) return matrix.getAutogradNode();

        Node<Matrix>* node = graph->createNode();
        node->requiresGrad = true;
        node->container = &matrix;
        matrix.setAutogradNode(node);
        return node;
      }

      static Node<Matrix>* _createNodeHelper(Matrix& A,Matrix& B,Matrix& out, ComputationGraph<Matrix>* graph, types::OpType type){
        Node<Matrix>* nodeA = _getOrCreateNode(graph, A);
        Node<Matrix>* nodeB = _getOrCreateNode(graph, B);

        Node<Matrix>* nodeResult = graph->createNode();
        nodeResult->requiresGrad = true;
        nodeResult->parents = {nodeA, nodeB};
        nodeResult->operation = type;
        nodeResult->container = &out;
        return nodeResult;
      }
      static Node<Matrix>* _createUnaryNodeHelper(Matrix& in, Matrix& out, ComputationGraph<Matrix>* graph, types::OpType type){
        Node<Matrix>* nodeIn = _getOrCreateNode(graph, in);

        Node<Matrix>* nodeResult = graph->createNode();
        nodeResult->requiresGrad = true;
        nodeResult->parents = {nodeIn};
        nodeResult->operation = type;
        nodeResult->container = &out;
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
            Node<Matrix>* nodeResult = _createNodeHelper(A, B, result, graph, types::OpType::ADD);
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

          Node<Matrix>* nodeResult = _createNodeHelper(A, B,result, graph, types::OpType::SUB);
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

          Node<Matrix>* nodeResult = _createNodeHelper(A, B,result, graph, types::OpType::HADAMARD);
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

      if(graph == nullptr) return result;

      if(A.getRequiresGrad() || B.getRequiresGrad()){

          Node<Matrix>* nodeResult = _createNodeHelper(A, B,result, graph, types::OpType::MUL);
          nodeResult->backward_fn = [targetArena](Node<Matrix>* self){
            Node<Matrix>* nodeA = self->parents[0];
            Node<Matrix>* nodeB = self->parents[1];
            if(nodeA != nullptr){
              nodeA->grad += Matrix::mul(
                self->grad,
                Matrix::t(*nodeB->container,targetArena)
              );
            }
            if(nodeB != nullptr){
              nodeB->grad += Matrix::mul(
                Matrix::t(*nodeA->container,targetArena),
                self->grad
              );
            }
          };
          result.setRequiresGrad(true);
          result.setAutogradNode(nodeResult);
      }
      return result;
    }

    static inline Matrix relu(const Matrix& Z,core::Arena *targetArena = nullptr ){
      Matrix result = ActivationFunctions<T>::relu(Z, targetArena);
      ComputationGraph<Matrix>* graph = ComputationGraph<Matrix>::getComputationGraph();

      if(graph == nullptr) return result;

      if(Z.getRequiresGrad()){

          Node<Matrix>* nodeResult = _createUnaryNodeHelper(Z, result, graph, types::OpType::RELU);
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

          Node<Matrix>* nodeResult = _createUnaryNodeHelper(Z, result, graph, types::OpType::SIGMOID);
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
      Matrix result = ActivationFunctions<T>::Tanh(Z, targetArena);
      ComputationGraph<Matrix>* graph = ComputationGraph<Matrix>::getComputationGraph();

      if(graph == nullptr) return result;

      if(Z.getRequiresGrad()){

          Node<Matrix>* nodeResult = _createUnaryNodeHelper(Z, result, graph, types::OpType::TANH);
          nodeResult->backward_fn = [targetArena](Node<Matrix>* self){
            Node<Matrix>* nodeZ = self->parents[0];
            if(nodeZ != nullptr){
              Matrix dA = ActivationFunctions<T>::TanhDerivative(*nodeZ->container, *self->container,targetArena);
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
            Node<Matrix>* nodeResult = _createUnaryNodeHelper(Z, result, graph, types::OpType::SOFTMAX);

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
        T lossVal = LossFunctions<T>::mse(predict, real, targetArena);
        Matrix lossMatrix(1, 1, targetArena);
        lossMatrix(0, 0) = lossVal;

        ComputationGraph<Matrix>* graph = ComputationGraph<Matrix>::getComputationGraph();
        if (graph != nullptr && predict.getRequiresGrad()) {
            
            Node<Matrix>* nodeLoss = _createUnaryNodeHelper(predict, lossMatrix, graph, types::OpType::MSE_LOSS);

            nodeLoss->backward_fn = [&predict, &real, targetArena](Node<Matrix>* self) {
                Node<Matrix>* nodePred = self->parents[0];

                if (nodePred != nullptr) {
                    Matrix gradMatrix = LossFunctions<T>::mseDerivative(predict, real, targetArena);
                    nodePred->grad += gradMatrix;
                }
            };
            lossMatrix.setRequiresGrad(true);
            lossMatrix.setAutogradNode(nodeLoss);
        }

        return lossMatrix;
    }

    static Matrix bce(const Matrix& predict, const Matrix& real, core::Arena* targetArena = nullptr) {
        T lossVal = LossFunctions<T>::bce(predict, real, targetArena);
        Matrix lossMatrix(1, 1, targetArena);
        lossMatrix(0, 0) = lossVal;

        ComputationGraph<Matrix>* graph = ComputationGraph<Matrix>::getComputationGraph();
        if (graph != nullptr && predict.getRequiresGrad()) {
            Node<Matrix>* nodeLoss = _createUnaryNodeHelper(predict, lossMatrix, graph, types::OpType::BCE_LOSS);
            nodeLoss->backward_fn = [&predict, &real, targetArena](Node<Matrix>* self) {
                Node<Matrix>* nodePred = self->parents[0];

                if (nodePred != nullptr) {
                    Matrix gradMatrix = LossFunctions<T>::bceDerivative(predict, real, targetArena);
                    nodePred->grad += gradMatrix;
                }
            };
            lossMatrix.setRequiresGrad(true);
            lossMatrix.setAutogradNode(nodeLoss);
        }
        return lossMatrix;
    }
  };
}
