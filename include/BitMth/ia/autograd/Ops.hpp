#pragma once

#include <BitMth/ia/autograd/Node.hpp>
#include <BitMth/ia/autograd/ComputationGraph.hpp>
#include <BitMth/ia/autograd/types/OpTypes.hpp>
#include <BitMth/ia/ActivationFunctions.hpp>
#include <BitMth/ia/LossFunctions.hpp>
#include <BitMth/linalg/Matrix.hpp>
#include <stdexcept>

namespace BitMth::ia{
  template<typename T>
  struct [[nodiscard]] Ops{
    private:
      using Matrix = linalg::Matrix<T>;
      using N = Node<Matrix>;
      using Graph = ComputationGraph<Matrix>;

      static Graph* _getGraph(){
        if(Graph::getComputationGraph() == nullptr){
          std::runtime_error("The ComputationGraph does not exist.");
        }
        return Graph::getComputationGraph();
      }

      static N* _createNodeHelper(const std::vector<N*>& inputs, types::OpType type){
        Graph* graph = _getGraph();
        N* newNode = graph->createNode();
        newNode->parents = inputs;
        newNode->operation = type;

        for (N* nodeInput : inputs) {
          if (nodeInput->requiresGrad){
            newNode->requiresGrad = true;
            break;
          }
        }
        return newNode;
      }
      static void _elementWiseHelper(N* element,const std::vector<N*>& inputs, core::Arena* targetArena){
        element->values = new Matrix(
          inputs[0]->values->getRows(),
          inputs[0]->values->getCols(),
          targetArena
        );
        if (element->requiresGrad) {
            element->grad = new Matrix(
                inputs[0]->values->getRows(),
                inputs[0]->values->getCols(),
                targetArena
            );
        }
      }
  //     static Node<Matrix>* _createUnaryNodeHelper(const Matrix& in, Matrix& out,core::Arena *arena, ComputationGraph<Matrix>* graph, types::OpType type){
  //       Node<Matrix>* nodeIn = _getOrCreateNode(graph, const_cast<Matrix&>(in),arena);

  //       Node<Matrix>* nodeResult = graph->createNode();
  //       nodeResult->requiresGrad = true;
  //       nodeResult->parents = {nodeIn};
  //       nodeResult->operation = type;
  //       nodeResult->container = &out;
  //       nodeResult->grad = Matrix(out.getRows(),out.getCols(),arena);
  //       return nodeResult;
  //     }
    public:

    static inline N* add(const std::vector<N*>& inputs, core::Arena* targetArena) {
      if (inputs.empty()) return nullptr;

      N* newNode = _createNodeHelper(inputs, types::OpType::ADD);
      _elementWiseHelper(newNode, inputs, targetArena);

      for (N* nodeInput : inputs) {
        *(newNode->values) += *(nodeInput->values);
      }

      newNode->backward_fn = [](N* self) {
          if (!self->grad || self->parents.empty()) return;
          for (N* parent : self->parents) {
              if (parent->requiresGrad) {
                *(parent->grad) += *(self->grad);
              }
          }
      };
      return newNode;

    }
    
    
    static inline N* sub(const std::vector<N*>& inputs,core::Arena *targetArena){
      if (inputs.empty()) return nullptr;

      N* newNode = _createNodeHelper(inputs, types::OpType::SUB);
      _elementWiseHelper(newNode, inputs, targetArena);

      *(newNode->values) += *(inputs[0]->values);
      for (size_t i = 1; i < inputs.size(); ++i) {
          *(newNode->values) -= *(inputs[i]->values);
      }

      newNode->backward_fn = [](N* self) {
          if (!self->grad || self->parents.empty()) return;

          N* parent0 = self->parents[0];
          if (parent0->requiresGrad && parent0->grad) {
              *(parent0->grad) += *(self->grad);
          }

          for (size_t i = 1; i < self->parents.size(); ++i) {
              N* parent = self->parents[i];
              if (parent->requiresGrad && parent->grad) {
                  *(parent->grad) -= *(self->grad);
              }
          }
      };

      return newNode;
    }

    static inline N* hadamard(const std::vector<N*>& inputs, core::Arena* targetArena){
      if (inputs.empty()) return nullptr;

      N* newNode = _createNodeHelper(inputs, types::OpType::HADAMARD);
      _elementWiseHelper(newNode, inputs, targetArena);
      newNode->values->setOne();

      for (N* nodeInput : inputs) {
        newNode->values->hadamardInPlace(*(nodeInput->values));
      }
      newNode->backward_fn = [](N* self) {
          if (!self->grad || self->parents.empty()) return;
          const size_t numParents = self->parents.size();

          for (size_t i = 0; i < numParents; ++i) {
              N* currentParent = self->parents[i];

              if (currentParent->requiresGrad && currentParent->grad) {
                  Matrix term = *(self->grad);

                  for (size_t j = 0; j < numParents; ++j) {
                      if (i != j) term.hadamardInPlace(*(self->parents[j]->values));
                  }

                  *(currentParent->grad) += term;
              }
          }
      };

      return newNode;
    }

    static inline N* mul(N* A,N* B,core::Arena *targetArena){
      N* newNode = _createNodeHelper({A,B}, types::OpType::MUL);

      newNode->values = new Matrix(
        A->values->getRows(),
        B->values->getCols(),
        targetArena
      );
      if (newNode->requiresGrad) {
          newNode->grad = new Matrix(
              newNode->values->getRows(),
              newNode->values->getCols(),
              targetArena
          );
      }

      *(newNode->values) = *(A->values) * (*(B->values));

      newNode->backward_fn = [](N* self) {
          if (!self->grad || self->parents.size() < 2) return;

          N* parentA = self->parents[0]; // Dim (m x k)
          N* parentB = self->parents[1]; // Dim (k x n)

          // dA = dY * B^T
          if (parentA->requiresGrad && parentA->grad) {
              *(parentA->grad) += *(self->grad) * Matrix::t(*(parentB->values));
          }

          // dB = A^T * dY
          if (parentB->requiresGrad && parentB->grad) {
              *(parentB->grad) += Matrix::t(*(parentA->values)) * (*(self->grad));
          }
      };
      return newNode;
    }

  //   static inline Matrix relu(const Matrix& Z,core::Arena *targetArena = nullptr ){
  //     Matrix result = ActivationFunctions<T>::relu(Z, targetArena);
  //     ComputationGraph<Matrix>* graph = ComputationGraph<Matrix>::getComputationGraph();

  //     if(graph == nullptr) return result;

  //     if(Z.getRequiresGrad()){

  //         Node<Matrix>* nodeResult = _createUnaryNodeHelper(Z, result,targetArena, graph, types::OpType::RELU);
  //         nodeResult->backward_fn = [targetArena](Node<Matrix>* self){
  //           Node<Matrix>* nodeZ = self->parents[0];
  //           if(nodeZ != nullptr){
  //             Matrix dZ = ActivationFunctions<T>::reluDerivative(*nodeZ->container, *self->container,targetArena);
  //             nodeZ->grad += Ops<T>::hadamard(self->grad, dZ, targetArena);
  //           }
  //         };
  //         result.setRequiresGrad(true);
  //         result.setAutogradNode(nodeResult);
  //     }
  //     return result;
  //   }

  //   static inline Matrix sigmoid(const Matrix& Z,core::Arena *targetArena = nullptr ){
  //     Matrix result = ActivationFunctions<T>::sigmoid(Z, targetArena);
  //     ComputationGraph<Matrix>* graph = ComputationGraph<Matrix>::getComputationGraph();

  //     if(graph == nullptr) return result;

  //     if(Z.getRequiresGrad()){

  //         Node<Matrix>* nodeResult = _createUnaryNodeHelper(Z, result,targetArena, graph, types::OpType::SIGMOID);
  //         nodeResult->backward_fn = [targetArena](Node<Matrix>* self){
  //           Node<Matrix>* nodeZ = self->parents[0];
  //           if(nodeZ != nullptr){
  //             Matrix dA = ActivationFunctions<T>::sigmoidDerivative(*nodeZ->container, *self->container,targetArena);
  //             nodeZ->grad += Ops<T>::hadamard(self->grad, dA, targetArena);
  //           }
  //         };
  //         result.setRequiresGrad(true);
  //         result.setAutogradNode(nodeResult);
  //     }
  //     return result;
  //   }

  //   static inline Matrix tanH(const Matrix& Z,core::Arena *targetArena = nullptr ){
  //     Matrix result = ActivationFunctions<T>::tanh(Z, targetArena);
  //     ComputationGraph<Matrix>* graph = ComputationGraph<Matrix>::getComputationGraph();

  //     if(graph == nullptr) return result;

  //     if(Z.getRequiresGrad()){

  //         Node<Matrix>* nodeResult = _createUnaryNodeHelper(Z, result,targetArena, graph, types::OpType::TANH);
  //         nodeResult->backward_fn = [targetArena](Node<Matrix>* self){
  //           Node<Matrix>* nodeZ = self->parents[0];
  //           if(nodeZ != nullptr){
  //             Matrix dA = ActivationFunctions<T>::tanhDerivative(*nodeZ->container, *self->container,targetArena);
  //             nodeZ->grad += Ops<T>::hadamard(self->grad, dA, targetArena);
  //           }
  //         };
  //         result.setRequiresGrad(true);
  //         result.setAutogradNode(nodeResult);
  //     }
  //     return result;
  //   }

  //   static inline Matrix softmax(const Matrix& Z, core::Arena* targetArena = nullptr) {
  //       Matrix result = ActivationFunctions<T>::softmax(Z, targetArena);

  //       ComputationGraph<Matrix>* graph = ComputationGraph<Matrix>::getComputationGraph();
  //       if (graph == nullptr) return result;

  //       if (Z.getRequiresGrad()) {
  //           Node<Matrix>* nodeResult = _createUnaryNodeHelper(Z, result,targetArena, graph, types::OpType::SOFTMAX);

  //           nodeResult->backward_fn = [targetArena](Node<Matrix>* self) {
  //               Node<Matrix>* nodeZ = self->parents[0];
  //               if (nodeZ != nullptr) {
  //                   Matrix dZ = ActivationFunctions<T>::softmaxDerivative(
  //                       *self->container, 
  //                       self->grad, 
  //                       targetArena
  //                   );
  //                   nodeZ->grad += dZ;
  //               }
  //           };
  //           result.setRequiresGrad(true);
  //           result.setAutogradNode(nodeResult);
  //       }
  //       return result;
  //   }

  //   static Matrix mse(const Matrix& predict, const Matrix& real, core::Arena* targetArena = nullptr) {
  //     Matrix lossMatrix = LossFunctions<T>::mse(predict, real, targetArena);

  //     ComputationGraph<Matrix>* graph = ComputationGraph<Matrix>::getComputationGraph();
  //     if (graph != nullptr && predict.getRequiresGrad()) {
  //         Matrix& nonConstPred = const_cast<Matrix&>(predict);
  //         Node<Matrix>* nodePred = _getOrCreateNode(graph, nonConstPred, targetArena);

  //         Node<Matrix>* nodeLoss = graph->createNode();
  //         nodeLoss->requiresGrad = true;
  //         nodeLoss->parents = {nodePred};
  //         nodeLoss->operation = types::OpType::MSE_LOSS;
  //         nodeLoss->grad = Matrix(lossMatrix.getRows(), lossMatrix.getCols(), targetArena);
  //         nodeLoss->grad.fill(0);

  //         lossMatrix.setRequiresGrad(true);
  //         lossMatrix.setAutogradNode(nodeLoss);

  //         nodeLoss->backward_fn = [nonConstPred, real, targetArena](Node<Matrix>* self) {
  //             Node<Matrix>* nodeP = self->parents[0];
  //             if (nodeP != nullptr && nodeP->requiresGrad) {
  //                 Matrix gradMatrix = LossFunctions<T>::mseDerivative(nonConstPred, real, targetArena);
  //                 nodeP->grad += gradMatrix;
  //             }
  //         };
  //     }

  //     return lossMatrix;

  //   }

  //   static Matrix bce(const Matrix& predict, const Matrix& real, core::Arena* targetArena = nullptr) {
  //     Matrix lossMatrix = LossFunctions<T>::bce(predict, real, targetArena);

  //     ComputationGraph<Matrix>* graph = ComputationGraph<Matrix>::getComputationGraph();
  //     if (graph != nullptr && predict.getRequiresGrad()) {
  //         Matrix& nonConstPred = const_cast<Matrix&>(predict);
  //         Node<Matrix>* nodePred = _getOrCreateNode(graph, nonConstPred, targetArena);

  //         Node<Matrix>* nodeLoss = graph->createNode();
  //         nodeLoss->requiresGrad = true;
  //         nodeLoss->parents = {nodePred};
  //         nodeLoss->operation = types::OpType::BCE_LOSS;
        
  //         nodeLoss->grad = Matrix(lossMatrix.getRows(), lossMatrix.getCols(), targetArena);
  //         nodeLoss->grad.fill(0);

  //         lossMatrix.setRequiresGrad(true);
  //         lossMatrix.setAutogradNode(nodeLoss);

  //         nodeLoss->backward_fn = [predict, real, targetArena](Node<Matrix>* self) {
  //             Node<Matrix>* nodeP = self->parents[0];
  //             if (nodeP != nullptr && nodeP->requiresGrad) {
  //                 Matrix gradMatrix = LossFunctions<T>::bceDerivative(predict, real, targetArena);
  //                 nodeP->grad += gradMatrix;
  //             }
  //         };
  //     }
  //     return lossMatrix;

  //   }
  };
}
