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
    static N* addVector(N* matNode, N* vecNode, core::Arena* targetArena = nullptr) {
      if (!matNode || !vecNode || !matNode->values || !vecNode->values) return nullptr;

      N* newNode = _createNodeHelper({matNode, vecNode}, types::OpType::ADD_VECTOR);
      _elementWiseHelper(newNode, {matNode, vecNode}, targetArena);

      *(newNode->values) = Matrix::addRowVectorS(*(matNode->values), *(vecNode->values), targetArena);

      newNode->backward_fn = [targetArena](N* self) {
          if (!self || !self->grad || self->parents.size() < 2) return;

          N* matParent = self->parents[0];
          N* vecParent = self->parents[1];

          if (matParent && matParent->requiresGrad && matParent->grad) {
              *(matParent->grad) += *(self->grad);
          }

          if (vecParent && vecParent->requiresGrad && vecParent->grad) {
              if (vecParent->values->getRows() == 1) {
                  *(vecParent->grad) += self->grad->reduceSumRows(targetArena);
              } else if (vecParent->values->getCols() == 1) {
                  *(vecParent->grad) += self->grad->reduceSumCols(targetArena);
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

    static inline N* relu(N* A,core::Arena *targetArena){
      if (!A) return nullptr;
      N* newNode = _createNodeHelper({A}, types::OpType::RELU);
      _elementWiseHelper(newNode, {A}, targetArena);

      *(newNode->values) = ActivationFunctions<T>::relu(*(A->values),targetArena); 
      newNode->backward_fn = [targetArena](N* self) {
        if (!self->grad || self->parents.empty()) return;

        N* parentA = self->parents[0]; 

        if (parentA->requiresGrad && parentA->grad) {
            *(parentA->grad) += ActivationFunctions<T>::reluDerivative(*(self->values), *(self->grad),targetArena); 
        }
      };
      return newNode;
    }

    static inline N* sigmoid(N* A,core::Arena *targetArena){
      if (!A) return nullptr;
      N* newNode = _createNodeHelper({A}, types::OpType::SIGMOID);
      _elementWiseHelper(newNode, {A}, targetArena);

      *(newNode->values) = ActivationFunctions<T>::sigmoid(*(A->values),targetArena); 
      newNode->backward_fn = [targetArena](N* self) {
        if (!self->grad || self->parents.empty()) return;

        N* parentA = self->parents[0]; 

        if (parentA->requiresGrad && parentA->grad) {
            *(parentA->grad) += ActivationFunctions<T>::sigmoidDerivative(*(self->values), *(self->grad),targetArena); 
        }
      };
      return newNode;
    }

    static inline N* tanH(N* A,core::Arena *targetArena){
      if (!A) return nullptr;
      N* newNode = _createNodeHelper({A}, types::OpType::TANH);
      _elementWiseHelper(newNode, {A}, targetArena);

      *(newNode->values) = ActivationFunctions<T>::tanh(*(A->values),targetArena); 
      newNode->backward_fn = [targetArena](N* self) {
        if (!self->grad || self->parents.empty()) return;

        N* parentA = self->parents[0]; 

        if (parentA->requiresGrad && parentA->grad) {
            *(parentA->grad) += ActivationFunctions<T>::tanhDerivative(*(self->values), *(self->grad),targetArena); 
        }
      };
      return newNode;
    }

    static inline N* softmax(N* A, core::Arena* targetArena) {
      if (!A) return nullptr;
      N* newNode = _createNodeHelper({A}, types::OpType::SOFTMAX);
      _elementWiseHelper(newNode, {A}, targetArena);

      *(newNode->values) = ActivationFunctions<T>::softmax(*(A->values),targetArena); 
      newNode->backward_fn = [targetArena](N* self) {
        if (!self->grad || self->parents.empty()) return;

        N* parentA = self->parents[0]; 

        if (parentA->requiresGrad && parentA->grad) {
            *(parentA->grad) += ActivationFunctions<T>::softmaxDerivative(*(self->values), *(self->grad),targetArena); 
        }
      };
      return newNode;
    }

    static N* mse(N* predict, N* real, core::Arena* targetArena) {
      if (!real || !predict) return nullptr;
      N* newNode = _createNodeHelper({predict,real}, types::OpType::MSE_LOSS);
      newNode->values = new Matrix(1,1,targetArena);
      if (newNode->requiresGrad) {
          newNode->grad = new Matrix(1,1,targetArena);
      }
      *(newNode->values) = LossFunctions<T>::mse(*(predict->values), *(real->values),targetArena);

      newNode->backward_fn = [targetArena](N* self) {
          if (!self->grad || self->parents.empty()) return;

          N* predNode = self->parents[0];
          N* realNode = self->parents[1];

          if (predNode->requiresGrad && predNode->grad) {
              T gradScal = self->grad->getValues()[0];

              *(predNode->grad) += LossFunctions<T>::mseDerivative(
                  *(predNode->values), 
                  *(realNode->values), 
                  gradScal, 
                  targetArena
              );
          }
      };
      return newNode;
    }

    static N* bce(N* predict, N* real, core::Arena* targetArena = nullptr) {
      if (!real || !predict) return nullptr;
      N* newNode = _createNodeHelper({predict,real}, types::OpType::BCE_LOSS);
      newNode->values = new Matrix(1,1,targetArena);
      if (newNode->requiresGrad) {
          newNode->grad = new Matrix(1,1,targetArena);
      }
      *(newNode->values) = LossFunctions<T>::bce(*(predict->values), *(real->values),targetArena);

      newNode->backward_fn = [targetArena](N* self) {
          if (!self->grad || self->parents.empty()) return;

          N* predNode = self->parents[0];
          N* realNode = self->parents[1];

          if (predNode->requiresGrad && predNode->grad) {
              T gradScal = self->grad->getValues()[0];

              *(predNode->grad) += LossFunctions<T>::bceDerivative(
                  *(predNode->values), 
                  *(realNode->values), 
                  gradScal, 
                  targetArena
              );
          }
      };
      return newNode;
    }
  };
}
