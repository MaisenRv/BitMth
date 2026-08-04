#pragma once

#include <memory>
#include <vector>

#include <BitMth/ia/autograd/Node.hpp>

namespace BitMth::ia{
  template <typename T>
  class ComputationGraph{
    private:
      inline static thread_local ComputationGraph<T>* activeGraph{nullptr};
      std::vector<std::unique_ptr<Node<T>>> nodes;
    public:
      ComputationGraph() = default;
      ComputationGraph(const ComputationGraph&) = delete;
      ComputationGraph& operator=(const ComputationGraph&) = delete;

      Node<T>* createNode() {
        std::unique_ptr<Node<T>> newNode = std::make_unique<Node<T>>();
        Node<T>* ptr = newNode.get();
        nodes.push_back(std::move(newNode)); 
        return ptr;
      }

      void backward(Node<T>* lossNode){
        
      }
      
      void clear(){ nodes.clear(); }

      void _makeActive() { activeGraph = this; }
      void _deactivate() { activeGraph = nullptr; }

      static ComputationGraph<T>* getComputationGraph(){return activeGraph;}
      static bool isCreateGraph(){ return activeGraph != nullptr; }
  };

  template<typename T>
  class RecordGraphGuard {
  private:
    ComputationGraph<T>& graphRef;
  public:
    explicit RecordGraphGuard(ComputationGraph<T>& graph) : graphRef(graph) {
      graphRef._makeActive(); 
    }
    RecordGraphGuard(const RecordGraphGuard&) = delete;
    RecordGraphGuard& operator=(const RecordGraphGuard&) = delete;

    ~RecordGraphGuard() { graphRef._deactivate(); }
  };

}
