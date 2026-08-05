#pragma once

#include <vector>
#include <functional>

#include <BitMth/ia/autograd/types/OpTypes.hpp>

namespace BitMth::ia{
  template<typename T>
  struct Node{
    T grad;
    T* container;
    std::vector<Node<T>*> parents;
    bool requiresGrad{false};
    types::OpType operation{types::OpType::NONE};
    std::function<void(Node<T>*)> backward_fn{nullptr};
  };
}
