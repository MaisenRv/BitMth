#pragma once

#include <vector>

#include <BitMth/ia/autograd/types/OpTypes.hpp>

namespace BitMth::ia{
  template<typename T>
  struct Node{
    T grad;
    T* container;
    std::vector<Node<T>*> parents;
    bool requiresGrad{false};
    types::OpType operation{types::OpType::NONE};
    void (*backward_fn)(Node<T>* self){nullptr};
  };
}
