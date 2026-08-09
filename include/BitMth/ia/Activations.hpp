#pragma once

#include <unordered_map>
#include <BitMth/ia/ActivationFunctions.hpp>

namespace BitMth::ia{
    template<typename T>
    inline const ia::types::ActivationContent<T>& getActivationFunction(const ia::types::ActivationFunctType type){
        static std::unordered_map<ia::types::ActivationFunctType, ia::types::ActivationContent<T>> functTable ={
            {ia::types::ActivationFunctType::RELU,    {ActivationFunctions<T>::relu, ActivationFunctions<T>::reluDerivative}},
            {ia::types::ActivationFunctType::SIGMOID, {ActivationFunctions<T>::sigmoid, ActivationFunctions<T>::sigmoidDerivative}},
            {ia::types::ActivationFunctType::TANH,    {ActivationFunctions<T>::tanh, ActivationFunctions<T>::tanhDerivative}},
            {ia::types::ActivationFunctType::SOFTMAX, {ActivationFunctions<T>::softmax, ActivationFunctions<T>::sigmoidDerivative}}
        };
        return functTable.at(type);
    }
}
