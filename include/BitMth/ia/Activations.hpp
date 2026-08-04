#pragma once

#include <unordered_map>
#include <BitMth/ia/ActivationFunctions.hpp>
#include <BitMth/ia/autograd/Ops.hpp>

namespace BitMth::ia{
    template<typename T>
    inline const ia::types::ActivationContent<T>& getActivationFunction(const ia::types::ActivationFunctType type, bool activateAutograd = false){
        if(activateAutograd){
            static std::unordered_map<ia::types::ActivationFunctType, ia::types::ActivationContent<T>> functTable ={
                {ia::types::ActivationFunctType::RELU,    {Ops<T>::relu, nullptr}},
                {ia::types::ActivationFunctType::SIGMOID, {Ops<T>::sigmoid, nullptr}},
                {ia::types::ActivationFunctType::TANH,    {Ops<T>::tanH, nullptr}},
                {ia::types::ActivationFunctType::SOFTMAX, {Ops<T>::softmax, nullptr}}
            };
            return functTable.act(type);
        }
        static std::unordered_map<ia::types::ActivationFunctType, ia::types::ActivationContent<T>> functTable ={
            {ia::types::ActivationFunctType::RELU,    {ActivationFunctions<T>::relu, ActivationFunctions<T>::reluDerivative}},
            {ia::types::ActivationFunctType::SIGMOID, {ActivationFunctions<T>::sigmoid, ActivationFunctions<T>::sigmoidDerivative}},
            {ia::types::ActivationFunctType::TANH,    {ActivationFunctions<T>::tanh, ActivationFunctions<T>::tanhDerivative}},
            {ia::types::ActivationFunctType::SOFTMAX, {ActivationFunctions<T>::softmax, ActivationFunctions<T>::sigmoidDerivative}}
        };
        return functTable.at(type);
    }
}
