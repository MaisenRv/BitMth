#pragma once
#include <unordered_map>

#include <BitMth/ia/LossFunctions.hpp>

namespace BitMth::ia{
    template<typename T>
    inline const ia::types::LossContent<T>& getLossFunction(const ia::types::LossFunctType type){
        static std::unordered_map<ia::types::LossFunctType, ia::types::LossContent<T>> functTable ={
            {ia::types::LossFunctType::MSE,                  {LossFunctions<T>::mse, LossFunctions<T>::mseDerivative}},
            {ia::types::LossFunctType::BINARY_CROSS_ENTROPY, {LossFunctions<T>::bce, LossFunctions<T>::bceDerivative}}
        };
        return functTable.at(type);
    } 
}
