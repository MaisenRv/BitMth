#pragma once

#include <cmath>
#include <unordered_map>

#include <BitMth/ia/types/OptimizerTypes.hpp>
#include <BitMth/utils/Constants.hpp>

namespace BitMth::ia{
  template<typename T>
  struct Optimizer{
    private:
      using Matrix = linalg::Matrix<T>;
    public:
      static void sgd(Matrix &parameter,const Matrix &delta, T learningRate, types::OptimizerState<T> &state, T weightDecay = utils::WEIGHT_DECAY<T>){
        parameter -= (learningRate * delta);
        state.stepCount++;
      }
     
      static void sgdMomentum(Matrix &parameter,const Matrix &delta, T learningRate, types::OptimizerState<T> &state, T weightDecay = utils::WEIGHT_DECAY<T>){
        if(state.stepCount == 0){
          state.firstMoment = Matrix(parameter.getRows(), parameter.getCols(), nullptr, true);
        }

        const T beta = utils::MOMENTUM_FACTOR<T>;
        state.firstMoment = Matrix::matrixApplyFunction(
          state.firstMoment, delta,nullptr,
          [beta](T mA, T mB){ return (beta * mA) + mB; }
        );

        parameter -= (learningRate * state.firstMoment);
        state.stepCount++;
      }
    
      static void adam(Matrix &parameter,const Matrix &delta, T learningRate, types::OptimizerState<T> &state, T weightDecay = utils::WEIGHT_DECAY<T>){
        if(state.stepCount == 0){
          state.firstMoment = Matrix(parameter.getRows(), parameter.getCols(), nullptr, true);
          state.secondMoment = Matrix(parameter.getRows(), parameter.getCols(), nullptr, true);
        }
        state.stepCount++;

        const T beta1 = utils::MOMENTUM_FACTOR<T>;
        const T beta2 = utils::MOMENTUM_FACTOR2<T>;
        const T difBeta1 = T(1) - beta1;
        const T difBeta2 = T(1) - beta2;
        state.firstMoment = Matrix::matrixApplyFunction(
          state.firstMoment, delta,nullptr,
          [beta1, difBeta1](T mA, T mB){ return (beta1 * mA) + (difBeta1 * mB); }
        );

        state.secondMoment = Matrix::matrixApplyFunction(
          state.secondMoment, delta,nullptr,
          [beta2, difBeta2](T mA, T mB){ return (beta2 * mA) + (difBeta2 * mB * mB); }
        );

        const T _mBiasVar = 1 - std::pow(beta1, T(state.stepCount));
        const T _sBiasVar = 1 - std::pow(beta2, T(state.stepCount));

        const T alpha_t = learningRate / _mBiasVar;
        const T epsilon = utils::EPSILON<T>;

        parameter -= Matrix::matrixApplyFunction(
            state.firstMoment, state.secondMoment, nullptr,
            [alpha_t, _sBiasVar, epsilon](T m_val, T s_val) { 
                T s_hat = s_val / _sBiasVar;
                return (alpha_t / (std::sqrt(s_hat) + epsilon)) * m_val; 
            }
        );
      }

      static void adamW(Matrix &parameter, const Matrix &delta,T learningRate, types::OptimizerState<T> &state, T weightDecay = utils::WEIGHT_DECAY<T>) {
        if(state.stepCount == 0){
          state.firstMoment = Matrix(parameter.getRows(), parameter.getCols(), nullptr, true);
          state.secondMoment = Matrix(parameter.getRows(), parameter.getCols(), nullptr, true);
        }

        state.stepCount++;

        const T beta1 = utils::MOMENTUM_FACTOR<T>;
        const T beta2 = utils::MOMENTUM_FACTOR2<T>;
        const T difBeta1 = static_cast<T>(1) - beta1;
        const T difBeta2 = static_cast<T>(1) - beta2;

        state.firstMoment = Matrix::matrixApplyFunction(
            state.firstMoment, delta, nullptr,
            [beta1, difBeta1](T mA, T mB) { 
                return (beta1 * mA) + (difBeta1 * mB); 
            }
        );

        state.secondMoment = Matrix::matrixApplyFunction(
            state.secondMoment, delta, nullptr,
            [beta2, difBeta2](T mA, T mB) { 
                return (beta2 * mA) + (difBeta2 * (mB * mB)); 
            }
        );

        const T _mBiasVar = static_cast<T>(1) - std::pow(beta1, static_cast<T>(state.stepCount));
        const T _sBiasVar = static_cast<T>(1) - std::pow(beta2, static_cast<T>(state.stepCount));

        const T alpha_t = learningRate / _mBiasVar;
        const T epsilon = utils::EPSILON<T>;

        if (weightDecay > static_cast<T>(0)) {
            const T decayFactor = static_cast<T>(1) - (learningRate * weightDecay);
            parameter *= decayFactor;
        }

        parameter -= Matrix::matrixApplyFunction(
            state.firstMoment, state.secondMoment, nullptr,
            [alpha_t, _sBiasVar, epsilon](T m_val, T s_val) { 
                T s_hat = s_val / _sBiasVar;
                return (alpha_t / (std::sqrt(s_hat) + epsilon)) * m_val; 
            }
        ); 
      }
  };

  template<typename T>
  inline const types::OptimizerContent<T> getOptimizer(const types::OptimizerType type){
    static std::unordered_map<types::OptimizerType, types::OptimizerContent<T>> functTable ={
      {types::OptimizerType::SGD,          {Optimizer<T>::sgd}},
      {types::OptimizerType::SGD_MOMENTUM, {Optimizer<T>::sgdMomentum}},
      {types::OptimizerType::ADAM,         {Optimizer<T>::adam}},
      {types::OptimizerType::ADAM_W,       {Optimizer<T>::adamW}},
    };
    return functTable[type];
  }

}
