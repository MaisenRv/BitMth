#pragma once

namespace BitMth::ia::types{

  enum class OpType {
    NONE, // Para tensores hoja (pesos, parámetros)

    // === SCALAR - MATRIX & MATRIX - MATRIX (Element-Wise / Binary) ===
    ADD,   // Suma (Matrix + Matrix o Matrix + Scalar)
    SUB,   // Resta (Matrix - Matrix o Matrix - Scalar)
    HADAMARD,// Multiplicación elemento a elemento
    // SCALE, // Multiplicación por escalar

    // === MATRIX - MATRIX (Linear Algebra) ===
    MUL, // Multiplicación de matrices (A x B)

    // === UNARY ACTIVATIONS & NORM (Single Matrix) ===
    RELU,
    // GELU,
    SIGMOID,
    TANH,
    SOFTMAX,
    // LAYERNORM,

    // === LOSSES ===
    MSE_LOSS,
    BCE_LOSS    

    // === SHAPE & TRANSFORMATIONS ===
    // TRANSPOSE,
    // RESHAPE

    // === BITWISE (Futuro: Int/Quantized Tensors) ===
    // BITWISE_AND,
    // BITWISE_OR,
    // BITWISE_XOR
  };
}
