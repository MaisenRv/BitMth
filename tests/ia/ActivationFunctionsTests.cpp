#include <BitTest/BitTest.hpp>
#include <BitMth/ia/ActivationFunctions.hpp>
#include <BitMth/utils/MathUtils.hpp>

BIT_GROUP_BEGIN(activationFunct)

BIT_TEST_CASE(ReluActivation) {
    using Act = BitMth::ia::ActivationFunctions<float>;
    using Matrix = BitMth::linalg::Matrix<float>;

    // Matriz de prueba: [-2.0, 0.0, 3.5]
    Matrix Z(1, 3);
    Z.getValues()[0] = -2.0f;
    Z.getValues()[1] =  0.0f;
    Z.getValues()[2] =  3.5f;

    Matrix res = Act::relu(Z);

    BIT_ASSERT_EQ(res.getRows(), static_cast<size_t>(1));
    BIT_ASSERT_EQ(res.getCols(), static_cast<size_t>(3));

    BIT_EXPECT_TRUE(BitMth::utils::isClose(res.getValues()[0], 0.0f));
    BIT_EXPECT_TRUE(BitMth::utils::isClose(res.getValues()[1], 0.0f));
    BIT_EXPECT_TRUE(BitMth::utils::isClose(res.getValues()[2], 3.5f));
}

BIT_TEST_CASE(SigmoidActivation) {
    using Act = BitMth::ia::ActivationFunctions<double>;
    using Matrix = BitMth::linalg::Matrix<double>;

    Matrix Z(1, 2);
    Z.getValues()[0] = 0.0;
    Z.getValues()[1] = 2.0;

    Matrix res = Act::sigmoid(Z);

    BIT_ASSERT_EQ(res.getRows(), static_cast<size_t>(1));
    BIT_ASSERT_EQ(res.getCols(), static_cast<size_t>(2));

    // sigmoid(0) = 0.5
    BIT_EXPECT_TRUE(BitMth::utils::isClose(res.getValues()[0], 0.5));
    // sigmoid(2) ≈ 0.880797
    BIT_EXPECT_TRUE(BitMth::utils::isClose(res.getValues()[1], 0.880797));
}

BIT_TEST_CASE(TanhActivation) {
    using Act = BitMth::ia::ActivationFunctions<float>;
    using Matrix = BitMth::linalg::Matrix<float>;

    Matrix Z(1, 2);
    Z.getValues()[0] = 0.0f;
    Z.getValues()[1] = 1.0f;

    Matrix res = Act::tanh(Z);

    // tanh(0) = 0
    BIT_EXPECT_TRUE(BitMth::utils::isClose(res.getValues()[0], 0.0f));
    // tanh(1) ≈ 0.761594f
    BIT_EXPECT_TRUE(BitMth::utils::isClose(res.getValues()[1], 0.761594f));
}

BIT_TEST_CASE(SoftmaxActivation) {
    using Act = BitMth::ia::ActivationFunctions<float>;
    using Matrix = BitMth::linalg::Matrix<float>;

    // Matriz de 1 fila y 3 columnas: [1.0, 2.0, 3.0]
    Matrix Z(1, 3);
    Z.getValues()[0] = 1.0f;
    Z.getValues()[1] = 2.0f;
    Z.getValues()[2] = 3.0f;

    Matrix res = Act::softmax(Z);

    BIT_ASSERT_EQ(res.getRows(), static_cast<size_t>(1));
    BIT_ASSERT_EQ(res.getCols(), static_cast<size_t>(3));

    // Comprobar que las probabilidades sumen 1.0
    float sum = res.getValues()[0] + res.getValues()[1] + res.getValues()[2];
    BIT_EXPECT_TRUE(BitMth::utils::isClose(sum, 1.0f));

    // Comprobar la propiedad de orden (e^1 < e^2 < e^3)
    BIT_EXPECT_TRUE(res.getValues()[0] < res.getValues()[1]);
    BIT_EXPECT_TRUE(res.getValues()[1] < res.getValues()[2]);
}

BIT_TEST_CASE(SoftmaxDerivative) {
    using Act = BitMth::ia::ActivationFunctions<float>;
    using Matrix = BitMth::linalg::Matrix<float>;

    Matrix Z(1, 2);
    Z.getValues()[0] = 0.0f;
    Z.getValues()[1] = 0.0f;

    Matrix A = Act::softmax(Z); // [0.5, 0.5]
    
    // Gradiente entrante simulado dL/dA = [1.0, 0.0]
    Matrix grad(1, 2);
    grad.getValues()[0] = 1.0f;
    grad.getValues()[1] = 0.0f;

    Matrix dZ = Act::softmaxDerivative(A, grad);

    BIT_ASSERT_EQ(dZ.getRows(), static_cast<size_t>(1));
    BIT_ASSERT_EQ(dZ.getCols(), static_cast<size_t>(2));

    // s0*(g0 - dot) = 0.5*(1.0 - 0.5) = 0.25
    // s1*(g1 - dot) = 0.5*(0.0 - 0.5) = -0.25
    BIT_EXPECT_TRUE(BitMth::utils::isClose(dZ.getValues()[0],  0.25f));
    BIT_EXPECT_TRUE(BitMth::utils::isClose(dZ.getValues()[1], -0.25f));
}

BIT_TEST_CASE(ActivationFunctionsEmptyMatrix) {
    using Act = BitMth::ia::ActivationFunctions<float>;
    using Matrix = BitMth::linalg::Matrix<float>;

    Matrix emptyZ(0, 0);
    Matrix resSoftmax = Act::softmax(emptyZ);

    BIT_ASSERT_EQ(resSoftmax.getRows(), static_cast<size_t>(0));
    BIT_ASSERT_EQ(resSoftmax.getCols(), static_cast<size_t>(0));
}

BIT_TEST_CASE(ReluBatch) {
    using Act = BitMth::ia::ActivationFunctions<float>;
    using Matrix = BitMth::linalg::Matrix<float>;

    // Batch 2x3
    Matrix Z(2, 3);
    Z.getValues()[0] = -2.0f; Z.getValues()[1] =  0.0f; Z.getValues()[2] = 3.5f; // Fila 0
    Z.getValues()[3] =  1.5f; Z.getValues()[4] = -0.5f; Z.getValues()[5] = 0.0f; // Fila 1

    Matrix res = Act::relu(Z);

    BIT_ASSERT_EQ(res.getRows(), size_t(2));
    BIT_ASSERT_EQ(res.getCols(), size_t(3));

    // Fila 0
    BIT_EXPECT_TRUE(BitMth::utils::isClose(res.getValues()[0], 0.0f));
    BIT_EXPECT_TRUE(BitMth::utils::isClose(res.getValues()[1], 0.0f));
    BIT_EXPECT_TRUE(BitMth::utils::isClose(res.getValues()[2], 3.5f));

    // Fila 1
    BIT_EXPECT_TRUE(BitMth::utils::isClose(res.getValues()[3], 1.5f));
    BIT_EXPECT_TRUE(BitMth::utils::isClose(res.getValues()[4], 0.0f));
    BIT_EXPECT_TRUE(BitMth::utils::isClose(res.getValues()[5], 0.0f));
}

BIT_TEST_CASE(SigmoidBatch) {
    using Act = BitMth::ia::ActivationFunctions<double>;
    using Matrix = BitMth::linalg::Matrix<double>;

    Matrix Z(2, 2);
    Z.getValues()[0] =  0.0; Z.getValues()[1] = 2.0; // Fila 0
    Z.getValues()[2] = -2.0; Z.getValues()[3] = 0.0; // Fila 1

    Matrix res = Act::sigmoid(Z);

    BIT_ASSERT_EQ(res.getRows(), size_t(2));
    BIT_ASSERT_EQ(res.getCols(), size_t(2));

    // Fila 0: sigmoid(0) = 0.5, sigmoid(2) ≈ 0.880797
    BIT_EXPECT_TRUE(BitMth::utils::isClose(res.getValues()[0], 0.5));
    BIT_EXPECT_TRUE(BitMth::utils::isClose(res.getValues()[1], 0.880797));

    // Fila 1: sigmoid(-2) ≈ 0.119202, sigmoid(0) = 0.5
    BIT_EXPECT_TRUE(BitMth::utils::isClose(res.getValues()[2], 0.119202));
    BIT_EXPECT_TRUE(BitMth::utils::isClose(res.getValues()[3], 0.5));
}

BIT_TEST_CASE(TanhBatch) {
    using Act = BitMth::ia::ActivationFunctions<float>;
    using Matrix = BitMth::linalg::Matrix<float>;

    Matrix Z(2, 2);
    Z.getValues()[0] =  0.0f; Z.getValues()[1] = 1.0f;  // Fila 0
    Z.getValues()[2] = -1.0f; Z.getValues()[3] = 0.0f;  // Fila 1

    Matrix res = Act::tanh(Z);

    BIT_EXPECT_TRUE(BitMth::utils::isClose(res.getValues()[0],  0.0f));
    BIT_EXPECT_TRUE(BitMth::utils::isClose(res.getValues()[1],  0.761594f));
    BIT_EXPECT_TRUE(BitMth::utils::isClose(res.getValues()[2], -0.761594f));
    BIT_EXPECT_TRUE(BitMth::utils::isClose(res.getValues()[3],  0.0f));
}

BIT_TEST_CASE(SoftmaxBatch) {
    using Act = BitMth::ia::ActivationFunctions<float>;
    using Matrix = BitMth::linalg::Matrix<float>;

    // Batch de 2 muestras con 3 clases
    Matrix Z(2, 3);
    Z.getValues()[0] = 1.0f; Z.getValues()[1] = 2.0f; Z.getValues()[2] = 3.0f; // Fila 0
    Z.getValues()[3] = 0.0f; Z.getValues()[4] = 0.0f; Z.getValues()[5] = 0.0f; // Fila 1

    Matrix res = Act::softmax(Z);

    BIT_ASSERT_EQ(res.getRows(), size_t(2));
    BIT_ASSERT_EQ(res.getCols(), size_t(3));

    // Suma de probabilidades por fila debe ser igual a 1.0
    float sumRow0 = res.getValues()[0] + res.getValues()[1] + res.getValues()[2];
    float sumRow1 = res.getValues()[3] + res.getValues()[4] + res.getValues()[5];

    BIT_EXPECT_TRUE(BitMth::utils::isClose(sumRow0, 1.0f));
    BIT_EXPECT_TRUE(BitMth::utils::isClose(sumRow1, 1.0f));

    // Fila 1 (todos los logits iguales -> equiprobable 1/3)
    BIT_EXPECT_TRUE(BitMth::utils::isClose(res.getValues()[3], 1.0f / 3.0f));
    BIT_EXPECT_TRUE(BitMth::utils::isClose(res.getValues()[4], 1.0f / 3.0f));
    BIT_EXPECT_TRUE(BitMth::utils::isClose(res.getValues()[5], 1.0f / 3.0f));
}

BIT_TEST_CASE(SoftmaxDerivativeBatch) {
    using Act = BitMth::ia::ActivationFunctions<float>;
    using Matrix = BitMth::linalg::Matrix<float>;

    Matrix A(2, 2);
    A.getValues()[0] = 0.5f; A.getValues()[1] = 0.5f; // Fila 0
    A.getValues()[2] = 0.8f; A.getValues()[3] = 0.2f; // Fila 1

    Matrix grad(2, 2);
    grad.getValues()[0] = 1.0f; grad.getValues()[1] = 0.0f; // Fila 0
    grad.getValues()[2] = 0.0f; grad.getValues()[3] = 1.0f; // Fila 1

    Matrix dZ = Act::softmaxDerivative(A, grad);

    BIT_ASSERT_EQ(dZ.getRows(), size_t(2));
    BIT_ASSERT_EQ(dZ.getCols(), size_t(2));

    // Fila 0: dot = 0.5, dZ_00 = 0.5*(1 - 0.5) = 0.25, dZ_01 = 0.5*(0 - 0.5) = -0.25
    BIT_EXPECT_TRUE(BitMth::utils::isClose(dZ.getValues()[0],  0.25f));
    BIT_EXPECT_TRUE(BitMth::utils::isClose(dZ.getValues()[1], -0.25f));

    // Fila 1: dot = 0.2, dZ_10 = 0.8*(0 - 0.2) = -0.16, dZ_11 = 0.2*(1 - 0.2) = 0.16
    BIT_EXPECT_TRUE(BitMth::utils::isClose(dZ.getValues()[2], -0.16f));
    BIT_EXPECT_TRUE(BitMth::utils::isClose(dZ.getValues()[3],  0.16f));
}
BIT_TEST_CASE(ReluDerivative) {
    using Act = BitMth::ia::ActivationFunctions<float>;
    using Matrix = BitMth::linalg::Matrix<float>;

    Matrix Z(1, 3);
    Z.getValues()[0] = -1.5f;
    Z.getValues()[1] =  0.0f;
    Z.getValues()[2] =  2.0f;

    Matrix A = Act::relu(Z); // A = [0.0, 0.0, 2.0]
    
    // Gradiente de entrada simular dL/dA = [1.0, 1.0, 1.0]
    Matrix gradOut(1, 3);
    gradOut.getValues()[0] = 1.0f;
    gradOut.getValues()[1] = 1.0f;
    gradOut.getValues()[2] = 1.0f;

    Matrix dZ = Act::reluDerivative(A, gradOut);

    BIT_ASSERT_EQ(dZ.getRows(), static_cast<size_t>(1));
    BIT_ASSERT_EQ(dZ.getCols(), static_cast<size_t>(3));

    BIT_EXPECT_TRUE(BitMth::utils::isClose(dZ.getValues()[0], 0.0f));
    BIT_EXPECT_TRUE(BitMth::utils::isClose(dZ.getValues()[1], 0.0f));
    BIT_EXPECT_TRUE(BitMth::utils::isClose(dZ.getValues()[2], 1.0f));
}

BIT_TEST_CASE(SigmoidDerivative) {
    using Act = BitMth::ia::ActivationFunctions<double>;
    using Matrix = BitMth::linalg::Matrix<double>;

    Matrix Z(1, 1);
    Z.getValues()[0] = 0.0;

    Matrix A = Act::sigmoid(Z); // A = 0.5
    
    Matrix gradOut(1, 1);
    gradOut.getValues()[0] = 1.0;

    Matrix dZ = Act::sigmoidDerivative(A, gradOut); // gradOut * a * (1 - a) = 1.0 * 0.25 = 0.25

    BIT_EXPECT_TRUE(BitMth::utils::isClose(dZ.getValues()[0], 0.25));
}

BIT_TEST_CASE(TanhDerivative) {
    using Act = BitMth::ia::ActivationFunctions<float>;
    using Matrix = BitMth::linalg::Matrix<float>;

    Matrix Z(1, 1);
    Z.getValues()[0] = 0.0f;

    Matrix A = Act::tanh(Z); // A = 0.0
    
    Matrix gradOut(1, 1);
    gradOut.getValues()[0] = 1.0f;

    Matrix dZ = Act::tanhDerivative(A, gradOut); // gradOut * (1 - 0^2) = 1.0

    BIT_EXPECT_TRUE(BitMth::utils::isClose(dZ.getValues()[0], 1.0f));
}

BIT_TEST_CASE(ReluDerivativeBatch) {
    using Act = BitMth::ia::ActivationFunctions<float>;
    using Matrix = BitMth::linalg::Matrix<float>;

    Matrix Z(2, 3);
    Z.getValues()[0] = -5.0f; Z.getValues()[1] = 0.0f; Z.getValues()[2] = 3.0f; // Fila 0
    Z.getValues()[3] =  1.2f; Z.getValues()[4] = -0.1f; Z.getValues()[5] = 0.0f; // Fila 1

    Matrix A = Act::relu(Z);
    
    // Gradiente entrante con escalar 2.0f para verificar que el gradiente se propaga multiplicando
    Matrix gradOut(2, 3);
    for (size_t i = 0; i < 6; ++i) gradOut.getValues()[i] = 2.0f;

    Matrix dZ = Act::reluDerivative(A, gradOut);

    BIT_ASSERT_EQ(dZ.getRows(), size_t(2));
    BIT_ASSERT_EQ(dZ.getCols(), size_t(3));

    // Fila 0 (dZ = gradOut * (A > 0))
    BIT_EXPECT_TRUE(BitMth::utils::isClose(dZ.getValues()[0], 0.0f));
    BIT_EXPECT_TRUE(BitMth::utils::isClose(dZ.getValues()[1], 0.0f));
    BIT_EXPECT_TRUE(BitMth::utils::isClose(dZ.getValues()[2], 2.0f)); // 2.0 * 1.0

    // Fila 1
    BIT_EXPECT_TRUE(BitMth::utils::isClose(dZ.getValues()[3], 2.0f)); // 2.0 * 1.0
    BIT_EXPECT_TRUE(BitMth::utils::isClose(dZ.getValues()[4], 0.0f));
    BIT_EXPECT_TRUE(BitMth::utils::isClose(dZ.getValues()[5], 0.0f));
}

BIT_TEST_CASE(SigmoidDerivativeBatch) {
    using Act = BitMth::ia::ActivationFunctions<double>;
    using Matrix = BitMth::linalg::Matrix<double>;

    Matrix A(2, 2);
    A.getValues()[0] = 0.5; A.getValues()[1] = 1.0; // Fila 0
    A.getValues()[2] = 0.0; A.getValues()[3] = 0.2; // Fila 1

    Matrix gradOut(2, 2);
    for (size_t i = 0; i < 4; ++i) gradOut.getValues()[i] = 1.0;

    Matrix dZ = Act::sigmoidDerivative(A, gradOut);

    // Formula: gradOut * A * (1 - A)
    BIT_EXPECT_TRUE(BitMth::utils::isClose(dZ.getValues()[0], 0.25)); // 1.0 * 0.5 * 0.5
    BIT_EXPECT_TRUE(BitMth::utils::isClose(dZ.getValues()[1], 0.00)); // 1.0 * 1.0 * 0.0
    BIT_EXPECT_TRUE(BitMth::utils::isClose(dZ.getValues()[2], 0.00)); // 1.0 * 0.0 * 1.0
    BIT_EXPECT_TRUE(BitMth::utils::isClose(dZ.getValues()[3], 0.16)); // 1.0 * 0.2 * 0.8
}

BIT_TEST_CASE(TanhDerivativeBatch) {
    using Act = BitMth::ia::ActivationFunctions<float>;
    using Matrix = BitMth::linalg::Matrix<float>;

    Matrix A(2, 2);
    A.getValues()[0] = 0.0f; A.getValues()[1] = 0.5f;  // Fila 0
    A.getValues()[2] = 1.0f; A.getValues()[3] = -0.5f; // Fila 1

    Matrix gradOut(2, 2);
    for (size_t i = 0; i < 4; ++i) gradOut.getValues()[i] = 1.0f;

    Matrix dZ = Act::tanhDerivative(A, gradOut);

    // Formula: gradOut * (1 - A^2)
    BIT_EXPECT_TRUE(BitMth::utils::isClose(dZ.getValues()[0], 1.00f)); // 1.0 * (1 - 0)
    BIT_EXPECT_TRUE(BitMth::utils::isClose(dZ.getValues()[1], 0.75f)); // 1.0 * (1 - 0.25)
    BIT_EXPECT_TRUE(BitMth::utils::isClose(dZ.getValues()[2], 0.00f)); // 1.0 * (1 - 1)
    BIT_EXPECT_TRUE(BitMth::utils::isClose(dZ.getValues()[3], 0.75f)); // 1.0 * (1 - 0.25)
}

BIT_GROUP_END()
