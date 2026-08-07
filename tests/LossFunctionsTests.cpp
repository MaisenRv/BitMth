#include <BitTest/BitTest.hpp>
#include <BitMth/ia/LossFunctions.hpp>
#include <BitMth/utils/MathUtils.hpp>

BIT_GROUP_BEGIN(lossFunctions)

BIT_TEST_CASE(MSELossBatch) {
    using Loss = BitMth::ia::LossFunctions<float>;
    using Matrix = BitMth::linalg::Matrix<float>;

    // Matriz 2x2 (4 elementos en total)
    Matrix predict(2, 2);
    predict.getValues()[0] = 1.0f; predict.getValues()[1] = 2.0f; // Fila 0
    predict.getValues()[2] = 3.0f; predict.getValues()[3] = 4.0f; // Fila 1

    Matrix real(2, 2);
    real.getValues()[0] = 1.0f; real.getValues()[1] = 0.0f; // Fila 0
    real.getValues()[2] = 1.0f; real.getValues()[3] = 2.0f; // Fila 1

    // Diferencias al cuadrado: [0, 4, 4, 4] -> Suma = 12 -> N = 4 -> MSE = 12 / 4 = 3.0
    Matrix loss = Loss::mse(predict, real);

    // MSE retorna una matriz escalar 1x1
    BIT_ASSERT_EQ(loss.getRows(), static_cast<size_t>(1));
    BIT_ASSERT_EQ(loss.getCols(), static_cast<size_t>(1));
    BIT_EXPECT_TRUE(BitMth::utils::isClose(loss.getValues()[0], 3.0f));
}

BIT_TEST_CASE(MSEDerivativeBatch) {
    using Loss = BitMth::ia::LossFunctions<float>;
    using Matrix = BitMth::linalg::Matrix<float>;

    Matrix predict(2, 2);
    predict.getValues()[0] = 1.0f; predict.getValues()[1] = 2.0f;
    predict.getValues()[2] = 3.0f; predict.getValues()[3] = 4.0f;

    Matrix real(2, 2);
    real.getValues()[0] = 1.0f; real.getValues()[1] = 0.0f;
    real.getValues()[2] = 1.0f; real.getValues()[3] = 2.0f;

    // dMSE = (pred - real) / N  donde N = 4 (invN = 0.25)
    // Diff: [0, 2, 2, 2] * 0.25 -> [0.0, 0.5, 0.5, 0.5]
    Matrix dLoss = Loss::mseDerivative(predict, real);

    BIT_ASSERT_EQ(dLoss.getRows(), static_cast<size_t>(2));
    BIT_ASSERT_EQ(dLoss.getCols(), static_cast<size_t>(2));

    BIT_EXPECT_TRUE(BitMth::utils::isClose(dLoss.getValues()[0], 0.0f));
    BIT_EXPECT_TRUE(BitMth::utils::isClose(dLoss.getValues()[1], 0.5f));
    BIT_EXPECT_TRUE(BitMth::utils::isClose(dLoss.getValues()[2], 0.5f));
    BIT_EXPECT_TRUE(BitMth::utils::isClose(dLoss.getValues()[3], 0.5f));
}

BIT_TEST_CASE(BCELossBatch) {
    using Loss = BitMth::ia::LossFunctions<double>;
    using Matrix = BitMth::linalg::Matrix<double>;

    Matrix predict(1, 2);
    predict.getValues()[0] = 0.8; predict.getValues()[1] = 0.2;

    Matrix real(1, 2);
    real.getValues()[0] = 1.0; real.getValues()[1] = 0.0;

    // Elemento 0: 1 * log(0.8) ≈ -0.22314
    // Elemento 1: 1 * log(1 - 0.2) = log(0.8) ≈ -0.22314
    // Suma = -0.44628 -> Loss = -(-0.44628) / 2 = 0.22314
    Matrix loss = Loss::bce(predict, real);

    BIT_ASSERT_EQ(loss.getRows(), static_cast<size_t>(1));
    BIT_ASSERT_EQ(loss.getCols(), static_cast<size_t>(1));
    double expectedLoss = -std::log(0.8); 
    BIT_EXPECT_TRUE(BitMth::utils::isClose(loss.getValues()[0], expectedLoss));
}

BIT_TEST_CASE(BCEDerivativeBatch) {
    using Loss = BitMth::ia::LossFunctions<double>;
    using Matrix = BitMth::linalg::Matrix<double>;

    Matrix predict(1, 2);
    predict.getValues()[0] = 0.8; predict.getValues()[1] = 0.2;

    Matrix real(1, 2);
    real.getValues()[0] = 1.0; real.getValues()[1] = 0.0;

    // Formula: (y_pred - y) / (y_pred * (1 - y_pred)) * (1 / N)
    // N = 2 -> 1/N = 0.5
    // Elem 0: (0.8 - 1.0) / (0.8 * 0.2) * 0.5 = -0.2 / 0.16 * 0.5 = -0.625
    // Elem 1: (0.2 - 0.0) / (0.2 * 0.8) * 0.5 =  0.2 / 0.16 * 0.5 =  0.625
    Matrix dLoss = Loss::bceDerivative(predict, real);

    BIT_ASSERT_EQ(dLoss.getRows(), static_cast<size_t>(1));
    BIT_ASSERT_EQ(dLoss.getCols(), static_cast<size_t>(2));

    BIT_EXPECT_TRUE(BitMth::utils::isClose(dLoss.getValues()[0], -0.625));
    BIT_EXPECT_TRUE(BitMth::utils::isClose(dLoss.getValues()[1],  0.625));
}
BIT_TEST_CASE(BCEClampingNumericalStability) {
    using Loss = BitMth::ia::LossFunctions<float>;
    using Matrix = BitMth::linalg::Matrix<float>;

    // Predicciones extremas que causarian log(0) o division por cero sin std::clamp
    Matrix predict(1, 2);
    predict.getValues()[0] = 0.0f; 
    predict.getValues()[1] = 1.0f;

    Matrix real(1, 2);
    real.getValues()[0] = 1.0f; 
    real.getValues()[1] = 0.0f;

    Matrix loss = Loss::bce(predict, real);
    Matrix dLoss = Loss::bceDerivative(predict, real);

    // Verificaciones limpias con el método de Matrix
    BIT_EXPECT_FALSE(loss.hasNaN());
    BIT_EXPECT_FALSE(dLoss.hasNaN());
}

BIT_GROUP_END()
