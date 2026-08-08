#include <BitTest/BitTest.hpp>
#include <BitMth/linalg/Matrix.hpp>

BIT_GROUP_BEGIN(matrix)
BIT_TEST_CASE(MatrixInitializationDefault){
    BitMth::linalg::Matrix<float> m;
    BIT_ASSERT_EQ(0, m.getRows());
    BIT_ASSERT_EQ(0, m.getCols());
    BIT_ASSERT_EQ(0, m.size());
    BIT_ASSERT_EQ(0, m.getStrides()[0]);
    BIT_ASSERT_EQ(0, m.getStrides()[1]);
    BIT_ASSERT_EQ(nullptr, m.getArena());
    BIT_ASSERT_EQ(nullptr, m.getValues());
    BIT_ASSERT_EQ(0, m.getRowJump());
    BIT_ASSERT_EQ(0, m.getColJump());
}

BIT_TEST_CASE(MatrixInitialization){
    BitMth::linalg::Matrix<double> m(2,3,nullptr);
    size_t size = sizeof(double);
    BIT_ASSERT_EQ(2, m.getRows());
    BIT_ASSERT_EQ(3, m.getCols());
    BIT_ASSERT_EQ(6, m.size());
    BIT_ASSERT_EQ(m.getCols() * size, m.getStrides()[0]);
    BIT_ASSERT_EQ(size, m.getStrides()[1]);
    BIT_ASSERT_EQ(nullptr, m.getArena());
    BIT_ASSERT_EQ(m.getStrides()[0] / size, m.getRowJump());
    BIT_ASSERT_EQ(m.getStrides()[1] / size, m.getColJump());
}

BIT_TEST_CASE(MatrixInitializationZero){
    BitMth::linalg::Matrix<float> m(2,3,nullptr,true);
    BitMth::linalg::Matrix<float> m2(2,3,nullptr,true);
    m2.setWith(0);
    size_t size = sizeof(float);
    BIT_ASSERT_EQ(2, m.getRows());
    BIT_ASSERT_EQ(3, m.getCols());
    BIT_ASSERT_EQ(6, m.size());
    BIT_ASSERT_EQ(nullptr, m.getArena());
    BIT_ASSERT_EQ(m.getStrides()[0] / size, m.getRowJump());
    BIT_ASSERT_EQ(m.getStrides()[1] / size, m.getColJump());
    BIT_ASSERT_EQ(m, m2);
}

// COPY

BIT_TEST_CASE(MatrixCopyConstructor){
    BitMth::linalg::Matrix<float> m(2,3,nullptr,true);
    size_t size = sizeof(float);
    m(1,2) = 78;
    m(0,1) = 754;
    m(1,0) = 342;
    BitMth::linalg::Matrix<float> m2(m);
    BIT_ASSERT_EQ(m,m2);
    BIT_ASSERT_EQ(nullptr, m2.getArena());
    BIT_ASSERT_EQ(m.getStrides()[0] / size, m2.getRowJump());
    BIT_ASSERT_EQ(m.getStrides()[1] / size, m2.getColJump());
    m2(1,2) = 999;
    BIT_ASSERT(m(1,2) != m2(1,2));
}

BIT_TEST_CASE(MatrixCloneFunct){
    BitMth::linalg::Matrix<double> m(2,3,nullptr,true);
    size_t size = sizeof(double);
    m(1,2) = 28;
    m(0,1) = 7544;
    m(1,0) = 392;
    BitMth::linalg::Matrix<double> m2 = m.clone(nullptr);
    BIT_ASSERT_EQ(m.getStrides()[0] / size, m2.getRowJump());
    BIT_ASSERT_EQ(m.getStrides()[1] / size, m2.getColJump());
    BIT_ASSERT_EQ(m,m2);
    BIT_ASSERT_EQ(nullptr, m2.getArena());
    m2(1,2) = 999;
    BIT_ASSERT(m(1,2) != m2(1,2));
}

BIT_TEST_CASE(MatrixMoveConstructor) {
    BitMth::linalg::Matrix<double> m1(2, 3, nullptr, true);
    size_t size = sizeof(double);
    m1(0, 0) = 5.5;
    m1(1, 2) = 10.5;
    const double* valuesPtr = m1.getValues(); 
    BitMth::linalg::Matrix<double> m2(std::move(m1));
    BIT_ASSERT_EQ(2, m2.getRows());
    BIT_ASSERT_EQ(3, m2.getCols());
    BIT_ASSERT_EQ(6, m2.size());
    BIT_ASSERT_EQ(5.5, m2(0, 0));
    BIT_ASSERT_EQ(10.5, m2(1, 2));
    BIT_ASSERT_EQ(valuesPtr, m2.getValues());
    BIT_ASSERT_EQ(m2.getStrides()[0] / size, m2.getRowJump());
    BIT_ASSERT_EQ(m2.getStrides()[1] / size, m2.getColJump());
    BIT_ASSERT_EQ(0, m1.getRows());
    BIT_ASSERT_EQ(0, m1.getCols());
    BIT_ASSERT_EQ(0, m1.size());
    BIT_ASSERT_EQ(0, m1.getStrides()[0]);
    BIT_ASSERT_EQ(0, m1.getStrides()[1]);
    BIT_ASSERT_EQ(nullptr, m1.getArena());
    BIT_ASSERT_EQ(nullptr, m1.getValues());
}

BIT_TEST_CASE(MatrixMoveAssignment) {
    BitMth::linalg::Matrix<double> m1(2, 3, nullptr, true);
    size_t size = sizeof(double);
    m1(0, 0) = 111.0;
    m1(1, 1) = 222.0;
    const double* valuesPtr = m1.getValues();
    BitMth::linalg::Matrix<double> m2(5, 5, nullptr, true);
    m2 = std::move(m1);
    BIT_ASSERT_EQ(2, m2.getRows());
    BIT_ASSERT_EQ(3, m2.getCols());
    BIT_ASSERT_EQ(6, m2.size());
    BIT_ASSERT_EQ(111.0, m2(0, 0) );
    BIT_ASSERT_EQ(222.0, m2(1, 1));
    BIT_ASSERT_EQ(valuesPtr, m2.getValues());
    BIT_ASSERT_EQ(m2.getStrides()[0] / size, m2.getRowJump());
    BIT_ASSERT_EQ(m2.getStrides()[1] / size, m2.getColJump());
    BIT_ASSERT_EQ(0, m1.getRows());
    BIT_ASSERT_EQ(0, m1.getCols());
    BIT_ASSERT_EQ(0, m1.size());
    BIT_ASSERT_EQ(0, m1.getStrides()[0]);
    BIT_ASSERT_EQ(0, m1.getStrides()[1]);
    BIT_ASSERT_EQ(nullptr, m1.getArena());
    BIT_ASSERT_EQ(nullptr, m1.getValues());
}

BIT_TEST_CASE(MatrixCopyAssignmentWithHeap) {
    BitMth::linalg::Matrix<double> src(2, 3, nullptr, true);
    size_t size = sizeof(double);
    src(0, 0) = 1.1;
    src(0, 1) = 2.2;
    src(1, 2) = 3.3;
    BitMth::core::Arena fakeArena(18 * sizeof(double));
    BitMth::linalg::Matrix<double> dest(4, 4, &fakeArena, false);
    dest = src;
    BIT_ASSERT_EQ(2, dest.getRows());
    BIT_ASSERT_EQ(3, dest.getCols());
    BIT_ASSERT_EQ(6, dest.size());
    BIT_ASSERT_EQ(nullptr, dest.getArena());
    BIT_ASSERT_EQ(dest.getStrides()[0] / size, dest.getRowJump());
    BIT_ASSERT_EQ(dest.getStrides()[1] / size, dest.getColJump());
    BIT_ASSERT_EQ(1.1, dest(0, 0));
    BIT_ASSERT_EQ(2.2, dest(0, 1));
    BIT_ASSERT_EQ(3.3, dest(1, 2));
    dest = dest;
    BIT_ASSERT_EQ(3.3, dest(1, 2));
}

BIT_TEST_CASE(MatrixOperatorIndexing) {
    BitMth::linalg::Matrix<double> m(2, 2, nullptr, true);
    m(0, 0) = 10.5;
    m(0, 1) = 20.5;
    m(1, 0) = 30.5;
    m(1, 1) = 40.5;
    BIT_ASSERT_EQ(10.5, m(0, 0));
    BIT_ASSERT_EQ(20.5, m(0, 1));
    BIT_ASSERT_EQ(30.5, m(1, 0));
    BIT_ASSERT_EQ(40.5, m(1, 1));
    const BitMth::linalg::Matrix<double>& constMatrix = m;
    BIT_ASSERT_EQ(10.5, constMatrix(0, 0));
    BIT_ASSERT_EQ(40.5, constMatrix(1, 1));
}

// INDEXING

BIT_TEST_CASE(MatrixIndexingOutOfBounds) {
    BitMth::linalg::Matrix<double> m(2, 3, nullptr, true);
    BIT_ASSERT_THROWS(m(2, 0));
    BIT_ASSERT_THROWS(m(0, 3));
}

// OPERATORS MATRIX - SCALAR

BIT_TEST_CASE(AdditionAssignmentOperator){
    BitMth::linalg::Matrix<int> m(3,2,nullptr,true);
    m.setWith(5);
    m += 3;
    BitMth::linalg::Matrix<int> m2(3,2,nullptr, true);
    m2.setWith(8);
    BIT_ASSERT_EQ(m2, m);
}

BIT_TEST_CASE(AdditionOperator){
    BitMth::linalg::Matrix<int> m(3,2,nullptr,true);
    m.setWith(5);
    BitMth::linalg::Matrix<int> m2 = m + 3;
    BitMth::linalg::Matrix<int> m3(3,2,nullptr,true);
    m3.setWith(8);
    BIT_ASSERT_EQ(m3, m2);
    BIT_ASSERT_EQ(nullptr, m2.getArena());
}

BIT_TEST_CASE(AddStaticFunct){
    BitMth::core::Arena arena(100);
    BitMth::linalg::Matrix<int> m(3,2,nullptr,true);
    m.setWith(5);
    BitMth::linalg::Matrix<int> m2 = BitMth::linalg::Matrix<int>::add(m,3, &arena);
    BitMth::linalg::Matrix<int> m3(3,2,nullptr,true);
    m3.setWith(8);
    BIT_ASSERT_EQ(m3, m2);
    BIT_ASSERT_EQ(&arena, m2.getArena());
    BIT_ASSERT_EQ(nullptr, m3.getArena());
}

BIT_TEST_CASE(SubtractionAssignmentOperator){
    BitMth::linalg::Matrix<int> m(3,2,nullptr,true);
    m.setWith(5);
    m -= 3;
    BitMth::linalg::Matrix<int> m2(3,2,nullptr, true);
    m2.setWith(2);
    BIT_ASSERT_EQ(m2, m);
}

BIT_TEST_CASE(SubtractionOperator){
    BitMth::linalg::Matrix<int> m(3,2,nullptr,true);
    m.setWith(5);
    BitMth::linalg::Matrix<int> m2 = m - 3;
    BitMth::linalg::Matrix<int> m3(3,2,nullptr,true);
    m3.setWith(2);
    BIT_ASSERT_EQ(m3, m2);
    BIT_ASSERT_EQ(nullptr, m2.getArena());
}

BIT_TEST_CASE(SubtractionOperatorFriend){
    BitMth::linalg::Matrix<int> m(3,2,nullptr,true);
    m.setWith(5);
    BitMth::linalg::Matrix<int> m2 = 3 - m;
    BitMth::linalg::Matrix<int> m3(3,2,nullptr,true);
    m3.setWith(-2);
    BIT_ASSERT_EQ(m3, m2);
    BIT_ASSERT_EQ(nullptr, m2.getArena());
}

BIT_TEST_CASE(SubStaticFunct1){
    BitMth::core::Arena arena(100);
    BitMth::linalg::Matrix<int> m(3,2,nullptr,true);
    m.setWith(5);
    BitMth::linalg::Matrix<int> m2 = BitMth::linalg::Matrix<int>::sub(m,3, &arena);
    BitMth::linalg::Matrix<int> m3(3,2,nullptr,true);
    m3.setWith(2);
    BIT_ASSERT_EQ(m3, m2);
    BIT_ASSERT_EQ(&arena, m2.getArena());
    BIT_ASSERT_EQ(nullptr, m3.getArena());
}

BIT_TEST_CASE(SubStaticFunct2){
    BitMth::core::Arena arena(100);
    BitMth::linalg::Matrix<int> m(3,2,nullptr,true);
    m.setWith(5);
    BitMth::linalg::Matrix<int> m2 = BitMth::linalg::Matrix<int>::sub(3,m, &arena);
    BitMth::linalg::Matrix<int> m3(3,2,nullptr,true);
    m3.setWith(-2);
    BIT_ASSERT_EQ(m3, m2);
    BIT_ASSERT_EQ(&arena, m2.getArena());
    BIT_ASSERT_EQ(nullptr, m3.getArena());
}

BIT_TEST_CASE(MultiplicationAssignmentOperator){
    BitMth::linalg::Matrix<int> m(3,4,nullptr,true);
    m.setWith(5);
    m *= 3;
    BitMth::linalg::Matrix<int> m2(3,4,nullptr, true);
    m2.setWith(15);
    BIT_ASSERT_EQ(m2, m);
}

BIT_TEST_CASE(MultiplicationOperator){
    BitMth::linalg::Matrix<int> m(3,2,nullptr,true);
    m.setWith(5);
    BitMth::linalg::Matrix<int> m2 = m * 3;
    BitMth::linalg::Matrix<int> m3(3,2,nullptr,true);
    m3.setWith(15);
    BIT_ASSERT_EQ(m3, m2);
    BIT_ASSERT_EQ(nullptr, m2.getArena());
}

BIT_TEST_CASE(MulStaticFunct){
    BitMth::core::Arena arena(100);
    BitMth::linalg::Matrix<int> m(3,2,nullptr,true);
    m.setWith(5);
    BitMth::linalg::Matrix<int> m2 = BitMth::linalg::Matrix<int>::mul(m,3, &arena);
    BitMth::linalg::Matrix<int> m3(3,2,nullptr,true);
    m3.setWith(15);
    BIT_ASSERT_EQ(m3, m2);
    BIT_ASSERT_EQ(&arena, m2.getArena());
    BIT_ASSERT_EQ(nullptr, m3.getArena());
}

BIT_TEST_CASE(DivisionAssignmentOperator){
    BitMth::linalg::Matrix<float> m(3,4,nullptr,true);
    m.setWith(15);
    m /= 3;
    BitMth::linalg::Matrix<float> m2(3,4,nullptr, true);
    m2.setWith(5);
    BIT_EXPECT_TRUE(m.isApprox(m2));
}

BIT_TEST_CASE(DivisionOperator){
    BitMth::linalg::Matrix<float> m(3,2,nullptr,true);
    m.setWith(15);
    BitMth::linalg::Matrix<float> m2 = m / 3;
    BitMth::linalg::Matrix<float> m3(3,2,nullptr,true);
    m3.setWith(5);
    BIT_EXPECT_TRUE(m2.isApprox(m3));
    BIT_ASSERT_EQ(nullptr, m2.getArena());
}

BIT_TEST_CASE(divStaticFunct){
    BitMth::core::Arena arena(100);
    BitMth::linalg::Matrix<float> m(3,2,nullptr,true);
    m.setWith(15);
    BitMth::linalg::Matrix<float> m2 = BitMth::linalg::Matrix<float>::div(m,3, &arena);
    BitMth::linalg::Matrix<float> m3(3,2,nullptr,true);
    m3.setWith(5);
    BIT_EXPECT_TRUE(m2.isApprox(m3));
    BIT_ASSERT_EQ(&arena, m2.getArena());
    BIT_ASSERT_EQ(nullptr, m3.getArena());
}

BIT_TEST_CASE(DivisionByZero){
    BitMth::linalg::Matrix<float> m(3,4,nullptr,true);
    m.setWith(15);
    BIT_ASSERT_THROWS(m /= 0);
    BIT_ASSERT_THROWS(m / 0);
    BIT_ASSERT_THROWS(BitMth::linalg::Matrix<float>::div(m,0,nullptr));
}

BIT_TEST_CASE(Pow) {
    BitMth::linalg::Matrix<float> m(3, 2, nullptr, true);
    m.setWith(3.0f);
    BitMth::linalg::Matrix<float> m2 = m.pow(2.0f);
    BitMth::linalg::Matrix<float> result(3, 2, nullptr, true);
    result.setWith(9.0f);
    BIT_EXPECT_TRUE(m2.isApprox(result));
    BitMth::linalg::Matrix<float> original(3, 2, nullptr, true);
    original.setWith(3.0f);
    BIT_EXPECT_TRUE(m.isApprox(original));
    BitMth::linalg::Matrix<float> m3 = m2.pow(1.5f);
    BitMth::linalg::Matrix<float> result2(3, 2, nullptr, true);
    result2.setWith(27.0f);
    BIT_EXPECT_TRUE(m3.isApprox(result2, 1e-5f));
    BIT_EXPECT_TRUE(m2.isApprox(result));
    BitMth::linalg::Matrix<float> m_zero = m.pow(0.0f);
    BitMth::linalg::Matrix<float> one(3, 2, nullptr, true);
    one.setOne();
    BIT_ASSERT_EQ(one, m_zero);
    BIT_EXPECT_TRUE(m.isApprox(original));
}


BIT_TEST_CASE(PowInPlace){
    BitMth::linalg::Matrix<float> m(3,2,nullptr,true);
    m.setWith(3);
    m.powInPlace(2);
    BitMth::linalg::Matrix<float> m2(3,2,nullptr,true);
    m2.setWith(9);
    BIT_EXPECT_TRUE(m.isApprox(m2));
    m.powInPlace(1.5);
    m2.setWith(27);
    BIT_EXPECT_TRUE(m2.isApprox(m, 1e-5));
    m.powInPlace(0);
    m2.setOne();
    BIT_ASSERT_EQ(m2,m);
}

// OPERATORS MATRIX - MATRIX

BIT_TEST_CASE(MatrixAdditionSuccess) {
    BitMth::linalg::Matrix<float> A(2, 3, nullptr, true);
    BitMth::linalg::Matrix<float> B(2, 3, nullptr, true);
    // A: [ 1, 2, 3 ]      B: [ 10, 20, 30 ]
    //    [ 4, 5, 6 ]         [ 40, 50, 60 ]
    float valA = 1.0f;
    float valB = 10.0f;
    for (size_t i = 0; i < A.size(); ++i) {
        A.getValues()[i] = valA++;
        B.getValues()[i] = valB;
        valB += 10.0f;
    }

    // result  : [ 11, 22, 33 ]
    //           [ 44, 55, 66 ]
    BitMth::linalg::Matrix<float> result(2, 3, nullptr, true);
    float valEsp = 11.0f;
    for (size_t i = 0; i < result.size(); ++i) {
        result.getValues()[i] = valEsp;
        valEsp += 11.0f;
    }
    BitMth::linalg::Matrix<float> C = A + B;
    BIT_EXPECT_TRUE(C.isApprox(result));
    BitMth::linalg::Matrix<float> A_copy(2, 3, nullptr, true);
    float valACopia = 1.0f;
    for (size_t i = 0; i < A_copy.size(); ++i) A_copy.getValues()[i] = valACopia++;
    BIT_EXPECT_TRUE(A.isApprox(A_copy));
    BitMth::linalg::Matrix<float> D = BitMth::linalg::Matrix<float>::add(A, B, nullptr);
    BIT_EXPECT_TRUE(D.isApprox(result));
    A += B;
    BIT_EXPECT_TRUE(A.isApprox(result));
}

BIT_TEST_CASE(MatrixAdditionDimensionsMismatch) {
    BitMth::linalg::Matrix<float> A(2, 3, nullptr, true);
    BitMth::linalg::Matrix<float> B(3, 2, nullptr, true);
    A.setWith(1.0f);
    B.setWith(2.0f);
    BIT_ASSERT_THROWS(A += B);
    BIT_ASSERT_THROWS(A + B);
    BIT_ASSERT_THROWS(BitMth::linalg::Matrix<float>::add(A, B, nullptr));
}

BIT_TEST_CASE(MatrixSubtractionSuccess) {
    BitMth::linalg::Matrix<float> A(2, 3, nullptr, true);
    BitMth::linalg::Matrix<float> B(2, 3, nullptr, true);
    // A: [ 1, 2, 3 ]      B: [ 10, 20, 30 ]
    //    [ 4, 5, 6 ]         [ 40, 50, 60 ]
    float valA = 1.0f;
    float valB = 10.0f;
    for (size_t i = 0; i < A.size(); ++i) {
        A.getValues()[i] = valA++;
        B.getValues()[i] = valB;
        valB += 10.0f;
    }
    // [ 10 - 1, 20 - 2, 30 - 3 ]  =>  [ 9, 18, 27 ]
    // [ 40 - 4, 50 - 5, 60 - 6 ]  =>  [ 36, 45, 54 ]
    BitMth::linalg::Matrix<float> result(2, 3, nullptr, true);
    float valEsp = 9.0f;
    for (size_t i = 0; i < result.size(); ++i) {
        result.getValues()[i] = valEsp;
        valEsp += 9.0f;
    }
    BitMth::linalg::Matrix<float> C = B - A;
    BIT_EXPECT_TRUE(C.isApprox(result));
    BitMth::linalg::Matrix<float> B_copy(2, 3, nullptr, true);
    float valBCopia = 10.0f;
    for (size_t i = 0; i < B_copy.size(); ++i) {
        B_copy.getValues()[i] = valBCopia;
        valBCopia += 10.0f;
    }
    BIT_EXPECT_TRUE(B.isApprox(B_copy));
    BitMth::linalg::Matrix<float> D = BitMth::linalg::Matrix<float>::sub(B, A, nullptr);
    BIT_EXPECT_TRUE(D.isApprox(result));
    B -= A;
    BIT_EXPECT_TRUE(B.isApprox(result));
}

BIT_TEST_CASE(MatrixSubtractionDimensionsMismatch) {
    BitMth::linalg::Matrix<float> A(2, 3, nullptr, true);
    BitMth::linalg::Matrix<float> B(3, 2, nullptr, true);
    A.setWith(1.0f);
    B.setWith(2.0f);
    BIT_ASSERT_THROWS(A -= B);
    BIT_ASSERT_THROWS(A - B);
    BIT_ASSERT_THROWS(BitMth::linalg::Matrix<float>::sub(A, B, nullptr));
}

BIT_TEST_CASE(MatrixMultiplicationSuccess) {
    // [ 1, 2, 3 ]
    // [ 4, 5, 6 ]
    BitMth::linalg::Matrix<float> A(2, 3, nullptr, true);
    float valA = 1.0f;
    for (size_t i = 0; i < A.size(); ++i) {
        A.getValues()[i] = valA++;
    }
    // [ 7,  8 ]
    // [ 9,  10 ]
    // [ 11, 12 ]
    BitMth::linalg::Matrix<float> B(3, 2, nullptr, true);
    float valB = 7.0f;
    for (size_t i = 0; i < B.size(); ++i) {
        B.getValues()[i] = valB++;
    }
    // Row 0: [ (1*7 + 2*9 + 3*11), (1*8 + 2*10 + 3*12) ] => [ 58, 64 ]
    // Row 1: [ (4*7 + 5*9 + 6*11), (4*8 + 5*10 + 6*12) ] => [ 139, 154 ]
    BitMth::linalg::Matrix<float> esperadaMul(2, 2, nullptr, true);
    esperadaMul.getValues()[0] = 58.0f;
    esperadaMul.getValues()[1] = 64.0f;
    esperadaMul.getValues()[2] = 139.0f;
    esperadaMul.getValues()[3] = 154.0f;
    BitMth::linalg::Matrix<float> C = A * B;
    BIT_ASSERT_EQ(C.getRows(), 2);
    BIT_ASSERT_EQ(C.getCols(), 2);
    BIT_EXPECT_TRUE(C.isApprox(esperadaMul));

    BitMth::linalg::Matrix<float> A_copia(2, 3, nullptr, true);
    float valACopia = 1.0f;
    for (size_t i = 0; i < A_copia.size(); ++i) A_copia.getValues()[i] = valACopia++;
    BIT_EXPECT_TRUE(A.isApprox(A_copia));

    BitMth::linalg::Matrix<float> D = BitMth::linalg::Matrix<float>::mul(A, B, nullptr);
    BIT_EXPECT_TRUE(D.isApprox(esperadaMul));

    A *= B;
    BIT_ASSERT_EQ(A.getRows(), 2);
    BIT_ASSERT_EQ(A.getCols(), 2);
    BIT_EXPECT_TRUE(A.isApprox(esperadaMul));
}

BIT_TEST_CASE(MatrixMultiplicationMismatch) {
    BitMth::linalg::Matrix<float> A(2, 3, nullptr, true);
    BitMth::linalg::Matrix<float> B(2, 3, nullptr, true);
    A.setWith(1.0f);
    B.setWith(2.0f);
    BIT_ASSERT_THROWS(A * B);
    BIT_ASSERT_THROWS(A *= B);
    BIT_ASSERT_THROWS(BitMth::linalg::Matrix<float>::mul(A, B, nullptr));
}

BIT_TEST_CASE(HadamardProductSuccess) {
    // A: [ 1, 2, 3 ]
    //    [ 4, 5, 6 ]
    BitMth::linalg::Matrix<float> A(2, 3, nullptr, true);
    float valA = 1.0f;
    for (size_t i = 0; i < A.size(); ++i) {
        A.getValues()[i] = valA++;
    }

    // B: [ 2, 2, 2 ]
    //    [ 2, 2, 2 ]
    BitMth::linalg::Matrix<float> B(2, 3, nullptr, true);
    B.setWith(2.0f);

    // Esperada: [ 2, 4, 6 ]
    //           [ 8, 10, 12 ]
    BitMth::linalg::Matrix<float> esperadaHadamard(2, 3, nullptr, true);
    float valEsp = 2.0f;
    for (size_t i = 0; i < esperadaHadamard.size(); ++i) {
        esperadaHadamard.getValues()[i] = valEsp;
        valEsp += 2.0f;
    }

    BitMth::linalg::Matrix<float> C = BitMth::linalg::Matrix<float>::hadamard(A, B, nullptr);
    BIT_EXPECT_TRUE(C.isApprox(esperadaHadamard));
    BitMth::linalg::Matrix<float> A_copia(2, 3, nullptr, true);
    float valACopia = 1.0f;
    for (size_t i = 0; i < A_copia.size(); ++i) A_copia.getValues()[i] = valACopia++;
    BIT_EXPECT_TRUE(A.isApprox(A_copia));

    A.hadamardInPlace(B);
    BIT_EXPECT_TRUE(A.isApprox(esperadaHadamard));
}

BIT_TEST_CASE(HadamardProductMismatch) {
    BitMth::linalg::Matrix<float> A(2, 3, nullptr, true);
    BitMth::linalg::Matrix<float> B(3, 2, nullptr, true);
    A.setWith(1.0f);
    B.setWith(2.0f);

    BIT_ASSERT_THROWS(A.hadamardInPlace(B));
    BIT_ASSERT_THROWS(BitMth::linalg::Matrix<float>::hadamard(A, B, nullptr));
}

BIT_TEST_CASE(MatrixTransposeSuccess) {
    // [ 1, 2, 3 ]
    // [ 4, 5, 6 ]
    BitMth::linalg::Matrix<float> A(2, 3, nullptr, true);
    float val = 1.0f;
    for (size_t i = 0; i < A.size(); ++i) {
        A.getValues()[i] = val++;
    }
    const size_t origStride0 = A.getStrides()[0];
    const size_t origStride1 = A.getStrides()[1];
    A.tInPlace();
    BIT_ASSERT_EQ(A.getRows(), 3);
    BIT_ASSERT_EQ(A.getCols(), 2);

    BIT_EXPECT_EQ(A.getStrides()[0], origStride1);
    BIT_EXPECT_EQ(A.getStrides()[1], origStride0);

    // [ 1, 4 ]
    // [ 2, 5 ]
    // [ 3, 6 ]
    BIT_EXPECT_EQ(A(0, 0), 1.0f);
    BIT_EXPECT_EQ(A(0, 1), 4.0f);
    BIT_EXPECT_EQ(A(1, 0), 2.0f);
    BIT_EXPECT_EQ(A(1, 1), 5.0f);
    BIT_EXPECT_EQ(A(2, 0), 3.0f);
    BIT_EXPECT_EQ(A(2, 1), 6.0f);

    BitMth::linalg::Matrix<float> B(2, 3, nullptr, true);
    val = 1.0f;
    for (size_t i = 0; i < B.size(); ++i) B.getValues()[i] = val++;

    BitMth::linalg::Matrix<float> B_transposed = BitMth::linalg::Matrix<float>::t(B, nullptr);

    BIT_ASSERT_EQ(B_transposed.getRows(), 3);
    BIT_ASSERT_EQ(B_transposed.getCols(), 2);
    BIT_EXPECT_EQ(B_transposed(0, 1), 4.0f);

    BIT_EXPECT_EQ(B.getRows(), 2);
    BIT_EXPECT_EQ(B.getCols(), 3);
    BIT_EXPECT_EQ(B(0, 1), 2.0f);
}

BIT_TEST_CASE(MatrixDivisionSuccessAndFailures) {
    BitMth::linalg::Matrix<float> A(2, 2, nullptr, true);
    A.getValues()[0] = 10.0f; A.getValues()[1] = 20.0f;
    A.getValues()[2] = 30.0f; A.getValues()[3] = 40.0f;

    BitMth::linalg::Matrix<float> B(2, 2, nullptr, true);
    B.getValues()[0] = 2.0f; B.getValues()[1] = 4.0f;
    B.getValues()[2] = 5.0f; B.getValues()[3] = 8.0f;

    BitMth::linalg::Matrix<float> esperada(2, 2, nullptr, true);
    esperada.getValues()[0] = 5.0f; esperada.getValues()[1] = 5.0f;
    esperada.getValues()[2] = 6.0f; esperada.getValues()[3] = 5.0f;

    BitMth::linalg::Matrix<float> resultado = A / B;
    BIT_EXPECT_TRUE(resultado.isApprox(esperada));

    BitMth::linalg::Matrix<float> C(2, 3, nullptr, true);
    C.setWith(1.0f);
    BIT_ASSERT_THROWS(A / C);

    BitMth::linalg::Matrix<float> zeroDivisor(2, 2, nullptr, true);
    zeroDivisor.getValues()[0] = 2.0f;
    zeroDivisor.getValues()[1] = 0.00000001f; 
    zeroDivisor.getValues()[2] = 5.0f;
    zeroDivisor.getValues()[3] = 8.0f;

    BIT_ASSERT_THROWS(A / zeroDivisor);
}

BIT_TEST_CASE(MatrixStrictEquality) {
    BitMth::linalg::Matrix<float> A(2, 2, nullptr, true);
    A.getValues()[0] = 1.0f; A.getValues()[1] = 2.0f;
    A.getValues()[2] = 3.0f; A.getValues()[3] = 4.0f;

    BitMth::linalg::Matrix<float> B(2, 2, nullptr, true);
    B.getValues()[0] = 1.0f; B.getValues()[1] = 2.0f;
    B.getValues()[2] = 3.0f; B.getValues()[3] = 4.0f;

    BIT_EXPECT_TRUE(A == B);
    B.getValues()[3] = 4.001f; 
    BIT_EXPECT_FALSE(A == B);

    BitMth::linalg::Matrix<float> C(1, 4, nullptr, true);
    C.getValues()[0] = 1.0f; C.getValues()[1] = 2.0f;
    C.getValues()[2] = 3.0f; C.getValues()[3] = 4.0f;
    BIT_EXPECT_FALSE(A == C);
}

BIT_TEST_CASE(MatrixApproximateEquality) {
    BitMth::linalg::Matrix<float> A(2, 2, nullptr, true);
    A.getValues()[0] = 1.00000f;
    A.getValues()[1] = 2.00000f;

    BitMth::linalg::Matrix<float> B(2, 2, nullptr, true);
    B.getValues()[0] = 1.00005f; 
    B.getValues()[1] = 1.99995f; 

    BIT_EXPECT_TRUE(A.isApprox(B, 0.0001f));
    BIT_EXPECT_FALSE(A.isApprox(B, 0.00001f));
    BitMth::linalg::Matrix<float> C(1, 2, nullptr, true);
    BIT_EXPECT_FALSE(A.isApprox(C, 0.1f));
}

BIT_TEST_CASE(MatrixReductionsContiguous) {
    // [ 1.0, 2.0, 3.0 ]
    // [ 4.0, 5.0, 6.0 ]
    BitMth::linalg::Matrix<float> A(2, 3, nullptr, true);
    float val = 1.0f;
    for (size_t i = 0; i < A.size(); ++i) A.getValues()[i] = val++;

    float totalSum = A.reduceSumTotal();
    float diffTotal = totalSum - 21.0f;
    if (diffTotal < 0) diffTotal = -diffTotal;
    BIT_EXPECT_TRUE(diffTotal < BitMth::utils::EPSILON<float>);

    // [ 1+2+3 ] = [ 6  ]
    // [ 4+5+6 ]   [ 15 ]
    BitMth::linalg::Matrix<float> colSum = A.reduceSumCols();
    BIT_ASSERT_EQ(colSum.getRows(), 2);
    BIT_ASSERT_EQ(colSum.getCols(), 1);
    
    BitMth::linalg::Matrix<float> esperadaCol(2, 1, nullptr, true);
    esperadaCol.getValues()[0] = 6.0f;
    esperadaCol.getValues()[1] = 15.0f;
    BIT_EXPECT_TRUE(colSum.isApprox(esperadaCol));

    // [ 1+4, 2+5, 3+6 ] = [ 5, 7, 9 ]
    BitMth::linalg::Matrix<float> rowSum = A.reduceSumRows();
    BIT_ASSERT_EQ(rowSum.getRows(), 1);
    BIT_ASSERT_EQ(rowSum.getCols(), 3);
    
    BitMth::linalg::Matrix<float> esperadaRow(1, 3, nullptr, true);
    esperadaRow.getValues()[0] = 5.0f;
    esperadaRow.getValues()[1] = 7.0f;
    esperadaRow.getValues()[2] = 9.0f;
    BIT_EXPECT_TRUE(rowSum.isApprox(esperadaRow));
}

BIT_TEST_CASE(MatrixAddRowVectorBroadcasting) {
    // Matriz base A (2x3):
    // [ 1, 2, 3 ]
    // [ 4, 5, 6 ]
    BitMth::linalg::Matrix<float> A(2, 3, nullptr, true);
    float val = 1.0f;
    for (size_t i = 0; i < A.size(); ++i) A.getValues()[i] = val++;

    // Vector fila (1x3):
    // [ 10, 20, 30 ]
    BitMth::linalg::Matrix<float> rowVec(1, 3, nullptr, true);
    rowVec.getValues()[0] = 10.0f;
    rowVec.getValues()[1] = 20.0f;
    rowVec.getValues()[2] = 30.0f;

    // Resultado esperado:
    // [ 11, 22, 33 ]
    // [ 14, 25, 36 ]
    BitMth::linalg::Matrix<float> resultado = A.addRowVector(rowVec);
    BIT_ASSERT_EQ(resultado.getRows(), 2);
    BIT_ASSERT_EQ(resultado.getCols(), 3);

    BitMth::linalg::Matrix<float> esperada(2, 3, nullptr, true);
    esperada.getValues()[0] = 11.0f; esperada.getValues()[1] = 22.0f; esperada.getValues()[2] = 33.0f;
    esperada.getValues()[3] = 14.0f; esperada.getValues()[4] = 25.0f; esperada.getValues()[5] = 36.0f;

    BIT_EXPECT_TRUE(resultado.isApprox(esperada));

    BitMth::linalg::Matrix<float> badRowVec(1, 2, nullptr, true);
    BIT_ASSERT_THROWS(A.addRowVector(badRowVec));
}

BIT_TEST_CASE(MatrixUtilityOperations) {
    BitMth::linalg::Matrix<float> A(3, 3, nullptr, true);

    A.setWith(7.5f);
    for (size_t i = 0; i < A.size(); ++i) {
        BIT_EXPECT_EQ(A.getValues()[i], 7.5f);
    }

    A.clear();
    for (size_t i = 0; i < A.size(); ++i) {
        BIT_EXPECT_EQ(A.getValues()[i], 0.0f);
    }

    A.setOne();
    for (size_t i = 0; i < A.size(); ++i) {
        BIT_EXPECT_EQ(A.getValues()[i], 1.0f);
    }

    A.setIdentity();
    BitMth::linalg::Matrix<float> esperadaIdentidad(3, 3, nullptr, true);
    esperadaIdentidad.getValues()[0] = 1.0f; // (0,0)
    esperadaIdentidad.getValues()[4] = 1.0f; // (1,1)
    esperadaIdentidad.getValues()[8] = 1.0f; // (2,2)

    BIT_EXPECT_TRUE(A == esperadaIdentidad);

    BitMth::linalg::Matrix<float> B(2, 3, nullptr, true);
    BIT_ASSERT_THROWS(B.setIdentity());
}

BIT_TEST_CASE(MatrixHasNaNValidation) {
    using Matrix = BitMth::linalg::Matrix<float>;

    Matrix m(2, 2);
    m.getValues()[0] = 1.0f;
    m.getValues()[1] = 2.0f;
    m.getValues()[2] = 3.0f;
    m.getValues()[3] = 4.0f;

    BIT_EXPECT_FALSE(m.hasNaN());

    // Inyectar un NaN deliberadamente
    m(1,0) = std::numeric_limits<float>::quiet_NaN();

    BIT_EXPECT_TRUE(m.hasNaN());
}

BIT_TEST_CASE(Parallel_AdditionOperators) {
    BitMth::linalg::Matrix<int> m(500, 400, nullptr, true);
    m.setWith(5);
    
    m += 3;
    BitMth::linalg::Matrix<int> m2(500, 400, nullptr, true);
    m2.setWith(8);
    BIT_ASSERT_EQ(m2, m);

    BitMth::linalg::Matrix<int> m_plus = m + 2;
    BitMth::linalg::Matrix<int> m_plus_exp(500, 400, nullptr, true);
    m_plus_exp.setWith(10);
    BIT_ASSERT_EQ(m_plus_exp, m_plus);
    BIT_ASSERT_EQ(nullptr, m_plus.getArena());

    BitMth::linalg::Matrix<int> m_static = BitMth::linalg::Matrix<int>::add(m, 2, nullptr);
    BIT_ASSERT_EQ(m_plus_exp, m_static);
    BIT_ASSERT_EQ(nullptr, m_static.getArena());
}

BIT_TEST_CASE(Parallel_SubtractionOperators) {
    BitMth::linalg::Matrix<int> m(500, 400, nullptr, true);
    m.setWith(10);

    m -= 3;
    BitMth::linalg::Matrix<int> m2(500, 400, nullptr, true);
    m2.setWith(7);
    BIT_ASSERT_EQ(m2, m);

    BitMth::linalg::Matrix<int> m_sub = m - 2;
    BitMth::linalg::Matrix<int> m_sub_exp(500, 400, nullptr, true);
    m_sub_exp.setWith(5);
    BIT_ASSERT_EQ(m_sub_exp, m_sub);

    BitMth::linalg::Matrix<int> m_friend = 12 - m; // 12 - 7 = 5
    BIT_ASSERT_EQ(m_sub_exp, m_friend);

    BitMth::linalg::Matrix<int> m_st1 = BitMth::linalg::Matrix<int>::sub(m, 2, nullptr); // 7 - 2 = 5
    BIT_ASSERT_EQ(m_sub_exp, m_st1);
    BIT_ASSERT_EQ(nullptr, m_st1.getArena());

    BitMth::linalg::Matrix<int> m_st2 = BitMth::linalg::Matrix<int>::sub(12, m, nullptr); // 12 - 7 = 5
    BIT_ASSERT_EQ(m_sub_exp, m_st2);
    BIT_ASSERT_EQ(nullptr, m_st2.getArena());
}

BIT_TEST_CASE(Parallel_MultiplicationAndDivisionOperators) {
    BitMth::linalg::Matrix<float> m(500, 400, nullptr, true);
    m.setWith(20.0f);

    m *= 3.0f;
    BitMth::linalg::Matrix<float> m2(500, 400, nullptr, true);
    m2.setWith(60.0f);
    BIT_EXPECT_TRUE(m.isApprox(m2));

    m /= 2.0f;
    BitMth::linalg::Matrix<float> m3(500, 400, nullptr, true);
    m3.setWith(30.0f);
    BIT_EXPECT_TRUE(m.isApprox(m3));

    BitMth::linalg::Matrix<float> m_div = BitMth::linalg::Matrix<float>::div(m, 3.0f, nullptr); // 30 / 3 = 10
    BitMth::linalg::Matrix<float> m_div_exp(500, 400, nullptr, true);
    m_div_exp.setWith(10.0f);
    
    BIT_EXPECT_TRUE(m_div.isApprox(m_div_exp));
    BIT_ASSERT_EQ(nullptr, m_div.getArena());

    BIT_ASSERT_THROWS(m /= 0.0f);
    BIT_ASSERT_THROWS(m / 0.0f);
    BIT_ASSERT_THROWS(BitMth::linalg::Matrix<float>::div(m, 0.0f, nullptr));
}

BIT_TEST_CASE(Parallel_PowOperations) {
    BitMth::linalg::Matrix<float> m(500, 400, nullptr, true);
    m.setWith(4.0f);

    BitMth::linalg::Matrix<float> m_pow2 = m.pow(2.0f);
    BitMth::linalg::Matrix<float> exp_pow2(500, 400, nullptr, true);
    exp_pow2.setWith(16.0f);
    BIT_EXPECT_TRUE(m_pow2.isApprox(exp_pow2));

    BitMth::linalg::Matrix<float> m_pow_frac = exp_pow2.pow(0.5f);
    BIT_EXPECT_TRUE(m_pow_frac.isApprox(m, 1e-4f));

    m.powInPlace(2.0f); // 4^2 = 16
    BIT_EXPECT_TRUE(m.isApprox(exp_pow2));

    m.powInPlace(0.0f);
    BitMth::linalg::Matrix<float> one(500, 400, nullptr, true);
    one.setOne();
    BIT_ASSERT_EQ(one, m);
}

BIT_TEST_CASE(Parallel_MatrixMatrixAddition) {
    BitMth::linalg::Matrix<int> m1(500, 400, nullptr, true);
    BitMth::linalg::Matrix<int> m2(500, 400, nullptr, true);
    
    m1.setWith(5);
    m2.setWith(3);

    m1 += m2; // Cada elemento: 5 + 3 = 8
    
    BitMth::linalg::Matrix<int> expected_sum(500, 400, nullptr, true);
    expected_sum.setWith(8);
    BIT_ASSERT_EQ(expected_sum, m1);

    m1.setWith(5);

    BitMth::linalg::Matrix<int> m_res = m1 + m2;
    BIT_ASSERT_EQ(expected_sum, m_res);
    BIT_ASSERT_EQ(nullptr, m_res.getArena());

    BitMth::linalg::Matrix<int> m_static = BitMth::linalg::Matrix<int>::add(m1, m2, nullptr);
    BIT_ASSERT_EQ(expected_sum, m_static);
    BIT_ASSERT_EQ(nullptr, m_static.getArena());

    BitMth::linalg::Matrix<int> m_wrong(500, 399, nullptr, true);
    BIT_ASSERT_THROWS(m1 += m_wrong);
    BIT_ASSERT_THROWS(m1 + m_wrong);
    BIT_ASSERT_THROWS(BitMth::linalg::Matrix<int>::add(m1, m_wrong, nullptr));
}

BIT_TEST_CASE(Parallel_MatrixMatrixSubtraction) {
    BitMth::linalg::Matrix<int> m1(500, 400, nullptr, true);
    BitMth::linalg::Matrix<int> m2(500, 400, nullptr, true);
    
    m1.setWith(10);
    m2.setWith(4);

    m1 -= m2; // Cada elemento: 10 - 4 = 6
    
    BitMth::linalg::Matrix<int> expected_sub(500, 400, nullptr, true);
    expected_sub.setWith(6);
    BIT_ASSERT_EQ(expected_sub, m1);

    m1.setWith(10);

    BitMth::linalg::Matrix<int> m_res = m1 - m2;
    BIT_ASSERT_EQ(expected_sub, m_res);
    BIT_ASSERT_EQ(nullptr, m_res.getArena());

    BitMth::linalg::Matrix<int> m_static = BitMth::linalg::Matrix<int>::sub(m1, m2, nullptr);
    BIT_ASSERT_EQ(expected_sub, m_static);
    BIT_ASSERT_EQ(nullptr, m_static.getArena());

    BitMth::linalg::Matrix<int> m_wrong(499, 400, nullptr, true);
    BIT_ASSERT_THROWS(m1 -= m_wrong);
    BIT_ASSERT_THROWS(m1 - m_wrong);
    BIT_ASSERT_THROWS(BitMth::linalg::Matrix<int>::sub(m1, m_wrong, nullptr));
}

BIT_TEST_CASE(Parallel_MatrixMatrixMultiplication) {
    BitMth::linalg::Matrix<float> m1(500, 400, nullptr, true);
    m1.setWith(2.0f);

    BitMth::linalg::Matrix<float> m2(400, 300, nullptr, true);
    m2.setWith(3.0f);

    BitMth::linalg::Matrix<float> expected_mul(500, 300, nullptr, true);
    expected_mul.setWith(2400.0f);

    BitMth::linalg::Matrix<float> m_static = BitMth::linalg::Matrix<float>::mul(m1, m2, nullptr);
    BIT_EXPECT_TRUE(m_static.isApprox(expected_mul));
    BIT_ASSERT_EQ(nullptr, m_static.getArena());

    BitMth::linalg::Matrix<float> m_operator = m1 * m2;
    BIT_EXPECT_TRUE(m_operator.isApprox(expected_mul));
    BIT_ASSERT_EQ(nullptr, m_operator.getArena());

    BitMth::linalg::Matrix<float> m_wrong(399, 300, nullptr, true);
    BIT_ASSERT_THROWS(BitMth::linalg::Matrix<float>::mul(m1, m_wrong, nullptr));
    BIT_ASSERT_THROWS(m1 * m_wrong);
}

BIT_TEST_CASE(Parallel_MatrixMatrixMultiplication_IdentityAndSmall) {
    BitMth::linalg::Matrix<float> m1(500, 400, nullptr, true);
    m1.setWith(7.5f);

    BitMth::linalg::Matrix<float> identity(400, 400, nullptr, true);
    identity.setIdentity(); // Matriz identidad 400x400

    BitMth::linalg::Matrix<float> m_identity_res = m1 * identity;
    BIT_EXPECT_TRUE(m_identity_res.isApprox(m1));
}

BIT_TEST_CASE(MatrixInPlaceMultiplication_Operator) {
    BitMth::linalg::Matrix<float> largeA(500, 400, nullptr, true);
    largeA.setWith(2.0f);

    BitMth::linalg::Matrix<float> largeB(400, 300, nullptr, true);
    largeB.setWith(3.0f);

    BitMth::linalg::Matrix<float> expected_large(500, 300, nullptr, true);
    expected_large.setWith(2400.0f);
    largeA *= largeB;
   
    BIT_ASSERT_EQ(500, largeA.getRows());
    BIT_ASSERT_EQ(300, largeA.getCols());
    BIT_EXPECT_TRUE(largeA.isApprox(expected_large));
    BitMth::linalg::Matrix<float> invalidB(400, 300, nullptr, true);
    BIT_ASSERT_THROWS(largeA *= invalidB);
}
BIT_GROUP_END()
