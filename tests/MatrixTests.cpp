#include <BitTest/BitTest.hpp>
#include <BitMth/linalg/Matrix.hpp>

BIT_TEST_CASE(MatrixInitializationDefault){
    BitMth::linalg::Matrix<float> m;
    BIT_ASSERT_EQ(0, m.getRows());
    BIT_ASSERT_EQ(0, m.getCols());
    BIT_ASSERT_EQ(0, m.size());
    BIT_ASSERT_EQ(0, m.getStrides()[0]);
    BIT_ASSERT_EQ(0, m.getStrides()[1]);
    BIT_ASSERT_EQ(nullptr, m.getArena());
    BIT_ASSERT_EQ(nullptr, m.getValues());
}

BIT_TEST_CASE(MatrixInitialization){
    BitMth::linalg::Matrix<double> m(2,3,nullptr);
    BIT_ASSERT_EQ(2, m.getRows());
    BIT_ASSERT_EQ(3, m.getCols());
    BIT_ASSERT_EQ(6, m.size());
    BIT_ASSERT_EQ(m.getCols() * sizeof(double), m.getStrides()[0]);
    BIT_ASSERT_EQ(sizeof(double), m.getStrides()[1]);
    BIT_ASSERT_EQ(nullptr, m.getArena());
}

BIT_TEST_CASE(MatrixInitializationZero){
    BitMth::linalg::Matrix<float> m(2,3,nullptr,true);
    BIT_ASSERT_EQ(2, m.getRows());
    BIT_ASSERT_EQ(3, m.getCols());
    BIT_ASSERT_EQ(6, m.size());
    BIT_ASSERT_EQ(nullptr, m.getArena());
    for (size_t i = 0; i < m.getRows(); i++){
        for (size_t j = 0; j < m.getCols(); j++){
            BIT_ASSERT_EQ(0, m(i,j));
        }
    }
}

BIT_TEST_CASE(MatrixCopyConstructor){
    BitMth::linalg::Matrix<float> m(2,3,nullptr,true);
    m(1,2) = 78;
    m(0,1) = 754;
    m(1,0) = 342;
    BitMth::linalg::Matrix<float> m2(m);
    BIT_ASSERT_EQ(m,m2);
    BIT_ASSERT_EQ(nullptr, m2.getArena());
    m2(1,2) = 999;
    BIT_ASSERT(m(1,2) != m2(1,2));
}

BIT_TEST_CASE(MatrixCloneFunct){
    BitMth::linalg::Matrix<double> m(2,3,nullptr,true);
    m(1,2) = 28;
    m(0,1) = 7544;
    m(1,0) = 392;
    BitMth::linalg::Matrix<double> m2 = m.clone(nullptr);
    BIT_ASSERT_EQ(m,m2);
    BIT_ASSERT_EQ(nullptr, m2.getArena());
    m2(1,2) = 999;
    BIT_ASSERT(m(1,2) != m2(1,2));
}

BIT_TEST_CASE(MatrixMoveConstructor) {
    BitMth::linalg::Matrix<double> m1(2, 3, nullptr, true);
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
    src(0, 0) = 1.1;
    src(0, 1) = 2.2;
    src(1, 2) = 3.3;
    BitMth::core::Arena fakeArena(18 * sizeof(double));
    BitMth::linalg::Matrix<double> dest(4, 4, &fakeArena, false);
    dest = src;
    BIT_ASSERT(dest.getRows() == 2);
    BIT_ASSERT(dest.getCols() == 3);
    BIT_ASSERT(dest.size() == 6);
    BIT_ASSERT(dest.getArena() == nullptr);
    BIT_ASSERT(dest(0, 0) == 1.1);
    BIT_ASSERT(dest(0, 1) == 2.2);
    BIT_ASSERT(dest(1, 2) == 3.3);
    dest = dest;
    BIT_ASSERT(dest(1, 2) == 3.3);
}

BIT_TEST_CASE(MatrixOperatorIndexing) {
    BitMth::linalg::Matrix<double> m(2, 2, nullptr, true);
    m(0, 0) = 10.5;
    m(0, 1) = 20.5;
    m(1, 0) = 30.5;
    m(1, 1) = 40.5;
    BIT_ASSERT(m(0, 0) == 10.5);
    BIT_ASSERT(m(0, 1) == 20.5);
    BIT_ASSERT(m(1, 0) == 30.5);
    BIT_ASSERT(m(1, 1) == 40.5);
    const BitMth::linalg::Matrix<double>& constMatrix = m;
    BIT_ASSERT(constMatrix(0, 0) == 10.5);
    BIT_ASSERT(constMatrix(1, 1) == 40.5);
}

BIT_TEST_CASE(MatrixIndexingOutOfBounds) {
    BitMth::linalg::Matrix<double> m(2, 3, nullptr, true);
    try { 
        m(2, 0);
        BIT_FAIL("Expected exception for out-of-bounds row index (2, 0) but none was thrown.");
    } catch (...) { BIT_PASS();; }

    try {
        m(0, 3);
        BIT_FAIL("Expected exception for out-of-bounds column index (0, 3) but none was thrown.");
    } catch (...) { BIT_PASS();}
}