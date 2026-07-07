#include <BitTest/BitTest.hpp>
#include <BitMth/linalg/Matrix.hpp>

BIT_TEST_CASE(MatrixInitializationDefault){
    BitMth::linalg::Matrix<float> m;
    BIT_ASSERT(m.getRows() == 0);
    BIT_ASSERT(m.getCols() == 0);
    BIT_ASSERT(m.size() == 0);
    BIT_ASSERT(m.getStrides()[0] == 0);
    BIT_ASSERT(m.getStrides()[1] == 0);
    BIT_ASSERT(m.getArena() == nullptr);
    BIT_ASSERT(m.getValues() == nullptr);
}

BIT_TEST_CASE(MatrixInitialization){
    BitMth::linalg::Matrix<double> m(2,3,nullptr);
    BIT_ASSERT(m.getRows() == 2);
    BIT_ASSERT(m.getCols() == 3);
    BIT_ASSERT(m.size() == 6);
    BIT_ASSERT(m.getArena() == nullptr);
    BIT_ASSERT(m.getStrides()[0] == m.getCols() * sizeof(double));
    BIT_ASSERT(m.getStrides()[1] == sizeof(double));
}

BIT_TEST_CASE(MatrixInitializationZero){
    BitMth::linalg::Matrix<float> m(2,3,nullptr,true);
    BIT_ASSERT(m.getRows() == 2);
    BIT_ASSERT(m.getCols() == 3);
    BIT_ASSERT(m.size() == 6);
    BIT_ASSERT(m.getArena() == nullptr);
    for (size_t i = 0; i < m.getRows(); i++){
        for (size_t j = 0; j < m.getCols(); j++){
            BIT_ASSERT(m(i,j) == 0);
        }
    }
}

BIT_TEST_CASE(MatrixCopyConstructor){
    BitMth::linalg::Matrix<float> m(2,3,nullptr,true);
    m(1,2) = 78;
    m(0,1) = 754;
    m(1,0) = 342;
    BitMth::linalg::Matrix<float> m2(m);
    BIT_ASSERT(m.getRows() == m2.getRows());
    BIT_ASSERT(m.getCols() == m2.getCols());
    BIT_ASSERT(m.size() == m2.size());
    BIT_ASSERT(m.getArena() == nullptr);
    BIT_ASSERT(m2(1,2) == 78);
    BIT_ASSERT(m2(0,1) == 754);
    BIT_ASSERT(m2(1,0) == 342);
}

BIT_TEST_CASE(MatrixCloneFunct){
    BitMth::linalg::Matrix<double> m(2,3,nullptr,true);
    m(1,2) = 28;
    m(0,1) = 7544;
    m(1,0) = 392;
    BitMth::linalg::Matrix<double> m2 = m.clone(nullptr);
    BIT_ASSERT(m.getRows() == m2.getRows());
    BIT_ASSERT(m.getCols() == m2.getCols());
    BIT_ASSERT(m.size() == m2.size());
    BIT_ASSERT(m.getArena() == nullptr);
    BIT_ASSERT(m2(1,2) == 28);
    BIT_ASSERT(m2(0,1) == 7544);
    BIT_ASSERT(m2(1,0) == 392);
}

BIT_TEST_CASE(MatrixMoveConstructor) {
    BitMth::linalg::Matrix<double> m1(2, 3, nullptr, true);
    m1(0, 0) = 5.5;
    m1(1, 2) = 10.5;
    const double* valuesPtr = m1.getValues(); 
    BitMth::linalg::Matrix<double> m2(std::move(m1));
    BIT_ASSERT(m2.getRows() == 2);
    BIT_ASSERT(m2.getCols() == 3);
    BIT_ASSERT(m2.size() == 6);
    BIT_ASSERT(m2(0, 0) == 5.5);
    BIT_ASSERT(m2(1, 2) == 10.5);
    BIT_ASSERT(m2.getValues() == valuesPtr); 
    BIT_ASSERT(m1.getRows() == 0);
    BIT_ASSERT(m1.getCols() == 0);
    BIT_ASSERT(m1.size() == 0);
    BIT_ASSERT(m1.getStrides()[0] == 0);
    BIT_ASSERT(m1.getStrides()[1] == 0);
    BIT_ASSERT(m1.getValues() == nullptr);
    BIT_ASSERT(m1.getArena() == nullptr);
}

BIT_TEST_CASE(MatrixMoveAssignment) {
    BitMth::linalg::Matrix<double> src(2, 3, nullptr, true);
    src(0, 0) = 111.0;
    src(1, 1) = 222.0;
    const double* srcPtr = src.getValues();
    BitMth::linalg::Matrix<double> dest(5, 5, nullptr, true);
    dest = std::move(src);
    BIT_ASSERT(dest.getRows() == 2);
    BIT_ASSERT(dest.getCols() == 3);
    BIT_ASSERT(dest(0, 0) == 111.0);
    BIT_ASSERT(dest(1, 1) == 222.0);
    BIT_ASSERT(dest.getValues() == srcPtr);
    BIT_ASSERT(src.getRows() == 0);
    BIT_ASSERT(src.getCols() == 0);
    BIT_ASSERT(src.getValues() == nullptr);
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
    bool caughtRows = false;
    try {
        m(2, 0);
    } catch (...) {
        caughtRows = true;
    }
    BIT_ASSERT(caughtRows == true);

    bool caughtCols = false;
    try {
        m(0, 3);
    } catch (...) {
        caughtCols = true;
    }
    BIT_ASSERT(caughtCols == true);
}