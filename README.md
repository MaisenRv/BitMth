# BitMth

A modern C++ library for linear algebra, neural networks, signal processing, and geometry.

## ⚠️ Project Status: Currently in Active Development

This project is under active development. API and interfaces may change between versions.

## Features

- **Linear Algebra**: Matrix operations, vector types (`Vec2`), and linear algebra utilities
- **Neural Networks**: Full AI/ML stack with autograd, activation functions, loss functions, and optimizers
- **Signal Processing**: Fourier transforms (DFT) and signal analysis
- **Geometry**: Geometric types and factory operations
- **I/O**: WAV audio reading/writing, NumPy (.npy) file support for MNIST datasets
- **Random**: Random number generation and matrix randomization
- **Parallel Computing**: Arena and parallel execution utilities

## Installation

```bash
git clone https://github.com/MaisenRv/BitMth.git
cd BitMth
mkdir build && cd build
cmake -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON ..
cmake --build .
```

## Quick Start

### Matrix Operations
```cpp
#include <BitMth.hpp>

using namespace BitMth;

int main() {
    Matrix<float> m(5, 5);
    m.print();
    return 0;
}
```

### Neural Network
```cpp
#include <BitMth/ia/IA.hpp>

IA::Sequential model;
model.add(IA::Linear(784, 128));
model.add(IA::ReLU());
model.add(IA::Linear(128, 10));
model.add(IA::Softmax());
```

### I/O Operations
```cpp
#include <BitMth/io/DataNpyIO.hpp>

auto images = DataNpyIO::load("mnist_X_train.npy");
auto labels = DataNpyIO::load("mnist_y_train.npy");
```

## Building

```bash
# Build with tests
cmake -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON ..
cmake --build .

# Build examples only  
cmake -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=ON ..
cmake --build .

# Run example
./build/ejemplo_bitmth

# Run tests
./build/run_math_tests
```

## Repository Connections

This project is used in conjunction with:
- [EasySDL](https://github.com/MaisenRv/EasySDL) - SDL2 wrapper for C++
- [neural-network-from-scratch](https://github.com/MaisenRv/neural-network-from-scratch) - C++ neural network implementation

## License

MIT License - see LICENSE file for details.

## Author

David Santiago Mancera Robles
