cmake_minimum_required(VERSION 3.15)
project(BitMth VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(BitMth INTERFACE)

target_include_directories(BitMth INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_options(BitMth INTERFACE -O3 -ffast-math)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    message(STATUS "Modo desarrollo: Activado")
    
    include(FetchContent)
    FetchContent_Declare(
        BitTest
        GIT_REPOSITORY https://github.com/MaisenRv/BitTest.git
        GIT_TAG        main
    )

    FetchContent_MakeAvailable(BitTest)

    message(STATUS "Compilando ejemplos de BitMth")
    add_executable(ejemplo_bitmth examples/main.cpp)
    target_link_libraries(ejemplo_bitmth PRIVATE BitMth)

    message(STATUS "Compilando pruebas de BitMth usando BitTest")
    file(GLOB SOURCES_TESTS "tests/*.cpp")
    add_executable(run_math_tests ${SOURCES_TESTS})
    target_link_libraries(run_math_tests PRIVATE BitMth BitTest)
endif()