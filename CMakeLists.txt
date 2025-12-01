cmake_minimum_required(VERSION 3.30)
project(MedicalMNIST_NeuralNetwork CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
if(CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wextra")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/scripts)

# FASE 1: Convertidor de imágenes
add_executable(convert_images scripts/convert_images_to_csv.cpp)

# FASE 2: Ejemplos de entrenamiento
add_executable(train_xor examples/1_train_xor.cpp)

# FASE 3: Programa principal
add_executable(train_medical_mnist examples/2_train_medical_mnist.cpp)
add_executable(evaluate_model examples/3_evaluate_model.cpp)
add_executable(predict_single examples/4_predict_single.cpp)

# Tests (optional)
option(BUILD_TESTS "Build test executables" OFF)
if(BUILD_TESTS)
    # add_executable(test_tensor tests/test_tensor.cpp)
    # add_executable(test_layers tests/test_layers.cpp)
    # add_executable(test_data_loader tests/test_data_loader.cpp)
endif()
