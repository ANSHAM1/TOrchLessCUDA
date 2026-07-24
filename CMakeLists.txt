cmake_minimum_required(VERSION 3.25)

project(TorchLess LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

find_package(CUDAToolkit REQUIRED)

add_executable(app
    main.cpp

    src/tensor.cpp
    src/util_kernels.cu

    src/modules/Sequential.cpp
    src/modules/ActivationLayer.cpp 
    src/modules/FlattenLayer.cpp
    src/modules/Conv2dLayer.cpp
    src/modules/PoolingLayer.cpp
    src/modules/DenseLayer.cpp
    src/modules/OutputLayer.cpp

    src/losses/loss.cpp
    src/losses/SoftmaxCCE.cu

    src/kernels/ActivationKernel.cu 
    src/kernels/Conv2dKernel.cu 
    src/kernels/PoolingKernel.cu 
    src/kernels/DenseKernel.cu 
    src/kernels/OutputKernel.cu
)

target_include_directories(app PRIVATE
    include
    utils
)

target_link_libraries(app PRIVATE
    CUDA::cudart
)

set_target_properties(app PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)