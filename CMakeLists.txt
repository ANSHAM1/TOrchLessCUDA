cmake_minimum_required(VERSION 3.25)

project(TorchLess LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

find_package(CUDAToolkit REQUIRED)

add_executable(app
    main.cpp

    src/util_kernels.cu

    src/layers/ActivationLayer.cpp 
    src/layers/FlattenLayer.cpp
    src/layers/Conv2dLayer.cpp
    src/layers/PoolingLayer.cpp
    src/layers/DenseLayer.cpp
    src/layers/OutputLayer.cpp

    src/modules/tensor.cpp 
    src/modules/sequential.cpp
    src/modules/execution_ctx.cpp

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