cmake_minimum_required(VERSION 3.25)

project(TorchLess LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

add_executable(app
    src/main.cpp
    src/macros.cu
    src/kernels/kernel.cu
)

target_include_directories(app PRIVATE
    include
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include",
)


set_target_properties(app PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)