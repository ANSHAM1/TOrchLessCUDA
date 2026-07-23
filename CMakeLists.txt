cmake_minimum_required(VERSION 3.25)

project(TorchLess LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

find_package(CUDAToolkit REQUIRED)

add_executable(app
    main.cpp
    src/tensor.cpp
    src/util_kernels.cu
    src/modules/layer.cpp
)

target_include_directories(app PRIVATE
    include
)

target_link_libraries(app PRIVATE
    CUDA::cudart
)

set_target_properties(app PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)