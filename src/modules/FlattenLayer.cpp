#include "layer.hpp"

//#include "cuda_runtime.h"



FlattenLayer::FlattenLayer(const std::vector<size_t>& InputShape) {
    if (InputShape.size() < 2)
        throw std::invalid_argument("Flatten requires input dimension >= 2");

    size_t Batch = InputShape[0];

    size_t Features = 1;
    for (size_t i = 1; i < InputShape.size(); i++)
        Features *= InputShape[i];

    OutputShape = { Batch, Features };
}



void FlattenLayer::forward(const Tensor& input, Tensor& output, ExecutionContext&) {
    output = input.view(OutputShape);
}