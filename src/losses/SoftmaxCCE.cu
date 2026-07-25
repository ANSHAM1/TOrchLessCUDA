#include "kernel.cuh"
#include "template.cuh"

#include "tensor.hpp"

#include "device_launch_parameters.h"

#include <cmath>
#include <stdexcept>





__global__ void SoftmaxCCELossKernel(const float* prediction, const int* labels, float* loss, size_t Batch, size_t Classes) {
    size_t batch = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (batch >= Batch)
        return;

    int label = labels[batch];

    float probability = prediction[batch * Classes + label];

    probability = fmaxf(probability, 1e-7f);

    loss[batch] = -logf(probability);
}





float SoftmaxCCELoss(const Tensor& prediction, const Tensor& labels) {
    size_t Batch = prediction.shape()[0];

    size_t Classes = prediction.shape()[1];

    Tensor batchLoss({ Batch });

    int threads = 256;
    int blocks = (Batch + threads - 1) / threads;


    ExecuteKernel("SoftmaxCCELossKernel", blocks, threads, 0, 0, SoftmaxCCELossKernel,  prediction.data(),
        labels.int_data(), batchLoss.data(), Batch, Classes);


    std::vector<float> hostLoss(Batch);

    batchLoss.copyToHost(hostLoss.data(), Batch);

    float total = 0.0f;
    for (size_t i = 0; i < Batch; i++)
        total += hostLoss[i];

    return total / static_cast<float>(Batch);
}




__global__ void SoftmaxCCELossBackwardKernel(const float* prediction, const int* labels, float* grad, size_t Batch, size_t Classes) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    size_t total = Batch * Classes;
    if (idx >= total)
        return;

    size_t batch = idx / Classes;
    size_t cls = idx % Classes;

    float value = prediction[idx];

    if (cls == labels[batch])
        value -= 1.0f;

    grad[idx] = value;
}




void SoftmaxCCELossBackward(const Tensor& prediction, const Tensor& labels, Tensor& grad) {
    size_t Batch = prediction.shape()[0];

    size_t Classes = prediction.shape()[1];

    size_t size = Batch * Classes;

    int threads = 256;
    int blocks = (size + threads - 1) / threads;


    ExecuteKernel("SoftmaxCCELossBackwardKernel", blocks, threads, 0, 0, SoftmaxCCELossBackwardKernel,
        prediction.data(), labels.int_data(), grad.data(), Batch, Classes);
}