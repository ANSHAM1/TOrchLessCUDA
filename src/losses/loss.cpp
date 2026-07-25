#include "module.hpp"

#include <stdexcept>




float Loss::forward(const Tensor& prediction, const Tensor& target) const {
    if (Type == "CCE")
        return SoftmaxCCELoss(prediction, target);
   
    else
        throw std::runtime_error("Unsupported loss type");
}


void Loss::backward(const Tensor& prediction, const Tensor& target, Tensor& grad) const {
    if (Type == "CCE")
        SoftmaxCCELossBackward(prediction, target, grad);

    else
        throw std::runtime_error("Unsupported loss type");
}