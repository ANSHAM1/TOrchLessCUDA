#include "optimizer.hpp"




void SGD::step(const std::vector<Tensor*>& parameters, const std::vector<Tensor*>& gradients) {
    if (parameters.size() != gradients.size())
        throw std::runtime_error("Parameter/Gradient size mismatch.");

    for (size_t i = 0; i < parameters.size(); ++i) {
        sgdUpdate(*parameters[i], *gradients[i], LearningRate);
    }
}