#include "optimizer.hpp"



void Momentum::step(const std::vector<Tensor*>& Parameters, const std::vector<Tensor*>& Gradients) {
    if (Parameters.size() != Gradients.size())
        throw std::runtime_error("Parameter/Gradient size mismatch.");

    if (Velocity.empty()) {
        Velocity.reserve(Parameters.size());

        for (const Tensor* parameter : Parameters) {
            Velocity.emplace_back();
            Velocity.back().allocate(parameter->shape());
            Velocity.back().fill(0.0f);
        }
    }

    for (size_t i = 0; i < Parameters.size(); ++i) {
        momentumUpdate(*Parameters[i], *Gradients[i], Velocity[i], LearningRate, Beta);
    }
}