#include "optimizer.hpp"




void AdaGrad::step(const std::vector<Tensor*>& Parameters, const std::vector<Tensor*>& Gradients) {
    if (Parameters.size() != Gradients.size())
        throw std::runtime_error("Parameter/Gradient size mismatch.");

    if (Accumulator.empty()) {
        Accumulator.reserve(Parameters.size());

        for (const Tensor* parameter : Parameters) {
            Accumulator.emplace_back();
            Accumulator.back().allocate(parameter->shape());
            Accumulator.back().fill(0.0f);
        }
    }

    for (size_t i = 0; i < Parameters.size(); ++i) {
        adagradUpdate(*Parameters[i], *Gradients[i], Accumulator[i], LearningRate, Epsilon);
    }
}