#include "optimizer.hpp"




void RMSProp::step(const std::vector<Tensor*>& Parameters, const std::vector<Tensor*>& Gradients) {
    if (Parameters.size() != Gradients.size())
        throw std::runtime_error("Parameter/Gradient size mismatch.");

    if (MeanSquare.empty()) {
        MeanSquare.reserve(Parameters.size());

        for (const Tensor* parameter : Parameters) {
            MeanSquare.emplace_back();
            MeanSquare.back().allocate(parameter->shape());
            MeanSquare.back().fill(0.0f);
        }
    }

    for (size_t i = 0; i < Parameters.size(); ++i) {
        rmsPropUpdate(*Parameters[i], *Gradients[i], MeanSquare[i], LearningRate, Beta, Epsilon);
    }
}