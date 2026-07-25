#include "optimizer.hpp"




void Adam::step(const std::vector<Tensor*>& Parameters, const std::vector<Tensor*>& Gradients) {
    if (Parameters.size() != Gradients.size())
        throw std::runtime_error("Parameter/Gradient size mismatch.");

    if (FirstMoment.empty()) {
        FirstMoment.reserve(Parameters.size());
        SecondMoment.reserve(Parameters.size());

        for (const Tensor* parameter : Parameters) {
            FirstMoment.emplace_back();
            FirstMoment.back().allocate(parameter->shape());
            FirstMoment.back().fill(0.0f);

            SecondMoment.emplace_back();
            SecondMoment.back().allocate(parameter->shape());
            SecondMoment.back().fill(0.0f);
        }
    }

    ++TimeStep;
    for (size_t i = 0; i < Parameters.size(); ++i) {
        adamUpdate(*Parameters[i], *Gradients[i], FirstMoment[i], SecondMoment[i], LearningRate,
            Beta1, Beta2, Epsilon, TimeStep);
    }
}