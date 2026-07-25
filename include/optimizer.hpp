#pragma once

#include <vector>

#include "tensor.hpp"




class Optimizer {

protected:

    float LearningRate;

public:

    explicit Optimizer(float learningRate) : LearningRate(learningRate) {}

    virtual ~Optimizer() = default;

    virtual void step(
        const std::vector<Tensor*>& Parameters,
        const std::vector<Tensor*>& Gradients
    ) = 0;

    float learningRate() const { return LearningRate; }

    void setLearningRate(float lr) { LearningRate = lr; }

};




class SGD : public Optimizer {

public:

    explicit SGD(float learningRate = 1e-3f) : Optimizer(learningRate) {}

    void step(
        const std::vector<Tensor*>& Parameters,
        const std::vector<Tensor*>& Gradients
    ) override;

};




class Momentum : public Optimizer {

private:

    float Beta;

    std::vector<Tensor> Velocity;

public:

    Momentum(float learningRate = 1e-3f, float beta = 0.9f) : Optimizer(learningRate), Beta(beta) {}

    void step(
        const std::vector<Tensor*>& Parameters,
        const std::vector<Tensor*>& Gradients
    ) override;

};




class AdaGrad : public Optimizer {

private:

    float Epsilon;

    std::vector<Tensor> Accumulator;

public:

    AdaGrad(float learningRate = 1e-2f, float epsilon = 1e-8f) : Optimizer(learningRate), Epsilon(epsilon) {}

    void step(
        const std::vector<Tensor*>& Parameters,
        const std::vector<Tensor*>& Gradients
    ) override;

};




class RMSProp : public Optimizer {

private:

    float Beta;
    float Epsilon;

    std::vector<Tensor> MeanSquare;

public:

    RMSProp(float learningRate = 1e-3f, float beta = 0.99f, float epsilon = 1e-8f) : Optimizer(learningRate), Beta(beta), Epsilon(epsilon) {}

    void step(
        const std::vector<Tensor*>& Parameters,
        const std::vector<Tensor*>& Gradients
    ) override;

};




class Adam : public Optimizer {

private:

    float Beta1;
    float Beta2;
    float Epsilon;

    size_t TimeStep = 0;

    std::vector<Tensor> FirstMoment;
    std::vector<Tensor> SecondMoment;

public:

    Adam(float learningRate = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
        : Optimizer(learningRate), Beta1(beta1), Beta2(beta2), Epsilon(epsilon) {}

    void step(
        const std::vector<Tensor*>& Parameters,
        const std::vector<Tensor*>& Gradients
    ) override;

};