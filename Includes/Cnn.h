#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <functional>
#include <string>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <iomanip>

#include "KernelWrappers.cuh"
#include "Batch.h"
using namespace std;

static void showProgressBar(int current, int total, int barWidth);

class Layer {
public:
    virtual void forward(shared_ptr<BatchWrapper>& BWi) = 0;
    virtual shared_ptr<BatchWrapper> getBWout() const = 0;
    bool isForTesting = true;

    virtual ~Layer() {}
};

class Conv2dLayer : public Layer {
public:
    shared_ptr<BatchWrapper> BWk;
    shared_ptr<BatchWrapper> BWout;
    int S, P;

    Conv2dLayer(shared_ptr<BatchWrapper>& BWi, int Cout, int Cin, int kH, int kW, int s, int p);

    void forward(shared_ptr<BatchWrapper>& BWi) override;
    shared_ptr<BatchWrapper> getBWout() const override { return BWout; }
};

class ActivationLayer : public Layer {
public:
    shared_ptr<BatchWrapper> BWout;
    string TYPE;

    ActivationLayer(shared_ptr<BatchWrapper>& BWi, const string& type);

    void forward(shared_ptr<BatchWrapper>& BWi) override;
    shared_ptr<BatchWrapper> getBWout() const override { return BWout; }
};

class PoolingLayer : public Layer {
public:
    shared_ptr<BatchWrapper> BWout;
    int kH, kW, S, P;
    string TYPE;

    PoolingLayer(shared_ptr<BatchWrapper>& BWi, int kh, int kw, int s, int p, const string& type);

    void forward(shared_ptr<BatchWrapper>& BWi) override;
    shared_ptr<BatchWrapper> getBWout() const override { return BWout; }
};

class BatchNormLayer : public Layer {
public:
    shared_ptr<BatchWrapper> BWout;
    shared_ptr<BatchWrapper> GAMMA;
    shared_ptr<BatchWrapper> BETA;

    BatchNormLayer(shared_ptr<BatchWrapper>& BWi);

    void forward(shared_ptr<BatchWrapper>& BWi) override;
    shared_ptr<BatchWrapper> getBWout() const override { return BWout; }
};

class DropoutLayer : public Layer {
public:
    shared_ptr<BatchWrapper> BWmask;
    shared_ptr<BatchWrapper> BWout;
    float Prob;

    DropoutLayer(shared_ptr<BatchWrapper>& BWi, float p);

    void forward(shared_ptr<BatchWrapper>& BWi) override;
    shared_ptr<BatchWrapper> getBWout() const override { return BWout; }
};

class DenseLayer : public Layer {
public:
    shared_ptr<BatchWrapper> WEIGHTS;
    shared_ptr<BatchWrapper> BIAS;
    shared_ptr<BatchWrapper> BWout;
    int FEATURES_COUNT;

    DenseLayer(shared_ptr<BatchWrapper>& BWi, int NumFeatures);

    void forward(shared_ptr<BatchWrapper>& BWi) override;
    shared_ptr<BatchWrapper> getBWout() const override { return BWout; }
};

class OutputLayer : public Layer {
public:
    shared_ptr<BatchWrapper> BWout;
    shared_ptr<BatchWrapper> LABELS;

    OutputLayer(shared_ptr<BatchWrapper>& BWi, shared_ptr<BatchWrapper>& labels);

    void forward(shared_ptr<BatchWrapper>& BWi) override;
    shared_ptr<BatchWrapper> getBWout() const override { return BWout; }
};

class Network {
public:
    shared_ptr<BatchWrapper> CONFUSION_MATRIX;
    float ACCURACY;
    vector<float> PRECISION, RECALL, F1;

    int NUM_BATCHS;

    vector<shared_ptr<BatchWrapper>> BWs;
    vector<shared_ptr<Layer>> LAYERS;

    vector<shared_ptr<BatchWrapper>> TestBWs;
    vector<shared_ptr<Layer>> TestLAYERS;

    bool isTrained = false;
    bool isTested = false;

    void Input(int b, int c, int h, int w, int miniBatchSize);
    void Conv2D(int out_channels, int in_channels, int kernel_h, int kernel_w, int stride, int padding);
    void Activation(const string& type);
    void Pooling(int kHeight, int kWidth, int stride, int padding, const string& type);
    void BatchNorm();
    void Dropout(float p);
    void Dense(int NumFeatures);

private:
    shared_ptr<OutputLayer> OutputprivateLayer(const vector<vector<vector<vector<float>>>>& miniBatch);

public:
    void Train(const vector<vector<vector<vector<float>>>>& inputs, const vector<vector<vector<vector<float>>>>& labels, 
        int epochs, float lr, float R2, float clipGrad);

    void Test(const vector<vector<vector<vector<float>>>>& tensor, const vector<vector<vector<vector<float>>>>& labels);
    vector<int> Predict(const vector<vector<vector<vector<float>>>>& tensor);
    void showEvaluation();


private:
    vector<int> TP, TN, FP, FN;

    vector<vector<vector<vector<float>>>> getMiniBatch(const vector<vector<vector<vector<float>>>>& batch, int idx) const;

    void calculateTP_FP_FN_TN(vector<float> matrix, int C);
    void Evaluate();
};