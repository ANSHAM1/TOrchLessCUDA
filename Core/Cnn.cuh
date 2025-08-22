#ifndef __CNN_ARCH__
#define __CNN_ARCH__

#include "Tensor.cuh"

template<typename T>
concept FloatingTensorType = std::same_as<T, float> || std::same_as<T, double> || std::same_as<T, __half>;

_AM_START

template<FloatingTensorType T>
class Layer {
public:
    virtual void forward(const Tensor<T>& Shape) = 0;
    //virtual Tensor<T> retrieveOutputs() const = 0;
    bool SkipInInference = false;

    virtual ~Layer() {}
};

template<FloatingTensorType T>
class Conv2dLayer : public Layer<T> {
public:
    Tensor<T> Kernel;

    Tensor<T> Output;
    size_t Stride, Padding;

    Conv2dLayer(const std::vector<size_t>& InputShape, size_t Cout, size_t Cin, size_t kH, size_t kW, size_t s, size_t p) 
        : Stride(s), Padding(p) {

        Kernel = Tensor<T>({ Cout, Cin, kH, kW });
        Kernel.fillRandomOptimized(1234ULL);

        size_t Hout = (InputShape[2] + 2 * Padding - kH) / Stride + 1;
        size_t Wout = (InputShape[3] + 2 * Padding - kW) / Stride + 1;
        Output = Tensor<T>({ InputShape[0], Cout, Hout, Wout });
    }

    void forward(const Tensor<T>& Tensor) override {};
    //Tensor<T> retrieveOutputs() const override { return Output; }
};

template<FloatingTensorType T>
class ActivationLayer : public Layer<T> {
public:
    Tensor<T> Output;
    std::string Type;

    ActivationLayer(const std::vector<size_t>& InputShape, const std::string& type) 
        : Type(type) {

        if (Type != "relu" && Type != "tanh" && Type != "sigmoid") throw std::runtime_error("not a valid activation function");
		Output = Tensor<T>(InputShape);
    }

    void forward(const Tensor<T>& Tensor) override {};
    //Tensor<T> retrieveOutputs() const override { return Output; }
};

template<FloatingTensorType T>
class PoolingLayer : public Layer<T> {
public:
    size_t kH, kW, S, P;

    Tensor<T> Output;
    std::string Type;

    PoolingLayer(const std::vector<size_t>& InputShape, int kh, int kw, int s, int p, const std::string& type) 
        : kH(kh), kW(kw), S(s), P(p), Type(type) {

        if (type != "max" && type != "avg" && type != "min") throw std::runtime_error("invalid pooling type");

        int Hout = ((InputShape[2] - kH + 2 * P) / S) + 1;
        int Wout = ((InputShape[3] - kW + 2 * P) / S) + 1;

        Output.zeros({ InputShape[0], InputShape[1], Hout, Wout });
    }

    void forward(const Tensor<T>& Tensor) override {};
    //Tensor<T> retrieveOutputs() const override { return Output; }
};

template<FloatingTensorType T>
class BatchNormLayer : public Layer<T> {
public:
    Tensor<T> Gamma;
    Tensor<T> Beta;

    Tensor<T> Output;

    BatchNormLayer(const std::vector<size_t>& InputShape) {
        Gamma = Tensor<T>({ 1, InputShape[1], 1, 1 });
        Gamma.fill((T)1);
        Beta.zeros({ 1, InputShape[1], 1, 1 });
        Output = Tensor<T>(InputShape);
    }

    void forward(const Tensor<T>& Tensor) override {};
    //Tensor<T> retrieveOutputs() const override { return Output; }
};

template<FloatingTensorType T>
class DropoutLayer : public Layer<T> {
public:
    Tensor<T> Mask;

    Tensor<T> Output;
    T Prob;

    DropoutLayer(const std::vector<size_t>& InputShape, float p)
        : Prob(p) {

        Mask.zeros(InputShape);
        Output = Tensor<T>(InputShape);
    }

    void forward(const Tensor<T>& Tensor) override {};
    //Tensor<T> retrieveOutputs() const override { return Output; }
};

template<FloatingTensorType T>
class DenseLayer : public Layer<T> {
public:
    Tensor<T> Weight;
    Tensor<T> Bias;

    Tensor<T> Output;
    size_t OutNumFeature;

    DenseLayer(const std::vector<size_t>& InputShape, int outNumFeature)
        : OutNumFeature(outNumFeature) {

        int InNumFeature = shape_product(InputShape) / InputShape[0];

        Weight = Tensor<T>({ OutNumFeature, InNumFeature, 1, 1 });
        Bias.zeros({ 1, 1, 1, OutNumFeature });
        Output = Tensor<T>({ InputShape[0], 1, 1, OutNumFeature });
	}

    void forward(const Tensor<T>& Tensor) override {};
    //Tensor<T> retrieveOutputs() const override { return Output; }
};

template<FloatingTensorType T>
class OutputLayer : public Layer<T> {
public:
    Tensor<T> Output;

    OutputLayer(const std::vector<size_t>& InputShape) {
        Output = Tensor<T>(InputShape);
    }

    void forward(const Tensor<T>& Tensor) override {};
    //Tensor<T> retrieveOutputs() const override { return Output; }
};

    
template<FloatingTensorType T>
class Sequential {
private:
	std::vector<size_t> InputShape;
    std::vector<std::unique_ptr<Layer<T>>> Layers;

public:
    Sequential() = default;

    void Input(const std::vector<size_t>& inputShape) {
        if (inputShape.size() != 4) {
            throw std::invalid_argument("Input shape must be of size 4 -> (N, C, H, W)");
        }
        InputShape = inputShape;
	}

    //void Conv2D(size_t Cout, size_t Cin, size_t kH, size_t kW, size_t s = 1, size_t p = 0) {
    //    if (Layers.empty()) {
    //        Layers.push_back(std::make_unique<Conv2dLayer<T>>(InputShape, Cout, Cin, kH, kW, s, p));
    //    }

    //    Layers.push_back(
    //        std::make_unique<Conv2dLayer<T>>(InputShape, Cout, Cin, kH, kW, s, p)
    //    );
    //}

    //void Activate(const std::string& type) {
    //    Layers.push_back(
    //        std::make_unique<ActivationLayer<T>>(InputShape, type)
    //    );
    //}

    //void Pooling(const std::vector<size_t>& InputShape, int kh, int kw, int s, int p, const std::string& type) {
    //    Layers.push_back(
    //        std::make_unique<PoolingLayer<T>>(InputShape, kh, kw, s, p, type)
    //    );
    //}

 //   void BatchNorm(const std::vector<size_t>& InputShape) {
 //       Layers.push_back(
 //           std::make_unique<BatchNormLayer<T>>(InputShape)
 //       );
 //   }

 //   void Dropout(const std::vector<size_t>& InputShape, float p) {
 //       Layers.push_back(
 //           std::make_unique<DropoutLayer<T>>(InputShape, p)
	//	);
 //   }

 //   void Dense(const std::vector<size_t>& InputShape, int outNumFeature) {
 //       Layers.push_back(
 //           std::make_unique<DenseLayer<T>>(InputShape, outNumFeature)
 //       );
	//}



//    shared_ptr<BatchWrapper> CONFUSION_MATRIX;
//    float ACCURACY;
//    vector<float> PRECISION, RECALL, F1;
//
//    int NUM_BATCHS;
//
//    vector<shared_ptr<BatchWrapper>> BWs;
//    vector<shared_ptr<Layer>> LAYERS;
//
//    vector<shared_ptr<BatchWrapper>> TestBWs;
//    vector<shared_ptr<Layer>> TestLAYERS;
//
//    bool isTrained = false;
//    bool isTested = false;
//
//    void Input(int b, int c, int h, int w, int miniBatchSize);
//    void Conv2D(int out_channels, int in_channels, int kernel_h, int kernel_w, int stride, int padding);
//    void Activation(const string& type);
//    void Pooling(int kHeight, int kWidth, int stride, int padding, const string& type);
//    void BatchNorm();
//    void Dropout(float p);
//    void Dense(int NumFeatures);
//
//private:
//    shared_ptr<OutputLayer> OutputprivateLayer(const vector<vector<vector<vector<float>>>>& miniBatch);
//
//public:
//    void Train(const vector<vector<vector<vector<float>>>>& inputs, const vector<vector<vector<vector<float>>>>& labels,
//        int epochs, float lr, float R2, float clipGrad);
//
//    void Test(const vector<vector<vector<vector<float>>>>& tensor, const vector<vector<vector<vector<float>>>>& labels);
//    vector<int> Predict(const vector<vector<vector<vector<float>>>>& tensor);
//    void showEvaluation();
//
//
//private:
//    vector<int> TP, TN, FP, FN;
//
//    vector<vector<vector<vector<float>>>> getMiniBatch(const vector<vector<vector<vector<float>>>>& batch, int idx) const;
//
//    void calculateTP_FP_FN_TN(vector<float> matrix, int C);
//    void Evaluate();
};


_AM_END

#endif // !__CNN_ARCH__
