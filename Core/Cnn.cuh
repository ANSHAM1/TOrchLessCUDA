#ifndef __CNN_ARCH__
#define __CNN_ARCH__

#include "Tensor.cuh"

template<typename T>
concept FloatingTensorType = std::same_as<T, float> || std::same_as<T, double> || std::same_as<T, __half>;

_AM_START

template<FloatingTensorType T>
class Layer {
protected:
    Tensor<T> InputTensor;
    std::vector<size_t> OutputShape;

public:
    virtual void forward(const Tensor<T>& Input, Tensor<T>& Output) = 0;

    //virtual void backward(const Tensor<T>& UpstreamGrad, Tensor<T>& DownstreamGrad) = 0;

    virtual const std::vector<size_t>& getOutputShape() const {
        return OutputShape;
    }

    virtual ~Layer() {}
};

template<FloatingTensorType T>
class Conv2dLayer : public Layer<T> {
public:
    Tensor<T> Kernel;
    size_t Stride, Padding;

    Conv2dLayer(const std::vector<size_t>& InputShape, std::vector<size_t>& KernelShape, size_t s, size_t p)
        : Stride(s), Padding(p) {

        Kernel = Tensor<T>(KernelShape);
        Kernel.fillRandomOptimized(1234ULL);

        size_t Hout = (InputShape[2] + 2 * Padding - kH) / Stride + 1;
        size_t Wout = (InputShape[3] + 2 * Padding - kW) / Stride + 1;
        this->OutputShape = { InputShape[0], Cout, Hout, Wout };
    }

    void forward(const Tensor<T>& Input, Tensor<T>& Output) override {
        this->InputTensor = Tensor<T>(Input.shape(), Input.data());

        const auto& I_Shape = Input.shape();
        const auto& K_Shape = Kernel.shape();
        const auto& O_Shape = Output.shape();

        const size_t N = I_Shape[0], C = I_Shape[1], H = I_Shape[2], W = I_Shape[3];
        const size_t K = K_Shape[0], KH = K_Shape[2], KW = K_Shape[3];
        const size_t OH = O_Shape[2], OW = O_Shape[3];

        const size_t TILE_DIM = 16;
        const size_t BLOCK_ROWS = 16;

        const int PADDED_TILE_DIM = (TILE_DIM - 1) * Stride + KW;
        size_t SharedMem = (PADDED_TILE_DIM * PADDED_TILE_DIM + KW * KH) * sizeof(T);

        dim3 threads(TILE_DIM, BLOCK_ROWS);
        dim3 blocks(
            (OW + TILE_DIM - 1) / TILE_DIM,
            (OH + TILE_DIM - 1) / TILE_DIM,
            N * K
        );

        // --- Kernel Specialization: The reason for the if/else block ---
        // We must call a specific template instantiation with a COMPILE-TIME constant (e.g., 5 or 3).
        // This allows the compiler to perform critical optimizations like loop unrolling,
        // which is why we have separate versions for common kernel sizes.
        size_t KERNEL_TILE_DIM = 0;
        if (KH == 3 && KW == 3)
            KERNEL_TILE_DIM = 3;
        else if (KH == 5 && KW == 5)
            KERNEL_TILE_DIM = 5;
        else if (KH == 7 && KW == 7)
            KERNEL_TILE_DIM = 7;
        else
            throw std::runtime_error("No optimized tiled kernel available for this kernel size.");

        ARCH::ExecuteKernel("Conv2D Kernel Launch", blocks, threads, SharedMem, 0,
            KERNEL::TiledConv2dKernelF<T, TILE_DIM, BLOCK_ROWS, KERNEL_TILE_DIM>, Input.data(), Kernel.data(), Output.data(),
            N, C, H, W, K, KH, KW, Stride, Padding, OH, OW);
    }
};

template<FloatingTensorType T>
class ActivationLayer : public Layer<T> {
public:
    std::string Type;

    ActivationLayer(const std::vector<size_t>& InputShape, const std::string& type)
        : Type(type) {

        if (Type != "relu" && Type != "tanh" && Type != "sigmoid")
            throw std::runtime_error("Not a valid activation function");
        this->OutputShape = InputShape;
    }

    void forward(const Tensor<T>& Input, Tensor<T>& Output) override {
        this->InputTensor = Tensor<T>(Input.shape(), Input.data());

        size_t size = Input.numel();
        if (size == 0) return;

        dim3 blocks((size + 255) / 256);
        dim3 threads(256);

        ARCH::ExecuteKernel("Activation", blocks, threads, 0, 0, KERNEL::ActivationKernelF<T, Type>, 
            Input.data(), Output.data(), size);
    }
};

template<FloatingTensorType T>
class PoolingLayer : public Layer<T> {
public:
    size_t kH, kW, Stride, Padding;
    std::string Type;

    PoolingLayer(const std::vector<size_t>& InputShape, size_t kh, size_t kw, size_t s, size_t p, const std::string& type)
        : kH(kh), kW(kw), Stride(s), Padding(p), Type(type) {

        if (type != "max" && type != "avg")
            throw std::runtime_error("invalid pooling type");

        size_t Hout = ((InputShape[2] - kH + 2 * Padding) / Stride) + 1;
        size_t Wout = ((InputShape[3] - kW + 2 * Padding) / Stride) + 1;
        this->OutputShape = { InputShape[0], InputShape[1], Hout, Wout };
    }

    void forward(const Tensor<T>& Input, Tensor<T>& Output) override {
        this->InputTensor = Tensor<T>(InputShape, Input.data());

        int Hout = this->OutputShape[2];
        int Wout = this->OutputShape[3];

        dim3 threads(16, 16);
        dim3 blocks((Wout + block.x - 1) / block.x, this->OutputShape[1], this->OutputShape[0]);

        if(Type == "avg")
            ARCH::ExecuteKernel("avg Pooling", blocks, threads, 0, 0, KERNEL::AvgPool2DKernelF<T>, Input.data(), Output.data(),
                this->OutputShape[0], this->OutputShape[1], Input.shape()[2], Input.shape()[3],
                Hout, Wout, kH, kW, Stride, Padding);
		else
            ARCH::ExecuteKernel("max Pooling", blocks, threads, 0, 0, KERNEL::MaxPool2DKernelF<T>, Input.data(), Output.data(),
                this->OutputShape[0], this->OutputShape[1], Input.shape()[2], Input.shape()[3],
                Hout, Wout, kH, kW, Stride, Padding);
    }
};

template<FloatingTensorType T>
class DenseLayer : public Layer<T> {
public:
    Tensor<T> Input, Weight, Bias;
    size_t OutNumFeature;

    DenseLayer(const std::vector<size_t>& InputShape, size_t outNumFeature)
        : OutNumFeature(outNumFeature) {

        size_t InNumFeature = shape_product(InputShape) / InputShape[0];

        Weight = Tensor<T>({ OutNumFeature, InNumFeature, 1, 1 });
        Bias.zeros({ 1, 1, 1, OutNumFeature });

        this->OutputShape = { InputShape[0], 1, 1, OutNumFeature };
	}

    void forward(const Tensor<T>& Input, Tensor<T>& Output) override {
        this->InputTensor = Tensor<T>(Input.shape(), Input.data());

        int batch_size = Input.shape()[0];
        int InNumFeature = shape_product(Input.shape()) / batch_size;
        int OutNumFeature = DenseLayer.OutNumFeature;

        dim3 threads(16, 16);
        dim3 blocks((OutNumFeature + blockSize.x - 1) / blockSize.x,
            (batch_size + blockSize.y - 1) / blockSize.y);

		ARCH::ExecuteKernel("Dense Layer", blocks, threads, 0, 0, KERNEL::DenseKernelF<T>, Input.data(), Weight.data(),
            Bias.data(), Output.data(), batch_size, InNumFeature, OutNumFeature);
    }
};

template<FloatingTensorType T>
class OutputLayer : public Layer<T> {
public:
    Tensor<T> Input, Label, Loss;
	std::string Type;

    OutputLayer(const std::vector<size_t>& InputShape) {
        this->OutputShape = InputShape;
        
		size_t classDim = InputShape[3];
        if(classDim == 2) Type = "sigmoid";
		else if(classDim > 2) Type = "softmax";
		else throw std::runtime_error("Invalid output shape for output layer");
    }

    void forward(const Tensor<T>& Input, Tensor<T>& Output) override {
        this->InputTensor = Tensor<T>(Input.shape(), Input.data());
        
        int batch_size = Input.shape()[0];
        int classDim = Input.shape()[3];
        int total_elements = batch_size * classDim;

        dim3 threads(256);
        dim3 blocks((total_elements + blockSize.x - 1) / blockSize.x);

        if (Type == "sigmoid") {
            ARCH::ExecuteKernel("Sigmoid", blocks, threads, 0, 0, KERNEL::SigmoidKernelF<T>, 
                Input.data(), Output.data(), total_elements);
        }
        else if (Type == "softmax") {
            blocks = dim3((batch_size + blockSize.x - 1) / blockSize.x);
            ARCH::ExecuteKernel("Softmax", blocks, threads, 0, 0, KERNEL::SoftmaxKernelF<T>, 
				Input.data(), Output.data(), batch_size, classDim);
        }
        else {
            throw std::runtime_error("Unsupported output type");
        }

    }
};


template<FloatingTensorType T>
class Sequential {
private:
    std::vector<size_t> InputShape;
    std::vector<std::unique_ptr<Layer<T>>> Layers;

    size_t NumBatch;

    Tensor<T> workspace_A_;
    Tensor<T> workspace_B_;
    bool is_built_ = false;

    const std::vector<size_t>& getCurrentOutputShape() {
        if (Layers.empty()) {
            if(InputShape.empty()) 
                throw std::runtime_error("Input shape not set. Call Input() before adding layers.");
            return InputShape;
        }
        return Layers.back()->getOutputShape();
    }

    void Compile() {
        if (InputShape.empty()) {
            throw std::runtime_error("Input shape not set. Call Input() before build().");
        }
        if (Layers.empty()) {
            return;
        }
        if (is_built_) {
            return;
        }

		is_built_ = true;

        size_t max_elements_A = 0;
        size_t max_elements_B = 0;

        max_elements_A = shape_product(InputShape);

        std::vector<size_t> current_shape = InputShape;

        for (size_t i = 0; i < Layers.size(); ++i) {
            const auto& OutputShape = Layers[i]->getOutputShape();
            size_t OutputElements = shape_product(OutputShape);

            if (i % 2 == 0) { // Writes to B
                if (OutputElements > max_elements_B) max_elements_B = OutputElements;
            }
            else { // Writes to A
                if (OutputElements > max_elements_A) max_elements_A = OutputElements;
            }
            current_shape = OutputShape;
        }

        workspace_A_.resize({ max_elements_A });
        workspace_B_.resize({ max_elements_B });

        is_built_ = true;
        std::cout << "Model built successfully." << std::endl;
        std::cout << "Workspace A size: " << (max_elements_A * sizeof(T)) / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "Workspace B size: " << (max_elements_B * sizeof(T)) / (1024.0 * 1024.0) << " MB" << std::endl;
    }

    Tensor<T>& forwardPassOptimized(const Tensor<T>& Input) {
        if (!is_built_) {
            throw std::runtime_error("Model not built. Call Compile() first.");
        }
        if (Input.shape() != InputShape) {
            throw std::invalid_argument("Input tensor shape does not match model's expected input shape.");
        }

        const Tensor<T>* source = &Input;
        Tensor<T>* destination = &workspace_B_;

        for (size_t i = 0; i < Layers.size(); ++i) {
            if (i % 2 == 0) {
                source = (i == 0) ? &Input : &workspace_A_;
                destination = &workspace_B_;
            }
            else {
                source = &workspace_B_;
                destination = &workspace_A_;
            }

            destination->resize(Layers[i]->getOutputShape());

            Layers[i]->forward(*source, *destination);
        }

        if ((Layers.size() - 1) % 2 == 0) {
            return workspace_B_;
        }
        else {
            return workspace_A_;
        }
    }

public:
    Sequential() = default;

    void Input(const std::vector<size_t>& Input_Shape, size_t batchSize) {
        if (Input_Shape.size() != 4) {
            throw std::invalid_argument("Input shape must be of size 4 -> (N, C, H, W)");
        }
        InputShape = { batchSize, Input_Shape[1], Input_Shape[2], Input_Shape[3] };
        NumBatch = Input_Shape[0] / batchSize;
    }

    void Conv2D(const std::vector<size_t>& Kernel_Shape, size_t s = 1, size_t p = 0) {
        if (Kernel_Shape.size() != 4) {
            throw std::invalid_argument("Input shape must be of size 4 -> (N, C, H, W)");
        }
        const auto& CurrentShape = getCurrentOutputShape();
        Layers.push_back(std::make_unique<Conv2dLayer<T>>(CurrentShape, Kernel_Shape, s, p));
    }

    void Activate(const std::string& type) {
        const auto& CurrentShape = getCurrentOutputShape();
        Layers.push_back(std::make_unique<ActivationLayer<T>>(CurrentShape, type));
    }

    void MaxPooling(size_t kh, size_t kw, size_t s = 1, size_t p = 0) {
        const auto& CurrentShape = getCurrentOutputShape();
        Layers.push_back(std::make_unique<PoolingLayer<T>>(CurrentShape, kh, kw, s, p, "max"));
	}

    void AvgPooling(size_t kh, size_t kw, size_t s = 1, size_t p = 0) {
        const auto& CurrentShape = getCurrentOutputShape();
        Layers.push_back(std::make_unique<PoolingLayer<T>>(CurrentShape, kh, kw, s, p, "avg"));
    }

    void Dense(int outNumFeature) {
        const auto& CurrentShape = getCurrentOutputShape();
        Layers.push_back(std::make_unique<DenseLayer<T>>(CurrentShape, outNumFeature));
	}

    void Output() {
        const auto& CurrentShape = getCurrentOutputShape();
        Layers.push_back(std::make_unique<OutputLayer<T>>(CurrentShape));
	}

    void Predict(const Tensor<T>& Input) {
        Compile();
        auto Predicted = forwardPassOptimized(Input);
    }

    void Testing(const Tensor<T>& Input, const Tensor<T>& Label) {
        Compile();

    }




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
