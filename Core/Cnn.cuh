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

	Conv2dLayer(const std::vector<size_t>& InputShape, const std::vector<size_t>& KernelShape, size_t s, size_t p)
		: Stride(s), Padding(p) {

		Kernel = Tensor<T>(KernelShape);
		Kernel.fillRandomOptimized(1234ULL);

		size_t Hout = (InputShape[2] + 2 * Padding - KernelShape[2]) / Stride + 1;
		size_t Wout = (InputShape[3] + 2 * Padding - KernelShape[3]) / Stride + 1;
		this->OutputShape = { InputShape[0], KernelShape[0], Hout, Wout };
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
		dim3 blocks((OW + TILE_DIM - 1) / TILE_DIM, (OH + TILE_DIM - 1) / TILE_DIM, N * K);

		switch (KH) {
		case 3:
			ARCH::ExecuteKernel("Conv2D Kernel Launch", blocks, threads, SharedMem, 0,
				KERNEL::TiledConv2dKernelF<T, TILE_DIM, BLOCK_ROWS, 3>, Input.data(), Kernel.data(), Output.data(),
				N, C, H, W, K, KH, KW, Stride, Padding, OH, OW);
			break;

		case 5:
			ARCH::ExecuteKernel("Conv2D Kernel Launch", blocks, threads, SharedMem, 0,
				KERNEL::TiledConv2dKernelF<T, TILE_DIM, BLOCK_ROWS, 5>, Input.data(), Kernel.data(), Output.data(),
				N, C, H, W, K, KH, KW, Stride, Padding, OH, OW);
			break;

		case 7:
			ARCH::ExecuteKernel("Conv2D Kernel Launch", blocks, threads, SharedMem, 0,
				KERNEL::TiledConv2dKernelF<T, TILE_DIM, BLOCK_ROWS, 7>, Input.data(), Kernel.data(), Output.data(),
				N, C, H, W, K, KH, KW, Stride, Padding, OH, OW);
			break;

		default:
			throw std::runtime_error("Unsupported kernel size");
		}
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

		size_t n = Input.numel();
		dim3 threads(256);
		dim3 blocks((n + 255) / 256);

		if (Type == "relu") {
			ARCH::ExecuteKernel("relu", blocks, threads, 0, 0, KERNEL::ActivationReluF<T>, Input.data(), Output.data(), n);
		}
		else if (Type == "tanh") {
			ARCH::ExecuteKernel("tanh", blocks, threads, 0, 0, KERNEL::ActivationTanhF<T>, Input.data(), Output.data(), n);
		}
		else if (Type == "sigmoid") {
			ARCH::ExecuteKernel("sigmoid", blocks, threads, 0, 0, KERNEL::ActivationSigmoidF<T>, Input.data(), Output.data(), n);
		}
	}
};

template<FloatingTensorType T>
class PoolingLayer : public Layer<T> {
public:
	size_t kH, kW;
	size_t strideH, strideW;
	size_t padH, padW;
	std::string Type;

	PoolingLayer(const std::vector<size_t>& InputShape, size_t kh, size_t kw, size_t sH, size_t sW, size_t pH, size_t pW, const std::string& type)
		: kH(kh), kW(kw), strideH(sH), strideW(sW), padH(pH), padW(pW), Type(type) {

		if (type != "max" && type != "avg")
			throw std::runtime_error("Invalid pooling type");

		size_t Hin = InputShape[2];
		size_t Win = InputShape[3];

		this->OutputShape = { InputShape[0], InputShape[1], ((Hin - kH + 2 * padH) / strideH) + 1, ((Win - kW + 2 * padW) / strideW) + 1 };
	}

	void forward(const Tensor<T>& Input, Tensor<T>& Output) override {
		this->InputTensor = Tensor<T>(Input.shape(), Input.data());

		int N = this->OutputShape[0];
		int C = this->OutputShape[1];
		int Hout = this->OutputShape[2];
		int Wout = this->OutputShape[3];

		int H = Input.shape()[2];
		int W = Input.shape()[3];

		dim3 threads(1, 1, 1);
		dim3 blocks(Hout * Wout, C, N);

		if (Type == "avg")
		{
			ARCH::ExecuteKernel("AvgPool2D", blocks, threads, 0, 0, KERNEL::AvgPool2D<T>, Input.data(), Output.data(),
				N, C, H, W, Hout, Wout, kH, kW, strideH, strideW, padH, padW );
		}
		else {
			ARCH::ExecuteKernel("MaxPool2D", blocks, threads, 0, 0, KERNEL::MaxPool2D<T>, Input.data(), Output.data(),
				N, C, H, W, Hout, Wout, kH, kW, strideH, strideW, padH, padW );
		}
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

		dim3 threads(16, 16);
		dim3 blocks((OutNumFeature + threads.x - 1) / threads.x,
			(batch_size + threads.y - 1) / threads.y);

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
		if (classDim == 2) Type = "sigmoid";
		else if (classDim > 2) Type = "softmax";
		else throw std::runtime_error("Invalid output shape for output layer");
	}

	void forward(const Tensor<T>& Input, Tensor<T>& Output) override {
		this->InputTensor = Tensor<T>(Input.shape(), Input.data());

		int batch_size = Input.shape()[0];
		int classDim = Input.shape()[3];
		int total_elements = batch_size * classDim;

		dim3 threads(256);
		dim3 blocks((total_elements + threads.x - 1) / threads.x);

		if (Type == "sigmoid") {
			ARCH::ExecuteKernel("Sigmoid", blocks, threads, 0, 0, KERNEL::SigmoidKernelF<T>,
				Input.data(), Output.data(), total_elements);
		}
		else if (Type == "softmax") {
			blocks = dim3((batch_size + threads.x - 1) / threads.x);
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
			if (InputShape.empty())
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

	void MaxPooling(size_t kh, size_t kw, size_t sh = 1, size_t sw = 1, size_t ph = 0, size_t pw = 0) {
		const auto& CurrentShape = getCurrentOutputShape();
		Layers.push_back(std::make_unique<PoolingLayer<T>>(CurrentShape, kh, kw, sh, sw, ph, pw, "max"));
	}

	void AvgPooling(size_t kh, size_t kw, size_t sh = 1, size_t sw = 1, size_t ph = 0, size_t pw = 0) {
		const auto& CurrentShape = getCurrentOutputShape();
		Layers.push_back(std::make_unique<PoolingLayer<T>>(CurrentShape, kh, kw, sh, sw, ph, pw, "avg"));
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
		Tensor<T>& Predicted = forwardPassOptimized(Input);
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
