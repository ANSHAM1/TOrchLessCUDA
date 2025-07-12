#include "../Includes/Cnn.h"

Conv2dLayer::Conv2dLayer(shared_ptr<BatchWrapper>& BWi, int Cout, int Cin, int kH, int kW, int s, int p)
	: S(s), P(p) {
	BWk = batchWrapperRandom(Cout, Cin, kH, kW);

	if (BWi->CHANNELs != BWk->CHANNELs)
		throw runtime_error("Conv2D: input channels do not match filter channels");

	int Hout = ((BWi->HEIGHTs - kH + 2 * P) / S) + 1;
	int Wout = ((BWi->WIDTHs - kW + 2 * P) / S) + 1;

	BWout = batchWrapperConst(BWi->BATCHs, Cout, Hout, Wout, 0.0f);
}

void Conv2dLayer::forward(shared_ptr<BatchWrapper>& BWi) {
	int size = BWout->SIZE;
	int threads = 256;
	int blocks = (size + threads - 1) / threads;

	Convo2DKernelForwardWrapper(BWout->deviceRaw_Ptr, BWi->deviceRaw_Ptr,
		BWk->deviceRaw_Ptr, S, P, blocks, threads);

	BWout->backward = [BWi, this](float lr, float r2, float gradclip) {
		int size = BWi->SIZE;
		int threads = 256;
		int blocks = (size + threads - 1) / threads;

		Convo2DKernelBackwardWrapper(BWi->deviceRaw_Ptr, this->BWout->deviceRaw_Ptr,
			this->BWk->deviceRaw_Ptr, this->S, this->P, r2, gradclip, blocks, threads);

		int blocksK = (this->BWk->SIZE + threads - 1) / threads;
		kernelsUpdateWrapper(this->BWk->deviceRaw_Ptr, lr, blocksK, threads);

		if(BWi->backward) BWi->backward(lr, r2, gradclip);

		BWk->resetGrad();
		BWout->resetData();
		BWout->resetGrad();
		};
}


ActivationLayer::ActivationLayer(shared_ptr<BatchWrapper>& BWi, const string& type) {
	if (type != "relu" && type != "tanh" && type != "sigmoid") throw runtime_error("not a valid activation function");
	TYPE = type;

	BWout = batchWrapperConst(BWi->BATCHs, BWi->CHANNELs, BWi->HEIGHTs, BWi->WIDTHs, 0.0f);
};

void ActivationLayer::forward(shared_ptr<BatchWrapper>& BWi) {
	int size = BWout->SIZE;
	int threads = 256;
	int blocks = (size + threads - 1) / threads;
	ActivationKernelForwardWrapper(BWout->deviceRaw_Ptr, BWi->deviceRaw_Ptr, TYPE, blocks, threads);

	BWout->backward = [BWi, this, blocks, threads](float lr, float r2, float gradclip) {
		ActivationKernelBackwardWrapper(BWi->deviceRaw_Ptr,
			this->BWout->deviceRaw_Ptr, this->TYPE, blocks, threads);

		if (BWi->backward) BWi->backward(lr, r2, gradclip);

		BWout->resetData();
		BWout->resetGrad();
		};
}


PoolingLayer::PoolingLayer(shared_ptr<BatchWrapper>& BWi, int kh, int kw, int s, int p, const string& type)
	: kH(kh), kW(kw), P(p), S(s), TYPE(type) {
	if (type != "max" && type != "avg" && type != "min") throw runtime_error("invalid pooling type");

	int Hout = ((BWi->HEIGHTs - kH + 2 * P) / S) + 1;
	int Wout = ((BWi->WIDTHs - kW + 2 * P) / S) + 1;

	BWout = batchWrapperConst(BWi->BATCHs, BWi->CHANNELs, Hout, Wout, 0.0f);
};

void PoolingLayer::forward(shared_ptr<BatchWrapper>& BWi) {
	int size = BWout->SIZE;
	int threads = 256;
	int blocks = (size + threads - 1) / threads;
	PoolingKernelForwardWrapper(BWout->deviceRaw_Ptr, BWi->deviceRaw_Ptr, TYPE,
		kH, kW, S, P, blocks, threads);

	BWout->backward = [BWi, this](float lr, float r2, float gradclip) {
		int size = BWi->SIZE;
		int threads = 256;
		int blocks = (size + threads - 1) / threads;
		PoolingKernelBackwardWrapper(BWi->deviceRaw_Ptr, BWout->deviceRaw_Ptr, this->TYPE,
			this->kH, this->kW, this->S, this->P, blocks, threads);

		if (BWi->backward) BWi->backward(lr, r2, gradclip);

		BWout->resetData();
		BWout->resetGrad();
		};
}


BatchNormLayer::BatchNormLayer(shared_ptr<BatchWrapper>& BWi) {
	GAMMA = batchWrapperConst(1, BWi->CHANNELs, 1, 1, 1.0f);
	BETA = batchWrapperConst(1, BWi->CHANNELs, 1, 1, 0.0f);
	BWout = batchWrapperConst(BWi->BATCHs, BWi->CHANNELs, BWi->HEIGHTs, BWi->WIDTHs, 0.0f);
};

void BatchNormLayer::forward(shared_ptr<BatchWrapper>& BWi) {
	int sizePerChannel = BWi->BATCHs * BWi->HEIGHTs * BWi->WIDTHs;

	int threads = 256;
	int blocks = (sizePerChannel + threads - 1) / threads;
	int threadsCH = 256;
	int blocksCH = (BWi->CHANNELs + threadsCH - 1) / threadsCH;

	float* mean, * var, * sumDy, * sumDyXmu;
	cudaCheck(cudaMalloc(&mean, BWi->CHANNELs * sizeof(float)));
	cudaCheck(cudaMalloc(&var, BWi->CHANNELs * sizeof(float)));
	cudaCheck(cudaMalloc(&sumDy, BWi->CHANNELs * sizeof(float)));
	cudaCheck(cudaMalloc(&sumDyXmu, BWi->CHANNELs * sizeof(float)));

	cudaCheck(cudaMemset(mean, 0, BWi->CHANNELs * sizeof(float)));
	cudaCheck(cudaMemset(var, 0, BWi->CHANNELs * sizeof(float)));
	cudaCheck(cudaMemset(sumDy, 0, BWi->CHANNELs * sizeof(float)));
	cudaCheck(cudaMemset(sumDyXmu, 0, BWi->CHANNELs * sizeof(float)));

	BatchNormKernelForwardWrapper(BWout->deviceRaw_Ptr, BWi->deviceRaw_Ptr, mean, var,
		GAMMA->deviceRaw_Ptr, BETA->deviceRaw_Ptr,
		BWi->SIZE, sizePerChannel, BWi->CHANNELs, blocks, threads, blocksCH, threadsCH);

	BWout->backward = [=](float lr, float r2, float gradclip) {
		int blocksCH = (BWi->CHANNELs + threads - 1) / threads;

		BatchNormKernelBackwardWrapper(BWi->deviceRaw_Ptr, this->BWout->deviceRaw_Ptr, mean, var,
			sumDy, sumDyXmu, GAMMA->deviceRaw_Ptr, BETA->deviceRaw_Ptr,
			BWi->SIZE, sizePerChannel, BWi->CHANNELs, r2, gradclip, blocks, threads);


		cudaCheck(cudaFree(mean)); cudaCheck(cudaFree(var));
		cudaCheck(cudaFree(sumDy)); cudaCheck(cudaFree(sumDyXmu));

		int blocksGB = (this->GAMMA->SIZE + threads - 1) / threads;
		gammabetaUpdateWrapper(GAMMA->deviceRaw_Ptr, BETA->deviceRaw_Ptr, lr, blocksGB, threads);

		if (BWi->backward) BWi->backward(lr, r2, gradclip);

		GAMMA->resetGrad();
		BETA->resetGrad();
		BWout->resetData();
		BWout->resetGrad();
		};
}


DropoutLayer::DropoutLayer(shared_ptr<BatchWrapper>& BWi, float p)
	: Prob(p) {
	isForTesting = false;
	BWmask = batchWrapperConst(BWi->BATCHs, BWi->CHANNELs, BWi->HEIGHTs, BWi->WIDTHs, 0.0f);
	BWout = batchWrapperConst(BWi->BATCHs, BWi->CHANNELs, BWi->HEIGHTs, BWi->WIDTHs, 0.0f);
};

void DropoutLayer::forward(shared_ptr<BatchWrapper>& BWi) {
	curandState* d_states;
	cudaCheck(cudaMalloc(&d_states, BWi->SIZE * sizeof(curandState)));

	int threads = 256;
	int blocks = (BWi->SIZE + threads - 1) / threads;

	DropoutKernelForwardWrapper(BWout->deviceRaw_Ptr, BWi->deviceRaw_Ptr, BWmask->deviceRaw_Ptr, Prob, BWi->SIZE, d_states, blocks, threads);

	BWout->backward = [BWi, this](float lr, float r2, float gradclip) {
		int size = this->BWout->SIZE;
		int threads = 256;
		int blocks = (size + threads - 1) / threads;
		DropoutKernelBackwardWrapper(BWi->deviceRaw_Ptr, this->BWout->deviceRaw_Ptr, 
			this->BWmask->deviceRaw_Ptr, this->Prob, blocks, threads);

		if (BWi->backward) BWi->backward(lr, r2, gradclip);

		BWout->resetData();
		BWout->resetGrad();
		};

	cudaCheck(cudaFree(d_states));
}


DenseLayer::DenseLayer(shared_ptr<BatchWrapper>& BWi, int NumFeatures)
	: FEATURES_COUNT(NumFeatures) {
	int in_features = BWi->CHANNELs * BWi->HEIGHTs * BWi->WIDTHs;

	WEIGHTS = batchWrapperRandom(NumFeatures, in_features, 1, 1);
	BIAS = batchWrapperConst(1, 1, 1, NumFeatures, 0.0f);
	BWout = batchWrapperConst(BWi->BATCHs, 1, 1, NumFeatures, 0.0f);
};

void DenseLayer::forward(shared_ptr<BatchWrapper>& BWi) {
	int total = BWi->BATCHs * FEATURES_COUNT;
	int threads = 256;
	int blocks = (total + threads - 1) / threads;

	DenseKernelForwardWrapper(BWout->deviceRaw_Ptr, BWi->deviceRaw_Ptr, WEIGHTS->deviceRaw_Ptr, 
		BIAS->deviceRaw_Ptr, blocks, threads);

	BWout->backward = [BWi, this](float lr, float r2, float gradclip) {
		int batch = BWi->BATCHs;
		int out_features = BWout->WIDTHs;
		int total = batch * out_features;

		int threads = 256;
		int blocks = (total + threads - 1) / threads;
		DenseKernelBackwardWrapper(BWi->deviceRaw_Ptr, this->BWout->deviceRaw_Ptr, 
			this->WEIGHTS->deviceRaw_Ptr, this->BIAS->deviceRaw_Ptr, r2, gradclip, blocks, threads);

		int blocksW = (this->WEIGHTS->SIZE + threads - 1) / threads;
		int blocksB = (this->BIAS->SIZE + threads - 1) / threads;
		weightbiasUpdateWrapper(this->WEIGHTS->deviceRaw_Ptr, this->BIAS->deviceRaw_Ptr,
			0.001f, blocksW, threads, blocksB, threads);

		if (BWi->backward) BWi->backward(lr, r2, gradclip);

		WEIGHTS->resetGrad();
		BIAS->resetGrad();
		BWout->resetData();
		BWout->resetGrad();
		};

}


OutputLayer::OutputLayer(shared_ptr<BatchWrapper>& BWi, shared_ptr<BatchWrapper>& labels)
	: LABELS(labels) {
	BWout = batchWrapperConst(BWi->BATCHs, BWi->CHANNELs, BWi->HEIGHTs, BWi->WIDTHs, 0.0f);
};

void OutputLayer::forward(shared_ptr<BatchWrapper>& BWi) {
	int total = BWi->BATCHs * BWi->WIDTHs;
	int threads = 256;
	int blocks = (total + threads - 1) / threads;

	OutputKernelWrapper(BWout->deviceRaw_Ptr, BWi->deviceRaw_Ptr, LABELS->deviceRaw_Ptr, blocks, threads);
	
	BWout->backward = [BWi, BWout = this->BWout](float lr, float r2, float gradclip) {
		int size = BWout->SIZE;
		int threads = 256;
		int blocks = (size + threads - 1) / threads;
		CopyGradWrapper(BWi->deviceRaw_Ptr, BWout->deviceRaw_Ptr, blocks, threads);

		if (BWi->backward) BWi->backward(lr, r2, gradclip);

		BWout->resetData();
		BWout->resetGrad();
		};
}

//_________________________________________________________________________________________

void Network::Input(int b, int c, int h, int w, int miniBatchSize = 600) {
	NUM_BATCHS = b / miniBatchSize;

	BWs.push_back(batchWrapperConst(miniBatchSize, c, h, w, 0.0f));
	TestBWs.push_back(batchWrapperConst(miniBatchSize, c, h, w, 0.0f));
}


void Network::Conv2D(int B, int C, int H, int W, int S, int P) {
	if (BWs.empty()) throw runtime_error("No input layer found");
	shared_ptr<BatchWrapper> BWi = BWs.back();
	auto L = make_shared<Conv2dLayer>(BWi, B, C, H, W, S, P);
	LAYERS.push_back(L);
	BWs.push_back(L->BWout);

	TestLAYERS.push_back(L);
	TestBWs.push_back(L->BWout);
};

void Network::Activation(const string& str) {
	if (BWs.empty()) throw runtime_error("No input layer found");
	shared_ptr<BatchWrapper> BWi = BWs.back();
	auto L = make_shared<ActivationLayer>(BWi, str);
	LAYERS.push_back(L);
	BWs.push_back(L->BWout);

	TestLAYERS.push_back(L);
	TestBWs.push_back(L->BWout);
};

void Network::Pooling(int kHeight, int kWidth, int stride, int padding, const string& str) {
	if (BWs.empty()) throw runtime_error("No input layer found");
	shared_ptr<BatchWrapper> BWi = BWs.back();
	auto L = make_shared<PoolingLayer>(BWi, kHeight, kWidth, stride, padding, str);
	LAYERS.push_back(L);
	BWs.push_back(L->BWout);

	TestLAYERS.push_back(L);
	TestBWs.push_back(L->BWout);
};

void Network::BatchNorm() {
	if (BWs.empty()) throw runtime_error("No input layer found");
	shared_ptr<BatchWrapper> BWi = BWs.back();
	auto L = make_shared<BatchNormLayer>(BWi);
	LAYERS.push_back(L);
	BWs.push_back(L->BWout);

	TestLAYERS.push_back(L);
	TestBWs.push_back(L->BWout);
};

void Network::Dropout(float p) {
	if (BWs.empty()) throw runtime_error("No input layer found");
	shared_ptr<BatchWrapper> BWi = BWs.back();
	auto L = make_shared<DropoutLayer>(BWi, p);
	LAYERS.push_back(L);
	BWs.push_back(L->BWout);

	//TestLAYERS.push_back(L); --> no need to push Dropout layer for testing.
	//TestBWs.push_back(L->BWout); --> no need to push dropout outputs as it is not required in testing phase.
};

void Network::Dense(int NumFeatures) {
	if (BWs.empty()) throw runtime_error("No input layer found");
	shared_ptr<BatchWrapper> BWi = BWs.back();
	auto L = make_shared<DenseLayer>(BWi, NumFeatures);
	LAYERS.push_back(L);
	BWs.push_back(L->BWout);

	TestLAYERS.push_back(L);
	TestBWs.push_back(L->BWout);
};

shared_ptr<OutputLayer> Network::OutputprivateLayer(const vector<vector<vector<vector<float>>>>& miniBatch) {
	if (BWs.empty()) throw runtime_error("No input layer found");

	shared_ptr<BatchWrapper> BWi = BWs.back();
	int B = miniBatch.size();
	int C = miniBatch[0][0][0].size();

	auto oneHot = batchWrapper(miniBatch);

	return make_shared<OutputLayer>(BWi, oneHot);
}

//___________________________________________________________________________________________________


void Network::Train(const vector<vector<vector<vector<float>>>>& inputs,
	const vector<vector<vector<vector<float>>>>& labels,
	int epochs, float lr, float r2, float gradclip) {

	isTrained = true;
	std::cout << "========== TRAINING STARTED ==========" << std::endl;

	const int totalSteps = NUM_BATCHS * LAYERS.size();

	for (int e = 0; e < epochs; e++) {
		std::cout << "\nEpoch " << (e + 1) << "/" << epochs << std::endl;

		for (int b = 0; b < NUM_BATCHS; b++) {
			BWs[0] = batchWrapper(getMiniBatch(inputs, b));
			auto miniLabels = getMiniBatch(labels, b);

			auto L = OutputprivateLayer(miniLabels);
			for (int l = 0; l < LAYERS.size(); l++) {
				LAYERS[l]->forward(BWs[l]);

				showProgressBar(b * LAYERS.size() + l + 1, totalSteps, 50);
			}
			L->forward(BWs.back());

			auto lastBW = L->getBWout();
			lastBW->backward(lr, r2, gradclip);

			BWs[0]->resetGrad();
		}

		std::cout << std::endl;
		std::cout << "Epoch " << (e + 1) << "/" << epochs << " completed." << std::endl;
	}

	std::cout << "\nTraining Completed Successfully!" << std::endl;
}



void Network::Test(const vector<vector<vector<vector<float>>>>& inputs, const vector<vector<vector<vector<float>>>>& labels) {
	if (isTrained) isTested = true;
	else throw runtime_error("Train the model First");
	int C = labels[0][0][0].size();

	for (int b = 0; b < NUM_BATCHS; b++) {
		TestBWs[0] = batchWrapper(getMiniBatch(inputs, b));
		auto miniLabels = getMiniBatch(labels, b);

		auto L = OutputprivateLayer(miniLabels);

		for (int l = 0; l < TestLAYERS.size(); l++) {
			TestLAYERS[l]->forward(TestBWs[l]);
		}

		L->forward(BWs.back());

		auto Outputs = L->getBWout();
		auto Labels = batchWrapper(labels);
		CONFUSION_MATRIX = batchWrapperConst(1, 1, C, C, 0.0f);

		int threads = 256;
		int blocks = (Outputs->BATCHs + threads - 1) / threads;
		EvaluateKernelWrapper(Outputs->deviceRaw_Ptr, Labels->deviceRaw_Ptr,
			CONFUSION_MATRIX->deviceRaw_Ptr, blocks, threads);
	}
}

vector<int> Network::Predict(const vector<vector<vector<vector<float>>>>& inputs) {
	if (!isTrained || !isTested) throw runtime_error("Model is either not Traied or not Tested");
	vector<int> predictedLabels;

	for (int b = 0; b < NUM_BATCHS; b++) {
		auto miniBatch = getMiniBatch(inputs, b);

		TestBWs[0] = batchWrapper(miniBatch);
		auto L = OutputprivateLayer(miniBatch);

		for (int l = 0; l < TestLAYERS.size(); l++) {
			TestLAYERS[l]->forward(TestBWs[l]);
		}

		L->forward(BWs.back());

		auto outputs = L->getBWout();
		vector<float> probabilities;
		int batchSize = outputs->BATCHs;
		int numClasses = outputs->WIDTHs;

		BatchDevice ptr;
		cudaMemcpy(&ptr, outputs->deviceRaw_Ptr, sizeof(BatchDevice), cudaMemcpyDeviceToHost);
		cudaMemcpy(probabilities.data(), ptr.DATA, ptr.SIZE * sizeof(float), cudaMemcpyDeviceToHost);

		for (int b = 0; b < batchSize; ++b) {
			float maxProb = -1.0f;
			int maxIndex = -1;

			for (int c = 0; c < numClasses; ++c) {
				float prob = probabilities[b * numClasses + c];
				if (prob > maxProb) {
					maxProb = prob;
					maxIndex = c;
				}
			}
			predictedLabels.push_back(maxIndex);
		}
	}

	return predictedLabels;
}

// -------------------------------------------------------------------------------------------------------------

vector<vector<vector<vector<float>>>> Network::getMiniBatch(const vector<vector<vector<vector<float>>>>& batch, int idx) const {
	int totalSamples = batch.size();
	int batchSize = totalSamples / NUM_BATCHS;

	int startIdx = idx * batchSize;
	int endIdx = startIdx + batchSize;

	if (endIdx > totalSamples) endIdx = totalSamples;

	return std::vector<std::vector<std::vector<std::vector<float>>>>(
		batch.begin() + startIdx,
		batch.begin() + endIdx
	);
}

void Network::calculateTP_FP_FN_TN(vector<float> matrix, int C) {
	TP.assign(C, 0);
	FP.assign(C, 0);
	FN.assign(C, 0);
	TN.assign(C, 0);

	float total = 0;
	for (int i = 0; i < C * C; ++i)
		total += matrix[i];

	for (int c = 0; c < C; ++c) {
		TP[c] = matrix[c * C + c];

		int rowSum = 0;
		int colSum = 0;
		for (int k = 0; k < C; ++k) {
			rowSum += matrix[c * C + k];
			colSum += matrix[k * C + c];
		}

		FP[c] = colSum - TP[c];
		FN[c] = rowSum - TP[c];
		TN[c] = total - (TP[c] + FP[c] + FN[c]);
	}
}


void Network::Evaluate() {
	BatchDevice matrix;
	cudaMemcpy(&matrix, CONFUSION_MATRIX->deviceRaw_Ptr, sizeof(BatchDevice), cudaMemcpyDeviceToHost);

	vector<float> hostConfusion(matrix.SIZE);
	cudaMemcpy(hostConfusion.data(), matrix.DATA, matrix.SIZE * sizeof(float), cudaMemcpyDeviceToHost);

	int C = CONFUSION_MATRIX->WIDTHs;

	calculateTP_FP_FN_TN(hostConfusion, C);

	PRECISION.resize(C, 0);
	RECALL.resize(C, 0);
	F1.resize(C, 0);

	int total = 0, correct = 0;
	for (int c = 0; c < C; ++c) {
		int TP = hostConfusion[c * C + c];
		int FP = 0, FN = 0;

		for (int i = 0; i < C; ++i) {
			if (i != c) {
				FP += hostConfusion[i * C + c]; 
				FN += hostConfusion[c * C + i];
			}
		}

		int predSum = TP + FP;
		int trueSum = TP + FN;

		PRECISION[c] = predSum ? static_cast<float>(TP) / predSum : 0.0f;
		RECALL[c] = trueSum ? static_cast<float>(TP) / trueSum : 0.0f;
		F1[c] = (PRECISION[c] + RECALL[c]) ?
			2.0f * PRECISION[c] * RECALL[c] / (PRECISION[c] + RECALL[c]) : 0.0f;

		correct += TP;
		total += trueSum;
	}

	ACCURACY = total ? static_cast<float>(correct) / total : 0.0f;
}


// -------------------------------------------------------------------------------------------------------------


void Network::showEvaluation() {
	Evaluate();

	int C = static_cast<int>(PRECISION.size());

	cout << "\n=========== EVALUATION METRICS ===========" << endl;
	cout << "Class\tPrecision\tRecall\t\tF1-Score" << endl;
	cout << "------------------------------------------" << endl;

	for (int i = 0; i < C; ++i) {
		printf("%5d\t%9.4f\t%9.4f\t%9.4f\n", i, PRECISION[i], RECALL[i], F1[i]);
	}

	cout << "------------------------------------------" << endl;
	printf("Overall Accuracy: %.4f\n", ACCURACY);
	cout << "==========================================\n" << endl;
}

void showProgressBar(int current, int total, int barWidth = 50) {
	float progress = static_cast<float>(current) / total;
	int pos = static_cast<int>(barWidth * progress);

	std::cout << "\r[";
	for (int i = 0; i < barWidth; ++i) {
		if (i < pos) std::cout << "=";
		else if (i == pos) std::cout << ">";
		else std::cout << " ";
	}
	std::cout << "] " << int(progress * 100.0) << "%";
	std::cout.flush();
}
