#include "KernelWrappers.cuh"

void Convo2DKernelForwardWrapper(BatchDevice* BDout, BatchDevice* BDi, BatchDevice* BDk,
	int S, int P, int blocks, int threads) {

	conv2dKernelF << <blocks, threads >> > (BDout, BDi, BDk, S, P);
	cudaCheck(cudaPeekAtLastError());
	cudaCheck(cudaDeviceSynchronize());
}


void Convo2DKernelBackwardWrapper(BatchDevice* BDi, BatchDevice* BDout, BatchDevice* BDk,
	int S, int P, float R2, float CLIP, int blocks, int threads) {

	conv2dKernelB<<<blocks, threads>>>(BDi, BDout, BDk, S, P, R2, CLIP);
	cudaCheck(cudaPeekAtLastError());
	cudaCheck(cudaDeviceSynchronize());
}

void ActivationKernelForwardWrapper(BatchDevice* BDout, BatchDevice* BDi, string type, 
	int blocks, int threads) {

	if (type == "relu") Relu<<<blocks, threads>>>(BDout, BDi);
	else if (type == "tanh") Tanh<<<blocks, threads>>>(BDout, BDi);
	else Sigmoid<<<blocks, threads>>>(BDout, BDi);
	cudaCheck(cudaPeekAtLastError());
	cudaCheck(cudaDeviceSynchronize());
}

void ActivationKernelBackwardWrapper(BatchDevice* BDi, BatchDevice* BDout, string type,
	int blocks, int threads) {

	if (type == "relu") dRelu<<<blocks, threads>>>(BDi, BDout);
	else if (type == "tanh") dTanh<<<blocks, threads >>>(BDi, BDout);
	else dSigmoid<<<blocks, threads>>>(BDi, BDout);
	cudaCheck(cudaPeekAtLastError());
	cudaCheck(cudaDeviceSynchronize());
}

void PoolingKernelForwardWrapper(BatchDevice* BDout, BatchDevice* BDi,
	string type, int kH, int kW, int S, int P, int blocks, int threads) {

	if (type == "max") MaxPoolKernelF<<<blocks, threads>>>(BDout, BDi, kH, kW, S, P);
	else if (type == "avg") AvgPoolKernelF<<<blocks, threads>>>(BDout, BDi, kH, kW, S, P);
	else MinPoolKernelF<<<blocks, threads>>>(BDout, BDi, kH, kW, S, P);
	cudaCheck(cudaPeekAtLastError());
	cudaCheck(cudaDeviceSynchronize());
}

void PoolingKernelBackwardWrapper(BatchDevice* BDi, BatchDevice* BDout, string type, 
	int kH, int kW, int S, int P, int blocks, int threads) {

	if (type == "max") MaxPoolKernelB<<<blocks, threads>>>(BDi, BDout, kH, kW, S, P);
	else if (type == "avg") AvgPoolKernelB<<<blocks, threads>>>(BDi, BDout, kH, kW, S, P);
	else MinPoolKernelB<<<blocks, threads>>>(BDi, BDout, kH, kW, S, P);
	cudaCheck(cudaPeekAtLastError());
	cudaCheck(cudaDeviceSynchronize());
}

void BatchNormKernelForwardWrapper(BatchDevice* BDout, BatchDevice* BDi, float* mean, 
	float* var, BatchDevice* gamma, BatchDevice* beta,
	int totalSize, int sizePerChannel, int C, int blocks, int threads, int blocksCH, int threadsCH) {

	MeanSumKernelF<<<blocks, threads>>>(BDi, mean);
	devideKernelF<<<blocksCH, threadsCH>>>(mean, sizePerChannel, C);
	VarSumKernelF<<<blocks, threads>>>(BDi, var, mean);
	devideKernelF<<<blocksCH, threadsCH>>>(var, sizePerChannel, C);

	BatchNormKernelF<<<(totalSize + threads - 1) / threads, threads>>>(
		BDout, BDi, var, mean, gamma, beta);
	cudaCheck(cudaPeekAtLastError());
	cudaCheck(cudaDeviceSynchronize());
}

void BatchNormKernelBackwardWrapper(BatchDevice* BDi, BatchDevice* BDout, float* mean,
	float* var, float* sumDy, float* sumDyXmu, BatchDevice* gamma, BatchDevice* beta,
	int totalSize, int sizePerChannel, int C, float R2, float CLIP, int blocks, int threads) {

	dGammaBetaKernelB<<<blocks, threads>>>(BDout, BDi, mean, var, gamma, beta, R2, CLIP);
	SumDyKernelB<<<blocks, threads>>>(sumDy, sumDyXmu, BDout, BDi, mean, var);
	BatchNormKernelB<<<(totalSize + threads - 1) / threads, threads>>>(
		BDi, BDout, mean, var, gamma, sumDy, sumDyXmu, C, sizePerChannel, R2, CLIP);
	cudaCheck(cudaPeekAtLastError());
	cudaCheck(cudaDeviceSynchronize());
}

void DropoutKernelForwardWrapper(BatchDevice* BDout, BatchDevice*BDi, BatchDevice* BDmask,  float P, int size,
	curandState* states, int blocks, int threads) {

	initCurand<<<blocks, threads>>>(time(NULL), states, size);
	cudaCheck(cudaPeekAtLastError());
	DropoutKernelF<<<blocks, threads>>>(BDout, BDi, BDmask, states, P);
	cudaCheck(cudaPeekAtLastError());
	cudaCheck(cudaDeviceSynchronize());
}

void DropoutKernelBackwardWrapper(BatchDevice* BDi, BatchDevice* BDout, BatchDevice* BDmask, 
	float P, int blocks, int threads) {

	DropoutKernelB<<<blocks, threads>>>(BDi, BDout, BDmask, P);
	cudaCheck(cudaPeekAtLastError());
	cudaCheck(cudaDeviceSynchronize());
}

void DenseKernelForwardWrapper(BatchDevice* BDout, BatchDevice* BDi, BatchDevice* weights, 
	BatchDevice* bias, int blocks, int threads) {

	DenseKernelF<<<blocks, threads>>>(BDout, BDi, weights, bias);
	cudaCheck(cudaPeekAtLastError());
	cudaCheck(cudaDeviceSynchronize());
}
void DenseKernelBackwardWrapper(BatchDevice* BDi, BatchDevice* BDout, BatchDevice* weights, 
	BatchDevice* bias, float R2, float CLIP, int blocks, int threads) {

	DenseKernelB<<<blocks, threads>>>(BDi, BDout, weights, bias, R2, CLIP);
	cudaCheck(cudaPeekAtLastError());
	cudaCheck(cudaDeviceSynchronize());
};

void OutputKernelWrapper(BatchDevice* BDout, BatchDevice* BDi, BatchDevice* labels, 
	int blocks, int threads) {

	OutputKernel<<<blocks, threads>>>(BDout, BDi, labels);
	cudaCheck(cudaPeekAtLastError());
	cudaCheck(cudaDeviceSynchronize());
}

void CopyGradWrapper(BatchDevice* BDi, BatchDevice* BDout, int blocks, int threads) {
	CopyGrad<<<blocks, threads>>>(BDi, BDout);
	cudaCheck(cudaPeekAtLastError());
	cudaCheck(cudaDeviceSynchronize());
}

// -----------------------------------------------------------------------------------------------------

void kernelsUpdateWrapper(BatchDevice* kernels, float lr, int blocks, int threads) {
	kernelsUpdate<<<blocks, threads>>>(kernels, lr);
	cudaCheck(cudaPeekAtLastError());
	cudaCheck(cudaDeviceSynchronize());
}

void gammabetaUpdateWrapper(BatchDevice* gamma, BatchDevice* beta, float lr, int blocks, int threads) {
	gammabetaUpdate<<<blocks, threads>>>(gamma, beta, lr);
	cudaCheck(cudaPeekAtLastError());
	cudaCheck(cudaDeviceSynchronize());
}

void weightbiasUpdateWrapper(BatchDevice* weights, BatchDevice* bias, float lr, 
	int blocksW, int threadsW, int blocksB, int threadsB) {
	weightUpdate<<<blocksW, threadsW>>>(weights, lr);
	biasUpdate<<<blocksB, threadsB>>>(bias, lr);
	cudaCheck(cudaPeekAtLastError());
	cudaCheck(cudaDeviceSynchronize());
}

// ----------------------------------------------------------------------------------------------------------

void EvaluateKernelWrapper(BatchDevice* probabilities, BatchDevice* actualLables, BatchDevice* Matrix, 
	int blocks, int threads) {
	EvaluateKernel<<<blocks, threads>>>(probabilities, actualLables, Matrix);
	cudaCheck(cudaPeekAtLastError());
	cudaCheck(cudaDeviceSynchronize());
}