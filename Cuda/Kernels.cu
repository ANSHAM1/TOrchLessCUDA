#include "kernels.cuh"

__global__ void initCurand(unsigned int seed, curandState* states, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) curand_init(seed, idx, 0, &states[idx]);
}

__global__ void conv2dKernelF(BatchDevice* output, BatchDevice* inputs, BatchDevice* kernels, int S, int P) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= output->SIZE) return;
	output->DATA[idx] = 0.0f;
	for (int i = 0; i < inputs->CHANNELs; i++)
		for (int j = 0; j < kernels->HEIGHTs; j++)
			for (int k = 0; k < kernels->WIDTHs; k++) {
				int bout, cout, hout, wout;
				output->flatToNd(idx, bout, cout, hout, wout);
				int hi = hout * S - P + j;
				int wi = wout * S - P + k;
				if (hi < 0 || hi >= inputs->HEIGHTs || wi < 0 || wi >= inputs->WIDTHs)
					continue;

				int idx_i = inputs->FlatIdx(bout, i, hi, wi);
				int idx_k = kernels->FlatIdx(cout, i, j, k);
				if (idx_i < 0 || idx_i >= inputs->SIZE || idx_k < 0 || idx_k >= kernels->SIZE)
					continue;

				output->DATA[idx] += inputs->DATA[idx_i] * kernels->DATA[idx_k];
			}
}

__global__ void conv2dKernelB(BatchDevice* input, BatchDevice* output, BatchDevice* kernels, int S, int P, float lambda, float clip_val) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < input->SIZE) {
		input->GRAD[idx] = 0.0f;

		for (int i = 0; i < output->CHANNELs; i++) {
			for (int j = 0; j < kernels->HEIGHTs; j++) {
				for (int k = 0; k < kernels->WIDTHs; k++) {
					int bout, c_in, h_in, w_in;
					input->flatToNd(idx, bout, c_in, h_in, w_in);

					int pad_h = kernels->HEIGHTs - 1 - P;
					int pad_w = kernels->WIDTHs - 1 - P;

					int h_out = h_in + pad_h - j;
					int w_out = w_in + pad_w - k;

					if (h_out % S != 0 || w_out % S != 0) continue;
					h_out /= S;
					w_out /= S;

					if (h_out < 0 || h_out >= output->HEIGHTs || w_out < 0 || w_out >= output->WIDTHs)
						continue;

					int dout_idx = output->FlatIdx(bout, i, h_out, w_out);
					int k_idx = kernels->FlatIdx(i, c_in, kernels->HEIGHTs - 1 - j, kernels->WIDTHs - 1 - k);
					int in_idx = input->FlatIdx(bout, c_in, h_in, w_in);

					float grad_val = output->GRAD[dout_idx] * kernels->DATA[k_idx];
					grad_val += lambda * kernels->DATA[k_idx];
					if (grad_val > clip_val) grad_val = clip_val;
					else if (grad_val < -clip_val) grad_val = -clip_val;

					input->GRAD[in_idx] += grad_val;

					atomicAdd(&(kernels->GRAD[k_idx]), output->GRAD[dout_idx] * input->DATA[in_idx]);
				}
			}
		}
	}
}


__global__ void Relu(BatchDevice* BWout, BatchDevice* BWi) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < BWout->SIZE) {
		float val = BWi->DATA[idx];
		BWout->DATA[idx] = val > 0 ? val : 0;
	}
}

__global__ void Sigmoid(BatchDevice* BWout, BatchDevice* BWi) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < BWout->SIZE) {
		float val = BWi->DATA[idx];
		BWout->DATA[idx] = 1.0f / (1.0f + expf(-val));
	}
}

__global__ void Tanh(BatchDevice* BWout, BatchDevice* BWi) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < BWout->SIZE) {
		float val = BWi->DATA[idx];
		BWout->DATA[idx] = tanhf(val);
	}
}

__global__ void dRelu(BatchDevice* BWi, BatchDevice* BWout) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < BWi->SIZE) {
		float y = BWout->DATA[idx];
		float grad_out = BWout->GRAD[idx];
		BWi->GRAD[idx] += (y > 0 ? 1.0f : 0.0f) * grad_out;
	}
}

__global__ void dSigmoid(BatchDevice* BWi, BatchDevice* BWout) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < BWi->SIZE) {
		float y = BWout->DATA[idx];
		float grad_out = BWout->GRAD[idx];
		BWi->GRAD[idx] += y * (1.0f - y) * grad_out;
	}
}

__global__ void dTanh(BatchDevice* BWi, BatchDevice* BWout) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < BWi->SIZE) {
		float y = BWout->DATA[idx];
		float grad_out = BWout->GRAD[idx];
		BWi->GRAD[idx] += (1.0f - y * y) * grad_out;
	}
}

__global__ void MaxPoolKernelF(BatchDevice* output, BatchDevice* inputs, int kH, int kW, int S, int P) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= output->SIZE) return;

	int b, c, h_out, w_out;
	output->flatToNd(idx, b, c, h_out, w_out);

	float max_val = -FLT_MAX;

	for (int i = 0; i < kH; i++) {
		for (int j = 0; j < kW; j++) {
			int h_in = h_out * S - P + i;
			int w_in = w_out * S - P + j;

			if (h_in >= 0 && h_in < inputs->HEIGHTs &&
				w_in >= 0 && w_in < inputs->WIDTHs) {

				int idx_i = inputs->FlatIdx(b, c, h_in, w_in);
				float val = inputs->DATA[idx_i];
				if (val > max_val) max_val = val;
			}
		}
	}

	output->DATA[idx] = max_val;
}

__global__ void AvgPoolKernelF(BatchDevice* output, BatchDevice* inputs, int kH, int kW, int S, int P) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= output->SIZE) return;

	int b, c, h_out, w_out;
	output->flatToNd(idx, b, c, h_out, w_out);

	float sum = 0.0f;
	int num = 0;

	for (int i = 0; i < kH; i++) {
		for (int j = 0; j < kW; j++) {
			int h_in = h_out * S - P + i;
			int w_in = w_out * S - P + j;

			if (h_in >= 0 && h_in < inputs->HEIGHTs &&
				w_in >= 0 && w_in < inputs->WIDTHs) {

				int idx_i = inputs->FlatIdx(b, c, h_in, w_in);
				sum += inputs->DATA[idx_i];
				num++;
			}
		}
	}

	output->DATA[idx] = sum / num;
}

__global__ void MinPoolKernelF(BatchDevice* output, BatchDevice* inputs, int kH, int kW, int S, int P) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= output->SIZE) return;

	int b, c, h_out, w_out;
	output->flatToNd(idx, b, c, h_out, w_out);

	float min_val = FLT_MAX;

	for (int i = 0; i < kH; i++) {
		for (int j = 0; j < kW; j++) {
			int h_in = h_out * S - P + i;
			int w_in = w_out * S - P + j;

			if (h_in >= 0 && h_in < inputs->HEIGHTs &&
				w_in >= 0 && w_in < inputs->WIDTHs) {

				int idx_i = inputs->FlatIdx(b, c, h_in, w_in);
				float val = inputs->DATA[idx_i];
				if (val < min_val) min_val = val;
			}
		}
	}

	output->DATA[idx] = min_val;
}


__global__ void MaxPoolKernelB(BatchDevice* inputs, BatchDevice* outputs, int kH, int kW, int S, int P) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= outputs->SIZE) return;

	int b, c, h_out, w_out;
	outputs->flatToNd(idx, b, c, h_out, w_out);

	float max_val = -FLT_MAX;
	int max_h = -1, max_w = -1;

	// Find max position from input that produced this output
	for (int i = 0; i < kH; i++) {
		for (int j = 0; j < kW; j++) {
			int h_in = h_out * S - P + i;
			int w_in = w_out * S - P + j;

			if (h_in >= 0 && h_in < inputs->HEIGHTs &&
				w_in >= 0 && w_in < inputs->WIDTHs) {

				int idx_i = inputs->FlatIdx(b, c, h_in, w_in);
				float val = inputs->DATA[idx_i];
				if (val > max_val) {
					max_val = val;
					max_h = h_in;
					max_w = w_in;
				}
			}
		}
	}

	if (max_h != -1 && max_w != -1) {
		int grad_idx = inputs->FlatIdx(b, c, max_h, max_w);
		int dout_idx = outputs->FlatIdx(b, c, h_out, w_out);
		atomicAdd(&inputs->GRAD[grad_idx], outputs->GRAD[dout_idx]);
	}
}

__global__ void AvgPoolKernelB(BatchDevice* input, BatchDevice* output, int kH, int kW, int S, int P) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= output->SIZE) return;

	int b, c, h_out, w_out;
	output->flatToNd(idx, b, c, h_out, w_out);

	float grad = output->GRAD[idx];
	int count = 0;

	for (int i = 0; i < kH; i++) {
		for (int j = 0; j < kW; j++) {
			int h_in = h_out * S - P + i;
			int w_in = w_out * S - P + j;

			if (h_in >= 0 && h_in < input->HEIGHTs &&
				w_in >= 0 && w_in < input->WIDTHs)
				count++;
		}
	}

	if (count == 0) return;
	float grad_div = grad / count;

	for (int i = 0; i < kH; i++) {
		for (int j = 0; j < kW; j++) {
			int h_in = h_out * S - P + i;
			int w_in = w_out * S - P + j;

			if (h_in >= 0 && h_in < input->HEIGHTs &&
				w_in >= 0 && w_in < input->WIDTHs) {
				int idx_in = input->FlatIdx(b, c, h_in, w_in);
				atomicAdd(&input->GRAD[idx_in], grad_div);
			}
		}
	}
}

__global__ void MinPoolKernelB(BatchDevice* input, BatchDevice* output, int kH, int kW, int S, int P) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= output->SIZE) return;

	int b, c, h_out, w_out;
	output->flatToNd(idx, b, c, h_out, w_out);

	float min_val = FLT_MAX;
	int h_min = -1, w_min = -1;

	// Find location of min value in input region
	for (int i = 0; i < kH; i++) {
		for (int j = 0; j < kW; j++) {
			int h_in = h_out * S - P + i;
			int w_in = w_out * S - P + j;

			if (h_in >= 0 && h_in < input->HEIGHTs &&
				w_in >= 0 && w_in < input->WIDTHs) {
				int idx_in = input->FlatIdx(b, c, h_in, w_in);
				float val = input->DATA[idx_in];

				if (val < min_val) {
					min_val = val;
					h_min = h_in;
					w_min = w_in;
				}
			}
		}
	}

	if (h_min >= 0 && w_min >= 0) {
		int idx_min = input->FlatIdx(b, c, h_min, w_min);
		atomicAdd(&input->GRAD[idx_min], output->GRAD[idx]);
	}
}


__global__ void devideKernelF(float* sum, int num_per_channel, int C) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= C) return;

	sum[idx] /= num_per_channel;
}

__global__ void MeanSumKernelF(BatchDevice* inputs, float* x_sum) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= inputs->SIZE) return;

	int ch = (idx % inputs->STRIDES[0]) / inputs->STRIDES[1];
	atomicAdd(&x_sum[ch], inputs->DATA[idx]);
}

__global__ void VarSumKernelF(BatchDevice* inputs, float* var_, float* x_) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= inputs->SIZE) return;

	int ch = (idx % inputs->STRIDES[0]) / inputs->STRIDES[1];
	float diff = inputs->DATA[idx] - x_[ch];
	atomicAdd(&var_[ch], diff * diff);
}

__global__ void BatchNormKernelF(BatchDevice* output, BatchDevice* inputs, float* mean, float* var, BatchDevice* gamma, BatchDevice* beta) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= output->SIZE) return;

	int b, c, h, w;
	inputs->flatToNd(idx, b, c, h, w);

	float x = inputs->DATA[idx];
	float mu = mean[c];
	float var_eps = var[c] + 1e-5f;
	float x_hat = (x - mu) / sqrtf(var_eps);

	float g = gamma->DATA[c];
	float b_val = beta->DATA[c];

	output->DATA[idx] = g * x_hat + b_val;
}

__global__ void dGammaBetaKernelB(BatchDevice* Y, BatchDevice* X, float* mean, float* var, BatchDevice* gamma, BatchDevice* beta, float lambda, float clip_val) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= Y->SIZE) return;

	int b, c, h, w;
	Y->flatToNd(idx, b, c, h, w);
	int i = Y->FlatIdx(b, c, h, w);

	float x_hat = (X->DATA[i] - mean[c]) / sqrtf(var[c] + 1e-5f);
	float dy = Y->GRAD[i];

	float dgamma = dy * x_hat + lambda * gamma->DATA[c];
	float dbeta = dy + lambda * beta->DATA[c];

	dgamma = fminf(fmaxf(dgamma, -clip_val), clip_val);
	dbeta = fminf(fmaxf(dbeta, -clip_val), clip_val);

	atomicAdd(&gamma->GRAD[c], dgamma);
	atomicAdd(&beta->GRAD[c], dbeta);
}

__global__ void SumDyKernelB(float* sum_dy, float* sum_dy_xhat, BatchDevice* Y, BatchDevice* X, float* mean, float* var) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= Y->SIZE) return;

	int b, c, h, w;
	Y->flatToNd(idx, b, c, h, w);
	int i = Y->FlatIdx(b, c, h, w);

	float x_hat = (X->DATA[i] - mean[c]) / sqrtf(var[c] + 1e-5f);
	float dy = Y->GRAD[i];

	atomicAdd(&sum_dy[c], dy);
	atomicAdd(&sum_dy_xhat[c], dy * x_hat);
}

__global__ void BatchNormKernelB(BatchDevice* X, BatchDevice* Y, float* mean, float* var, BatchDevice* gamma, float* sum_dy, float* sum_dy_xhat, int C, int N, float lambda, float clip_val) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= Y->SIZE) return;

	int b, c, h, w;
	Y->flatToNd(idx, b, c, h, w);
	int i = Y->FlatIdx(b, c, h, w);

	float x_hat = (X->DATA[i] - mean[c]) / sqrtf(var[c] + 1e-5f);
	float dy = Y->GRAD[i];
	float g = gamma->DATA[c];

	float mean_dy = sum_dy[c] / N;
	float mean_dy_xhat = sum_dy_xhat[c] / N;

	float grad = g * (dy - mean_dy - x_hat * mean_dy_xhat) / sqrtf(var[c] + 1e-5f);

	grad += lambda * X->DATA[i];
	grad = fminf(fmaxf(grad, -clip_val), clip_val);
	X->GRAD[i] = grad;
}


__global__ void DropoutKernelF(BatchDevice* outputs, BatchDevice* inputs, BatchDevice* mask, curandState* states, float p) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= outputs->SIZE) return;

	curandState localState = states[idx];
	float rand_val = curand_uniform(&localState);

	float keep = rand_val >= p ? 1.0f : 0.0f;

	mask->DATA[idx] = keep;
	outputs->DATA[idx] = inputs->DATA[idx] * keep / (1.0f - p);

	states[idx] = localState;
}

__global__ void DropoutKernelB(BatchDevice* inputs, BatchDevice* outputs, BatchDevice* mask, float p) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= inputs->SIZE) return;

	float keep = mask->DATA[idx];
	inputs->GRAD[idx] = mask->DATA[idx] * outputs->GRAD[idx] / (1.0f - p);
}

__global__ void DenseKernelF(BatchDevice* outputs, BatchDevice* inputs, BatchDevice* weights, BatchDevice* bias) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int B = inputs->BATCHs;
	int C_in = inputs->CHANNELs * inputs->HEIGHTs * inputs->WIDTHs;
	int C_out = outputs->WIDTHs;

	if (idx >= B * C_out) return;

	int b = idx / C_out;
	int j = idx % C_out;

	float sum = bias->DATA[j];

	for (int i = 0; i < C_in; i++) {
		int in_idx = b * C_in + i;
		int w_idx = j * C_in + i;
		sum += inputs->DATA[in_idx] * weights->DATA[w_idx];
	}

	outputs->DATA[idx] = sum;
}

__global__ void DenseKernelB(BatchDevice* dInputs, BatchDevice* dOutputs, BatchDevice* weights, BatchDevice* bias, float lambda, float clip_val) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int B = dInputs->BATCHs;
	int C_in = dInputs->CHANNELs * dInputs->HEIGHTs * dInputs->WIDTHs;
	int C_out = dOutputs->WIDTHs;

	if (idx >= B * C_out) return;

	int b = idx / C_out;
	int j = idx % C_out;

	float grad_out = dOutputs->GRAD[idx];

	for (int i = 0; i < C_in; i++) {
		int in_idx = b * C_in + i;
		int w_idx = j * C_in + i;

		atomicAdd(&dInputs->GRAD[in_idx], grad_out * weights->DATA[w_idx]);

		float grad_w = grad_out * dInputs->DATA[in_idx] + lambda * weights->DATA[w_idx];
		grad_w = fminf(fmaxf(grad_w, -clip_val), clip_val);
		atomicAdd(&weights->GRAD[w_idx], grad_w);
	}

	float grad_b = grad_out + lambda * bias->DATA[j];
	grad_b = fminf(fmaxf(grad_b, -clip_val), clip_val);
	atomicAdd(&bias->GRAD[j], grad_b);
}


__global__ void OutputKernel(BatchDevice* outputs, BatchDevice* inputs, BatchDevice* oneHot) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = inputs->BATCHs;
	int classes = inputs->WIDTHs;

	if (idx >= batch * classes) return;

	int b = idx / classes;
	int c = idx % classes;

	float maxLogit = -FLT_MAX;
	for (int i = 0; i < classes; ++i) {
		float val = inputs->DATA[b * classes + i];
		if (val > maxLogit) maxLogit = val;
	}

	float sumExp = 0.0f;
	for (int i = 0; i < classes; ++i) {
		sumExp += expf(inputs->DATA[b * classes + i] - maxLogit);
	}

	float expVal = expf(inputs->DATA[idx] - maxLogit);
	outputs->DATA[idx] = expVal / sumExp;

	outputs->GRAD[idx] = outputs->DATA[idx] - oneHot->DATA[idx];
}

__global__ void CopyGrad(BatchDevice* input, BatchDevice* output) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < input->SIZE) {
		input->GRAD[idx] = output->GRAD[idx];
	}
}


//__________________________________________________________________________________________________
//learnable parameters update kernels

__global__ void kernelsUpdate(BatchDevice* kernels, float lr) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= kernels->SIZE) return;

	kernels->DATA[idx] -= lr * kernels->GRAD[idx];
}

__global__ void gammabetaUpdate(BatchDevice* gamma, BatchDevice* beta, float lr) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= gamma->SIZE) return;

	gamma->DATA[idx] -= lr * gamma->GRAD[idx];
	beta->DATA[idx] -= lr * beta->GRAD[idx];
}

__global__ void weightUpdate(BatchDevice* weights, float lr) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= weights->SIZE) return;

	weights->DATA[idx] -= lr * weights->GRAD[idx];
}

__global__ void biasUpdate(BatchDevice* bias, float lr) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= bias->SIZE) return;

	bias->DATA[idx] -= lr * bias->GRAD[idx];
}

// ------------------------------------------------------------------------------------------------------

__global__ void EvaluateKernel(BatchDevice* predicted, BatchDevice* actual, BatchDevice* matrix) {
	int batchIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (batchIdx >= predicted->BATCHs) return;

	int C = predicted->WIDTHs;

	float maxVal = -FLT_MAX;
	int predClass = 0;
	for (int c = 0; c < C; ++c) {
		float val = predicted->DATA[batchIdx * C + c];
		if (val > maxVal) {
			maxVal = val;
			predClass = c;
		}
	}

	int actualClass = 0;
	for (int c = 0; c < C; ++c) {
		if (actual->DATA[batchIdx * C + c] == 1.0f) {
			actualClass = c;
			break;
		}
	}

	atomicAdd(&matrix->DATA[actualClass * C + predClass], 1.0f);
}
