#ifndef BATCH_CUH
#define BATCH_CUH

#include <cuda_runtime.h>
using namespace std;

struct BatchDevice {
	float* DATA;
	float* GRAD;
	int* STRIDES;
	
	int BATCHs; int CHANNELs; int HEIGHTs; int WIDTHs;
	int SIZE;

	__device__ int FlatIdx(int b, int ch, int h, int w) const {
		return STRIDES[0] * b + STRIDES[1] * ch + STRIDES[2] * h + STRIDES[3] * w;
	}

	__device__ void flatToNd(int flat, int& b, int& c, int& h, int& w) const {
		b = flat / STRIDES[0];
		flat = flat % STRIDES[0];

		c = flat / STRIDES[1];
		flat = flat % STRIDES[1];

		h = flat / STRIDES[2];
		flat = flat % STRIDES[2];

		w = flat / STRIDES[3];
	}
};

#endif