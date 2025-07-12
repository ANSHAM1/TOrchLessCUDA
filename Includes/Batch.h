#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <random>
#include <functional>

#include "BatchDevice.cuh"
using namespace std;

struct BatchWrapper {
	BatchDevice* deviceRaw_Ptr = nullptr;
	int BATCHs, CHANNELs, HEIGHTs, WIDTHs;
	int SIZE;

	function<void(float lr, float r2, float gradclip)> backward;

    void resetData() const;
    void resetGrad() const;

    ~BatchWrapper() {
        if (deviceRaw_Ptr) {
            BatchDevice temp;
            cudaMemcpy(&temp, deviceRaw_Ptr, sizeof(BatchDevice), cudaMemcpyDeviceToHost);

            cudaFree(temp.DATA);
            cudaFree(temp.GRAD);
            cudaFree(temp.STRIDES);

            cudaFree(deviceRaw_Ptr);
            deviceRaw_Ptr = nullptr;
        }
    }

};

shared_ptr<BatchWrapper> batchWrapper(const vector<vector<vector<vector<float>>>>& data);
shared_ptr<BatchWrapper> batchWrapperConst(int B, int C, int H, int W, float val);
shared_ptr<BatchWrapper> batchWrapperRandom(int B, int C, int H, int W);