#include "Batch.h"

void BatchWrapper::resetData() const {
    BatchDevice ptr;
    cudaMemcpy(&ptr, deviceRaw_Ptr, sizeof(BatchDevice), cudaMemcpyDeviceToHost);
    cudaMemset(ptr.DATA, 0, SIZE * sizeof(float));
}

void BatchWrapper::resetGrad() const {
    BatchDevice ptr;
    cudaMemcpy(&ptr, deviceRaw_Ptr, sizeof(BatchDevice), cudaMemcpyDeviceToHost);
    cudaMemset(ptr.GRAD, 0, SIZE * sizeof(float));
}

BatchDevice* CreateBatchDevice(int B, int C, int H, int W, const vector<float>& data) {
    int size = B * C * H * W;

    float* d_data;
    float* d_grad;
    int* d_strides;
    cudaMalloc(&d_data, sizeof(float) * size);
    cudaMalloc(&d_grad, sizeof(float) * size);
    cudaMalloc(&d_strides, sizeof(int) * 4);

    cudaMemcpy(d_data, data.data(), sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemset(d_grad, 0, sizeof(float) * size);

    int h_strides[4] = { C * H * W, H * W, W, 1};
    cudaMemcpy(d_strides, h_strides, sizeof(int) * 4, cudaMemcpyHostToDevice);

    BatchDevice hostDev;
    hostDev.DATA = d_data;
    hostDev.GRAD = d_grad;
    hostDev.STRIDES = d_strides;
    hostDev.BATCHs = B;
    hostDev.CHANNELs = C;
    hostDev.HEIGHTs = H;
    hostDev.WIDTHs = W;
    hostDev.SIZE = size;

    BatchDevice* devPtr;
    cudaMalloc(&devPtr, sizeof(BatchDevice));
    cudaMemcpy(devPtr, &hostDev, sizeof(BatchDevice), cudaMemcpyHostToDevice);

    return devPtr;
}

shared_ptr<BatchWrapper> batchWrapper(const vector<vector<vector<vector<float>>>>& data) {
    int B = data.size();
    int C = data[0].size();
    int H = data[0][0].size();
    int W = data[0][0][0].size();
    int size = B * C * H * W;

    vector<float> flat(size);
    int idx = 0;
    for (int b = 0; b < B; ++b)
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                    flat[idx++] = data[b][c][h][w];

    BatchDevice* dev = CreateBatchDevice(B, C, H, W, flat);

    auto bw = make_shared<BatchWrapper>();
    bw->deviceRaw_Ptr = dev;
    bw->BATCHs = B;
    bw->CHANNELs = C;
    bw->HEIGHTs = H;
    bw->WIDTHs = W;
    bw->SIZE = size;

    return bw;
}

shared_ptr<BatchWrapper> batchWrapperConst(int B, int C, int H, int W, float val) {
    int size = B * C * H * W;
    vector<float> flat(size, val);

    BatchDevice* dev = CreateBatchDevice(B, C, H, W, flat);

    auto bw = make_shared<BatchWrapper>();
    bw->deviceRaw_Ptr = dev;
    bw->BATCHs = B;
    bw->CHANNELs = C;
    bw->HEIGHTs = H;
    bw->WIDTHs = W;
    bw->SIZE = size;

    return bw;
}

shared_ptr<BatchWrapper> batchWrapperRandom(int B, int C, int H, int W) {
    int size = B * C * H * W;
    vector<float> randomData(size);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < size; ++i)
        randomData[i] = dist(gen);

    BatchDevice* dev = CreateBatchDevice(B, C, H, W, randomData);

    auto bw = make_shared<BatchWrapper>();
    bw->deviceRaw_Ptr = dev;
    bw->BATCHs = B; bw->CHANNELs = C;
    bw->HEIGHTs = H; bw->WIDTHs = W; bw->SIZE = size;

    return bw;
}
