#ifndef KERNEL_WRAPPERS_CUH
#define KERNEL_WRAPPERS_CUH

#include "Kernels.cuh"

extern "C" {
    void Convo2DKernelForwardWrapper(BatchDevice* BDout, BatchDevice* BDi, BatchDevice* BDk,
        int S, int P, int blocks, int threads);
    void Convo2DKernelBackwardWrapper(BatchDevice* BDi, BatchDevice* BDout, BatchDevice* BDk,
        int S, int P, float R2, float CLIP, int blocks, int threads);


    void ActivationKernelForwardWrapper(BatchDevice* BDout, BatchDevice* BDi, string type, 
        int blocks, int threads);
    void ActivationKernelBackwardWrapper(BatchDevice* BDi, BatchDevice* BDout, string type,
        int blocks, int threads);


    void PoolingKernelForwardWrapper(BatchDevice* BDout, BatchDevice* BDi,
        string type, int kH, int kW, int S, int P, int blocks, int threads);
    void PoolingKernelBackwardWrapper(BatchDevice* BDout, BatchDevice* BDi, string type, 
        int kH, int kW, int S, int P, int blocks, int threads);


    void BatchNormKernelForwardWrapper(BatchDevice* BDout, BatchDevice* BDi, float* mean,
        float* var, BatchDevice* gamma, BatchDevice* beta, int totalSize, int sizePerChannel, 
        int C, int blocks, int threads, int blocksCH, int threadsCH);
    void BatchNormKernelBackwardWrapper(BatchDevice* BDout, BatchDevice* BDi, float* mean,
        float* var, float* sumDy, float* sumDyXmu, BatchDevice* gamma, BatchDevice* beta,
        int totalSize, int sizePerChannel, int C, float R2, float CLIP, int blocks, int threads);


    void DropoutKernelForwardWrapper(BatchDevice* BDout, BatchDevice* BDi, BatchDevice* BDmask, float P, 
        int size, curandState* states, int blocks, int threads);
    void DropoutKernelBackwardWrapper(BatchDevice* BDi, BatchDevice* BDout, BatchDevice* BDmask,
        float P, int blocks, int threads);


    void DenseKernelForwardWrapper(BatchDevice* BDout, BatchDevice* BDi, BatchDevice* weights,
        BatchDevice* bias, int blocks, int threads);
    void DenseKernelBackwardWrapper(BatchDevice* BDi, BatchDevice* BDout, BatchDevice* weights,
        BatchDevice* bias, float R2, float CLIP, int blocks, int threads);


    void OutputKernelWrapper(BatchDevice* BDout, BatchDevice* BDi, BatchDevice* labels,
        int blocks, int threads);
    void CopyGradWrapper(BatchDevice* BDi, BatchDevice* BDout, int blocks, int threads);

    // learnable parameters update kernel wrapper
    void kernelsUpdateWrapper(BatchDevice* kernels, float lr, int blocks, int threads);
    void gammabetaUpdateWrapper(BatchDevice* gamma, BatchDevice* beta, float lr, int blocks, int threads);
    void weightbiasUpdateWrapper(BatchDevice* weights, BatchDevice* bias, float lr,
        int blocksW, int threadsW, int blocksB, int threadsB);

    // Evaluation Matrix
    void EvaluateKernelWrapper(BatchDevice* probabilities, BatchDevice* actualLables, BatchDevice* Matrix,
        int blocks, int threads);
}

#endif
