#pragma once

#include "tensor.hpp"

#include <stdexcept>
#include <algorithm>

class LabelLoader {
private:

    const Tensor& Dataset;

    size_t BatchSize;

    size_t NumSamples;

    size_t CurrentIndex = 0;


public:

    LabelLoader(const Tensor& dataset, size_t batchSize)
        :
        Dataset(dataset),
        BatchSize(batchSize)
    {

        if (dataset.shape().size() != 1)
            throw std::runtime_error("Labels must be 1D");


        NumSamples = dataset.shape()[0];
    }


    bool hasNext() const {
        return CurrentIndex < NumSamples;
    }


    Tensor next() {

        size_t Remaining = NumSamples - CurrentIndex;

        size_t CurrentBatch = std::min(BatchSize, Remaining);


        Tensor BatchTensor(
            { CurrentBatch },
            DataType::Int32
        );


        CUDA_CHECK(
            cudaMemcpy(
                BatchTensor.int_data(),
                Dataset.int_data() + CurrentIndex,
                CurrentBatch * sizeof(int),
                cudaMemcpyDeviceToDevice
            )
        );


        CurrentIndex += CurrentBatch;


        return BatchTensor;
    }


    void reset() {
        CurrentIndex = 0;
    }
};