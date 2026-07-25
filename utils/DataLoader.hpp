#pragma once

#include "tensor.hpp"

#include <stdexcept>
#include <algorithm>




class DataLoader {

protected:

    const Tensor& Dataset;

    size_t BatchSize;
    size_t NumSamples;

    size_t CurrentIndex = 0;

    DataLoader(const Tensor& dataset, size_t batchSize) 
        : Dataset(dataset), BatchSize(batchSize), NumSamples(dataset.shape()[0]), CurrentIndex(0) {

        if (BatchSize == 0)
            throw std::runtime_error("Batch size cannot be zero");
    }

public:

    bool hasNext() const {
        return CurrentIndex < NumSamples;
    }

    size_t batches() const {
        return (NumSamples + BatchSize - 1) / BatchSize;
    }

    virtual Tensor next() {
        return Tensor();
    };

    void reset() {
        CurrentIndex = 0;
    }
};




class LoadImages : public DataLoader {

public:

    LoadImages(const Tensor& dataset, size_t batchSize) : DataLoader(dataset, batchSize) {
        if (dataset.shape().size() != 4)
            throw std::runtime_error("Dataset must be NCHW");

        NumSamples = dataset.shape()[0];
    }

    Tensor next() override {
        if (!hasNext())
            throw std::runtime_error("No more batches");


        size_t Remaining = NumSamples - CurrentIndex;

        size_t CurrentBatch = std::min(BatchSize, Remaining);

        std::vector<size_t> Shape = { CurrentBatch, Dataset.shape()[1], Dataset.shape()[2], Dataset.shape()[3] };


        Tensor BatchTensor(Shape);

        size_t SampleSize = Dataset.shape()[1] * Dataset.shape()[2] * Dataset.shape()[3];

        size_t Offset = CurrentIndex * SampleSize;


        CUDA_CHECK(
            cudaMemcpy(BatchTensor.data(), Dataset.data() + Offset, CurrentBatch * SampleSize * sizeof(float), cudaMemcpyDeviceToDevice)
        );


        CurrentIndex += CurrentBatch;

        return BatchTensor;
    }
};




class LoadLabels : public DataLoader {

public:

    LoadLabels(const Tensor& dataset, size_t batchSize) : DataLoader(dataset, batchSize) {
        if (dataset.shape().size() != 1)
            throw std::runtime_error("Labels must be 1D");

        NumSamples = dataset.shape()[0];
    }

    Tensor next() override {
        size_t Remaining = NumSamples - CurrentIndex;

        size_t CurrentBatch = std::min(BatchSize, Remaining);


        Tensor BatchTensor({ CurrentBatch }, DataType::Int32);


        CUDA_CHECK(
            cudaMemcpy(BatchTensor.int_data(), Dataset.int_data() + CurrentIndex, CurrentBatch * sizeof(int), cudaMemcpyDeviceToDevice)
        );


        CurrentIndex += CurrentBatch;

        return BatchTensor;
    }
};