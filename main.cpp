#include "module.hpp"
#include "optimizer.hpp"

#include "DataLoader.hpp"
#include "MNISTLoader.hpp"

#include <iomanip>




static Sequential buildModel() {
    Sequential model;

    model.Input({ 60000,1,28,28 }, 32);


    model.Conv2D({ 32,1,3,3 }, 1, 1);
    model.ReLU();

    model.Conv2D({ 64,32,3,3 }, 1, 1);
    model.ReLU();

    model.MaxPooling(2, 2);

    model.Flatten();

    model.Dense(128);
    model.ReLU();

    model.Dense(10);

    model.Output("Softmax");


    return model;
}



constexpr int BatchSize = 32;
constexpr int Epochs = 2;




static void TrainingLoop(Sequential& model, ExecutionContext& ctx, Optimizer& opt, Loss& loss, Tensor& images, Tensor& labels) {
    const size_t TotalSamples = images.shape()[0];
    const size_t TotalBatches = (TotalSamples + BatchSize - 1) / BatchSize;

    for (int epoch = 1; epoch <= Epochs; ++epoch) {
        LoadImages imageLoader(images, BatchSize);
        LoadLabels labelLoader(labels, BatchSize);

        float EpochLoss = 0.0f;
        size_t Batch = 0;

        while (imageLoader.hasNext() && labelLoader.hasNext()) {
            Tensor ImagesBatch = imageLoader.next();
            Tensor LabelsBatch = labelLoader.next();

            float LossValue = model.Train(ctx, loss, opt, ImagesBatch, LabelsBatch);

            EpochLoss += LossValue;
            ++Batch;

            const int Progress = static_cast<int>(50.0 * Batch / TotalBatches);

            std::cout << "\rEpoch " << epoch << "/" << Epochs << " [";

            for (int i = 0; i < 50; ++i)
                std::cout << (i < Progress ? '=' : ' ');

            std::cout << "] "
                << std::setw(3)
                << static_cast<int>(100.0 * Batch / TotalBatches)
                << "%";

            std::cout.flush();
        }

        std::cout << "\nEpoch "
            << epoch
            << " Loss: "
            << EpochLoss / Batch
            << "\n\n";
    }
}


int main() {
    auto model = buildModel();

    Tensor Images =
        load_mnist_images(
            "D:/PROJECTs/TorchLess/Dataset/train-images.idx3-ubyte"
        );

    Tensor Labels =
        load_mnist_labels(
            "D:/PROJECTs/TorchLess/Dataset/train-labels.idx1-ubyte"
        );


    ExecutionContext ctx;
    Adam adam(1e-3f);
    Loss loss("CCE");

    TrainingLoop(model, ctx, adam, loss, Images, Labels);


    LoadImages predictionLoader(Images, 5);
    Tensor TestBatch = predictionLoader.next();
    Tensor& Prediction = model.Predict(ctx, TestBatch);

    Prediction.debug_statistics_print("Prediction");

    return 0;
}