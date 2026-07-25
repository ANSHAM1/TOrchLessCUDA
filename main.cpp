#include "module.hpp"
#include "optimizer.hpp"

#include "DataLoader.hpp"
#include "MNISTLoader.hpp"




int main() {
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



    Tensor Images =
        load_mnist_images(
            "D:/PROJECTs/TorchLess/Dataset/train-images.idx3-ubyte"
        );

    Tensor Labels =
        load_mnist_labels(
            "D:/PROJECTs/TorchLess/Dataset/train-labels.idx1-ubyte"
        );

    LoadImages imageLoader(Images, 32);
    LoadLabels labelLoader(Labels, 32);

    ExecutionContext ctx;
    Adam adam(1e-3f);

    while (imageLoader.hasNext() && labelLoader.hasNext()) {
        Tensor batch = imageLoader.next();
        Tensor label = labelLoader.next();

        model.Train(ctx, adam,  batch, label);
    }


 /*   DataLoader loader(Images, 32);

    ExecutionContext ctx;*/
 /*   size_t batch_id = 0;
    while (loader.hasNext()) {
        Tensor batch = loader.next();

        Tensor& prediction = model.Predict(ctx, batch);

        if (batch_id < 3)
            prediction.debug_print("prediction", 10);

        batch_id++;
    }

    std::cout << "Total batches: " << batch_id << std::endl;*/

    return 0;
}