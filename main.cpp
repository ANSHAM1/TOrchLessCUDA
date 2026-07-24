#include "layer.hpp"
#include "DataLoader.hpp"
#include "MNISTLoader.hpp"


int main() {
    Tensor Images =
        load_mnist_images(
            //"Dataset/train-images.idx3-ubyte"
            "D:/PROJECTs/TorchLess/Dataset/train-images.idx3-ubyte"
        );


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


    DataLoader loader(Images, 32);

    ExecutionContext ctx;
    size_t batch_id = 0;
    while (loader.hasNext()) {
        Tensor batch = loader.next();

        Tensor& prediction = model.Predict(ctx, batch);

        if (batch_id < 3)
            prediction.debug_print("prediction", 10);

        batch_id++;
    }

    std::cout << "Total batches: " << batch_id << std::endl;

    return 0;
}