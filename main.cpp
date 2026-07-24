#include "layer.hpp"

#include <iostream>


int main() {
    Sequential model;


    model.Input({ 32,3,224,224 }, 32);


    model.Conv2D({ 64,3,7,7 }, 2, 3);
    model.ReLU();

    model.MaxPooling(3, 3, 2, 2);


    model.Conv2D({ 128,64,3,3 }, 1, 1);
    model.ReLU();


    model.Flatten();


    model.Dense(256);
    model.ReLU();


    model.Dense(10);

    model.Output("Softmax");

    return 0;
}