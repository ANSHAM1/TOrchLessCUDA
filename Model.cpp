#include "Includes/Cnn.h"
#include "Images/ImgIncoder.h"

#include <vector>
#include <random>
#include <iostream>
using namespace std;


int main() {
    vector<vector<vector<vector<float>>>> inputs, labels;

    loadMNISTImages("Dataset/train-images-idx3-ubyte", inputs, 60000);
    loadMNISTLabels("Dataset/train-labels-idx1-ubyte", labels, 10, 60000);

    Network Model;
    Model.Input(inputs.size(), inputs[0].size(), inputs[0][0].size(), inputs[0][0][0].size(), 600);

    Model.Conv2D(32, 1, 3, 3, 1, 1);  
    Model.Activation("relu");

    Model.Conv2D(64, 32, 3, 3, 1, 1);
    Model.Activation("relu");

    Model.Pooling(2, 2, 2, 0, "max"); 

    Model.Conv2D(128, 64, 3, 3, 1, 1); 
    Model.Activation("relu");

    Model.Pooling(2, 2, 2, 0, "max");  

    Model.Dropout(0.5f);   

    Model.Dense(10);   


    int epochs = 4;
    float lr = 0.01f;
    float r2strength = 0.0001f;
    float clipVal = 1;

    Model.Train(inputs, labels, epochs, lr, r2strength, clipVal);
    //Model.Test(inputs, labels);

    //Model.showEvaluation();
}

//ResNet - 18 Inspired(no actual skip for now)
//Network Model;
//Model.Input(inputs);
//Model.Conv2D(64, 3, 7, 7, 2, 3);     // (B, 64, 10, 10)
//Model.BatchNorm();
//Model.Activation("relu");
//Model.Pooling(3, 3, 2, 1, "max");    // (B, 64, 5, 5)
//
//// "Residual Block" without skip
//Model.Conv2D(64, 64, 3, 3, 1, 1);
//Model.BatchNorm();
//Model.Activation("relu");
//Model.Conv2D(64, 64, 3, 3, 1, 1);
//Model.BatchNorm();
//Model.Activation("relu");
//
//// Output
//Model.Dense(5);
//Model.Output(labels);

//EfficientNet - B0(Simplified Version)
//Network Model;
//Model.Input(inputs);
//Model.Conv2D(32, 3, 3, 3, 1, 1);
//Model.BatchNorm();
//Model.Activation("swish");
//
//// Depthwise-like block
//Model.Conv2D(32, 1, 3, 3, 1, 1);  // simulate depthwise
//Model.BatchNorm();
//Model.Activation("swish");
//
//Model.Conv2D(64, 32, 1, 1, 1, 0); // pointwise
//Model.BatchNorm();
//Model.Activation("swish");
//
//Model.Pooling(2, 2, 2, 0, "max");
//
//Model.Conv2D(128, 64, 1, 1, 1, 0);
//Model.BatchNorm();
//Model.Activation("swish");
//
//Model.Dropout(0.3f);
//Model.Dense(5);
//Model.Output(labels);

//VGG16 Inspired(Stacked Conv + Pool)
//Network Model;
//Model.Input(inputs); // (B, 3, 20, 20)
//
//Model.Conv2D(64, 3, 3, 3, 1, 1);
//Model.Activation("relu");
//Model.Conv2D(64, 64, 3, 3, 1, 1);
//Model.Activation("relu");
//Model.Pooling(2, 2, 2, 0, "max");  // (B, 64, 10, 10)
//
//Model.Conv2D(128, 64, 3, 3, 1, 1);
//Model.Activation("relu");
//Model.Conv2D(128, 128, 3, 3, 1, 1);
//Model.Activation("relu");
//Model.Pooling(2, 2, 2, 0, "max");  // (B, 128, 5, 5)
//
//Model.Dropout(0.5f);
//Model.Dense(5);
//Model.Output(labels);
