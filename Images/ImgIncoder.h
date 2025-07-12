#pragma once

#include <iostream>
#include <fstream>
#include <vector>
using namespace std;

uint32_t readBigEndianUInt32(std::ifstream& file) {
    uint32_t val = 0;
    file.read(reinterpret_cast<char*>(&val), sizeof(val));
    return ((val & 0xFF) << 24) |
        ((val & 0xFF00) << 8) |
        ((val & 0xFF0000) >> 8) |
        ((val & 0xFF000000) >> 24);
}

void loadMNISTImages(const std::string& path,
    std::vector<std::vector<std::vector<std::vector<float>>>>& images,
    uint32_t max_images = 0) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Could not open image file!");

    uint32_t magic = readBigEndianUInt32(file);
    uint32_t num_images = readBigEndianUInt32(file);
    uint32_t rows = readBigEndianUInt32(file);
    uint32_t cols = readBigEndianUInt32(file);

    if(max_images != 0) num_images = min(num_images, max_images);

    images.resize(num_images, std::vector<std::vector<std::vector<float>>>(
        1, std::vector<std::vector<float>>(
            1, std::vector<float>(rows * cols))));

    for (uint32_t i = 0; i < num_images; ++i) {
        for (uint32_t j = 0; j < rows * cols; ++j) {
            unsigned char pixel = 0;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            images[i][0][0][j] = static_cast<float>(pixel) / 255.0f;
        }
    }
}

void loadMNISTLabels(const std::string& path,
    std::vector<std::vector<std::vector<std::vector<float>>>>& onehot_labels, int num_classes = 10,
    uint32_t max_labels = 0) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Could not open label file!");

    uint32_t magic = readBigEndianUInt32(file);
    uint32_t num_labels = readBigEndianUInt32(file);

    if (num_labels != 0) num_labels = min(num_labels, max_labels);

    onehot_labels.resize(num_labels, std::vector<std::vector<std::vector<float>>>(
        1, std::vector<std::vector<float>>(
            1, std::vector<float>(num_classes, 0.0f))));

    for (uint32_t i = 0; i < num_labels; ++i) {
        unsigned char label = 0;
        file.read(reinterpret_cast<char*>(&label), 1);
        onehot_labels[i][0][0][label] = 1.0f;
    }
}


int getLabelIndex(const std::vector<float>& oneHot) {
    for (size_t i = 0; i < oneHot.size(); ++i) {
        if (oneHot[i] > 0.5f) return static_cast<int>(i);
    }
    return -1;
}


void printVectorImage(const std::vector<float>& flatImage, int label) {
    std::cout << "Label: " << label << "\n[";
    for (size_t i = 0; i < flatImage.size(); ++i) {
        std::cout << std::fixed << std::setprecision(1) << flatImage[i];
        if (i < flatImage.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n----------------------\n";
}

void printVectorBatch(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& inputs,
    const std::vector<std::vector<std::vector<std::vector<float>>>>& labels,
    int count = 5)
{
    for (int i = 0; i < std::min((int)inputs.size(), count); ++i) {
        const auto& image = inputs[i][0][0];
        const auto& oneHot = labels[i][0][0];
        int label = getLabelIndex(oneHot);
        printVectorImage(image, label);
    }
}