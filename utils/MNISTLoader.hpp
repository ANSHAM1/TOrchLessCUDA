#pragma once

#include "tensor.hpp"

#include <fstream>
#include <vector>
#include <iostream>
#include <stdexcept>




inline uint32_t swap_endian(uint32_t val) {
    return ((val & 0x000000FF) << 24) | ((val & 0x0000FF00) << 8) | ((val & 0x00FF0000) >> 8) | ((val & 0xFF000000) >> 24);
}




inline Tensor load_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);

    if (!file)
        throw std::runtime_error("Cannot open image file");

    uint32_t magic;
    uint32_t count;
    uint32_t rows;
    uint32_t cols;


    file.read(reinterpret_cast<char*>(&magic), 4);

    magic = swap_endian(magic);
    if (magic != 2051) 
        throw std::runtime_error("Invalid MNIST image file");


    file.read(reinterpret_cast<char*>(&count), 4);
    file.read(reinterpret_cast<char*>(&rows), 4);
    file.read(reinterpret_cast<char*>(&cols), 4);


    count = swap_endian(count);
    rows = swap_endian(rows);
    cols = swap_endian(cols);

    std::cout << "Images: " << count << " " << rows << "x" << cols << std::endl;

    size_t total = static_cast<size_t>(count) * rows * cols;


    std::vector<float> host(total);
    for (size_t i = 0; i < total; i++) {
        unsigned char pixel;

        file.read(reinterpret_cast<char*>(&pixel), 1);

        host[i] = static_cast<float>(pixel) / 255.0f;
    }

    return Tensor({count, 1, rows, cols }, host.data());
}




inline Tensor load_mnist_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);

    if (!file)
        throw std::runtime_error("Cannot open label file");

    uint32_t magic;
    uint32_t count;

    file.read(reinterpret_cast<char*>(&magic), 4);

    magic = swap_endian(magic);
    if (magic != 2049)
        throw std::runtime_error("Invalid MNIST label file");

    file.read(reinterpret_cast<char*>(&count), 4);

    count = swap_endian(count);

    std::cout << "Label Idx: " << count << std::endl;

    std::vector<int> labels(count);
    for (size_t i = 0; i < count; i++) {
        unsigned char label;

        file.read(reinterpret_cast<char*>(&label), 1);

        labels[i] = static_cast<int>(label);
    }

    Tensor TensorLabels({count}, DataType::Int32);
    TensorLabels.copyFromHost(labels.data(), count);


    return TensorLabels;
}