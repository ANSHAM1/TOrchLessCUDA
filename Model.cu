#include "Core/Cnn.cuh"
#include <fstream>
#include <vector>
#include <stdexcept>
#include <iostream>

#include <thread>
#include <chrono>

inline uint32_t swap_endian(uint32_t val) {
	val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0x00FF00FF);
	return (val << 16) | (val >> 16);
}

template<FloatingTensorType T>
ARCH::Tensor<T> load_mnist_images(const std::string& path) {
	std::ifstream file(path, std::ios::binary);
	if (!file.is_open()) {
		throw std::runtime_error("Cannot open file: " + path);
	}

	int32_t magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;

	file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
	magic_number = swap_endian(magic_number);
	if (magic_number != 2051) {
		throw std::runtime_error("Invalid MNIST image file magic number.");
	}

	file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
	num_images = swap_endian(num_images);
	file.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
	num_rows = swap_endian(num_rows);
	file.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));
	num_cols = swap_endian(num_cols);

	std::cout << "Loading " << num_images << " images of size "
		<< num_rows << "x" << num_cols << std::endl;

	size_t total_pixels = static_cast<size_t>(num_images) * num_rows * num_cols;

	std::vector<T> normalized_host_buffer(total_pixels);

	for (size_t i = 0; i < total_pixels; ++i) {
		unsigned char pixel;
		file.read(reinterpret_cast<char*>(&pixel), 1);
		normalized_host_buffer[i] = static_cast<T>(static_cast<float>(pixel) / 255.0f);

	}

	std::vector<size_t> shape = { (size_t)num_images, 1, (size_t)num_rows, (size_t)num_cols };

	return ARCH::Tensor<T>(shape, normalized_host_buffer.data());
}


template<FloatingTensorType T>
ARCH::Tensor<T> load_mnist_labels(const std::string& path) {
	std::ifstream file(path, std::ios::binary);
	if (!file.is_open()) {
		throw std::runtime_error("Cannot open file: " + path);
	}

	int32_t magic_number = 0, num_labels = 0;
	file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
	magic_number = swap_endian(magic_number);
	if (magic_number != 2049) {
		throw std::runtime_error("Invalid MNIST label file magic number.");
	}

	file.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));
	num_labels = swap_endian(num_labels);

	std::cout << "Loading " << num_labels << " labels." << std::endl;

	std::vector<T> one_hot_host(num_labels * 10, T(0));

	for (int i = 0; i < num_labels; ++i) {
		unsigned char lbl;
		file.read(reinterpret_cast<char*>(&lbl), 1);
		if (lbl < 10) {
			one_hot_host[i * 10 + lbl] = T(1);
		}
	}

	std::vector<size_t> shape = { (size_t)num_labels, 10, 1, 1 };

	return ARCH::Tensor<T>(shape, one_hot_host.data());
}



int main() {
	auto ImageTrain = load_mnist_images<__half>("Dataset/train-images.idx3-ubyte");
	auto LabelTrain = load_mnist_labels<__half>("Dataset/train-labels.idx1-ubyte");

	ARCH::Sequential<__half> Model;
	Model.Input({ 60000, 1, 28, 28 }, 100);

	Model.Conv2D({ 32, 1, 3, 3 }, 1, 1);
	Model.Activate("relu");

	Model.Conv2D({ 64, 32, 3, 3 }, 1, 1);
	Model.Activate("relu");

	Model.Conv2D({ 128, 64, 3, 3 }, 1, 1);
	Model.Activate("relu");

	Model.MaxPooling(2, 2);

	Model.Dense(1024);
	Model.Activate("relu");

	Model.Dense(10);
	Model.Output();

	auto& p = Model.Predict(ImageTrain);

	std::cout << "Prediction shape: " << p.shape()[0] << std::endl;

	std::cout << "Program finished. Keeping alive for 20 seconds...\n";
	std::this_thread::sleep_for(std::chrono::seconds(20));
	return 0;
}