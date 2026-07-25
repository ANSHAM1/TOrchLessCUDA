#include "tensor.hpp"

#include <iostream>
#include <numeric>
#include <functional>
#include <algorithm>
#include <cmath>



Storage::Storage() noexcept : ptr_(nullptr), count_(0) {}



// Move Constructor
Storage::Storage(Storage&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
}



// Move Assignment
Storage& Storage::operator=(Storage&& other) noexcept {
    if (this != &other) {
        release();

        ptr_ = other.ptr_;
        count_ = other.count_;

        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    return *this;
}




void Storage::allocate(size_t count) {
    release();

    if (count == 0) return;

    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(float))
    );

    if (!ptr_)
        throw std::bad_alloc();

    count_ = count;
}




void Storage::release() noexcept {
    if (ptr_) {
        cudaFree(ptr_);
        ptr_ = nullptr;
        count_ = 0;
    }
}




/*
    * [[nodiscard]] is a C++17 attribute (also in C++20/23) that tells the compiler:
    * If the return value of this function (or type) is ignored(not assigned to variable), issue a warning.
*/

[[nodiscard]]
float* Storage::ptr() noexcept {
    return ptr_;
}




[[nodiscard]]
const float* Storage::ptr() const noexcept {
    return ptr_;
}




[[nodiscard]]
size_t Storage::size() const noexcept {
    return count_;
}




[[nodiscard]]
size_t Storage::bytes() const noexcept {
    return count_ * sizeof(float);
}




Storage::~Storage() {
    release();
}






size_t Tensor::shape_product(const std::vector<size_t>& shape) {
    if (shape.empty())
        return 0;

    return std::accumulate(
        shape.begin(),
        shape.end(),
        size_t(1),
        std::multiplies<size_t>()
    );
}




std::vector<size_t> Tensor::compute_strides(const std::vector<size_t>& shape) {
    std::vector<size_t> strides(shape.size());

    if (shape.empty())
        return strides;


    strides.back() = 1;

    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i)
    {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    return strides;
}




Tensor::Tensor(float* ptr, const std::vector<size_t>& shape, const std::vector<size_t>& strides, DataType dtype)
    : data_ptr_(ptr), shape_(shape), strides_(strides), is_view_(true), dtype_(dtype) {

    if (ptr == nullptr)
        throw std::runtime_error("Cannot create view from null pointer");
}




Tensor::Tensor(const std::vector<size_t>& shape, DataType dtype)
    : shape_(shape), strides_(compute_strides(shape)), dtype_(dtype) {

    size_t n = numel();

    if (n > 0)
        storage_.allocate(n);
}




Tensor::Tensor(const std::vector<size_t>& shape, const float* raw_ptr, bool from_host)
    : shape_(shape), strides_(compute_strides(shape)) {

    if (raw_ptr == nullptr)
        throw std::runtime_error("raw_ptr is null");

    size_t n = numel();
    storage_.allocate(n);

    if (from_host)
        CUDA_CHECK(
            cudaMemcpy(storage_.ptr(), raw_ptr, n * sizeof(float), cudaMemcpyHostToDevice)
        );
    else
        CUDA_CHECK(
            cudaMemcpy(storage_.ptr(), raw_ptr, n * sizeof(float), cudaMemcpyDeviceToDevice)
        );
}



// Move Constructor
Tensor::Tensor(Tensor&& other) noexcept : storage_(std::move(other.storage_)), data_ptr_(other.data_ptr_), 
    shape_(std::move(other.shape_)), strides_(std::move(other.strides_)), is_view_(other.is_view_) {

    other.data_ptr_ = nullptr;
    other.is_view_ = false;
}



// Move Assignment
Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        storage_ = std::move(other.storage_);

        data_ptr_ = other.data_ptr_;
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);

        is_view_ = other.is_view_;


        other.data_ptr_ = nullptr;
        other.is_view_ = false;
    }

    return *this;
}




void Tensor::allocate(const std::vector<size_t>& shape) {
    if (is_view_)
        throw std::runtime_error("Cannot allocate memory for a Tensor view");

    size_t required = shape_product(shape);

    if (required > storage_.size())
        storage_.allocate(required);

    shape_ = shape;
    strides_ = compute_strides(shape_);
}




Tensor Tensor::zeros(const std::vector<size_t>& shape) {
    Tensor t(shape);

    if (t.numel() == 0)
        return t;

    CUDA_CHECK(
        cudaMemset(t.data(), 0, t.numel() * sizeof(float))
    );

    return t;
}




void Tensor::reshape(const std::vector<size_t>& new_shape) {
    size_t required = shape_product(new_shape);

    size_t available = is_view_ ? numel() : storage_.size();

    if (required > available)
        throw std::runtime_error("Tensor::reshape(): insufficient memory.");

    shape_ = new_shape;
    strides_ = compute_strides(shape_);
}





void Tensor::fill(float value) {
    size_t n = numel();

    if (n == 0)
        return;

    if (value == 0.0f)
        CUDA_CHECK(
            cudaMemset(data(), 0, n * sizeof(float))
        );
    else
        tensorAssign(data(), value, n);
}




[[nodiscard]]
Tensor Tensor::view(const std::vector<size_t>& new_shape) const {
    if (shape_product(new_shape) != numel())
        throw std::runtime_error("Invalid view shape");
    
    return Tensor(const_cast<float*>(data()), new_shape, compute_strides(new_shape), dtype_);
}




[[nodiscard]]
const std::vector<size_t>& Tensor::shape() const {
    return shape_;
}




[[nodiscard]]
const std::vector<size_t>& Tensor::strides() const {
    return strides_;
}




[[nodiscard]]
size_t Tensor::dim() const noexcept {
    return shape_.size();
}




[[nodiscard]]
size_t Tensor::numel() const noexcept {
    return shape_product(shape_);
}




[[nodiscard]]
bool Tensor::allocated() const noexcept {
    return is_view_ || storage_.size() > 0;
}




[[nodiscard]]
float* Tensor::data() noexcept {
    if (is_view_)
        return data_ptr_;

    return storage_.ptr();
}




[[nodiscard]]
const float* Tensor::data() const noexcept {
    if (is_view_)
        return data_ptr_;

    return storage_.ptr();
}




[[nodiscard]]
int* Tensor::int_data() noexcept{
    if (is_view_)
        return reinterpret_cast<int*>(data_ptr_);

    return reinterpret_cast<int*>(storage_.ptr());
}




[[nodiscard]]
const int* Tensor::int_data() const noexcept {
    if (is_view_)
        return reinterpret_cast<const int*>(data_ptr_);

    return reinterpret_cast<const int*>(storage_.ptr());
}




[[nodiscard]]
bool Tensor::is_view() const noexcept {
    return is_view_;
}




void Tensor::copyFromHost(const float* data, size_t count) {
    if (dtype_ != DataType::Float32)
        throw std::runtime_error("Tensor is not Float32");

    if (count > numel())
        throw std::runtime_error("copyFromHost: size exceeds tensor capacity");

    CUDA_CHECK(
        cudaMemcpy(this->data(), data, count * sizeof(float), cudaMemcpyHostToDevice)
    );
}




void Tensor::copyToHost(float* data, size_t count) const {
    if (dtype_ != DataType::Float32)
        throw std::runtime_error("Tensor is not Float32");

    if (count > numel())
        throw std::runtime_error("copyToHost: size exceeds tensor capacity");

    CUDA_CHECK(
        cudaMemcpy(data, this->data(), count * sizeof(float), cudaMemcpyDeviceToHost)
    );
}




void Tensor::copyFromHost(const int* data, size_t count) {
    if (dtype_ != DataType::Int32)
        throw std::runtime_error("Tensor is not Int32");

    if (count > numel())
        throw std::runtime_error("copyToHost: size exceeds tensor capacity");

    CUDA_CHECK(
        cudaMemcpy(int_data(), data, count * sizeof(int), cudaMemcpyHostToDevice)
    );
}




void Tensor::copyToHost(int* data, size_t count) const {
    if (dtype_ != DataType::Int32)
        throw std::runtime_error("Tensor is not Int32");

    if (count > numel())
        throw std::runtime_error("copyToHost: size exceeds tensor capacity");

    CUDA_CHECK(
        cudaMemcpy(data, int_data(), count * sizeof(int), cudaMemcpyDeviceToHost)
    );
}




void Tensor::debug_statistics_print(const char* name) const {
    std::vector<float> host(numel());

    copyToHost(host.data(), numel());

    float minValue = FLT_MAX;
    float maxValue = -FLT_MAX;

    double sum = 0.0;
    double squareSum = 0.0;

    size_t zeroCount = 0;
    size_t nonZeroCount = 0;

    for (float value : host) {
        minValue = std::min(minValue, value);
        maxValue = std::max(maxValue, value);

        sum += value;
        squareSum += static_cast<double>(value) * value;


        if (value == 0.0f)
            zeroCount++;

        else
            nonZeroCount++;
    }


    float mean = static_cast<float>(sum / numel());

    float variance = static_cast<float>((squareSum / numel()) - (mean * mean));

    float stdDev = sqrtf(std::max(variance, 0.0f));

    std::sort(host.begin(), host.end());

    float median;
    if (numel() % 2 == 0)
        median = (host[numel() / 2 - 1] + host[numel() / 2]) * 0.5f;
    
    else
        median = host[numel() / 2];


    std::cout << "[Tensor Statistics: " << name << "]\n";

    std::cout << "Shape: ";

    for (size_t i = 0; i < shape_.size(); i++) {
        std::cout << shape_[i];

        if (i + 1 < shape_.size())
            std::cout << "x";
    }

    std::cout << "\n";

    std::cout << "Elements: " << numel() << "\n";

    std::cout << "Min: " << minValue << "\n";

    std::cout << "Max: " << maxValue << "\n";

    std::cout << "Mean: " << mean << "\n";

    std::cout << "Median: " << median << "\n";

    std::cout << "StdDev: " << stdDev  << "\n";

    std::cout << "Zero: " << zeroCount << " (" << (static_cast<float>(zeroCount) / numel()) * 100.0f << "%)"  << "\n";

    std::cout << "Non Zero: " << nonZeroCount << "\n";

    std::cout << std::endl;
}