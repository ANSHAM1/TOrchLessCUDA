#include "tensor.hpp"

#include <iostream>
#include <numeric>
#include <functional>



// ============================================================
// CUDA Memory Storage
// ============================================================

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


Storage::~Storage() {
    release();
}




// ============================================================
// Tensor
// ============================================================

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


Tensor::Tensor(float* ptr, const std::vector<size_t>& shape, const std::vector<size_t>& strides)
    : data_ptr_(ptr), shape_(shape), strides_(strides), is_view_(true) {

    if (ptr == nullptr)
        throw std::runtime_error("Cannot create view from null pointer");
}


Tensor::Tensor(const std::vector<size_t>& shape)
    : shape_(shape), strides_(compute_strides(shape)) {

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


[[nodiscard]]
Tensor Tensor::view(const std::vector<size_t>& new_shape) const {
    if (shape_product(new_shape) != numel())
        throw std::runtime_error("Invalid view shape");
    
    return Tensor(const_cast<float*>(data()), new_shape, compute_strides(new_shape));
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
bool Tensor::is_view() const noexcept {
    return is_view_;
}


void Tensor::reshape(const std::vector<size_t>& new_shape) {
    size_t required = shape_product(new_shape);

    size_t available = is_view_ ? numel() : storage_.size();

    if (required > available)
        throw std::runtime_error("Tensor::reshape(): insufficient memory.");
    
    shape_ = new_shape;
    strides_ = compute_strides(shape_);
}


[[nodiscard]]
float* Tensor::offset_ptr(size_t offset) {
    if (offset >= numel())
        throw std::out_of_range(
            "Tensor offset out of range"
        );

    return storage_.ptr() + offset;
}


[[nodiscard]]
const float* Tensor::offset_ptr(size_t offset) const {
    if (offset >= numel())
        throw std::out_of_range(
            "Tensor offset out of range"
        );

    return storage_.ptr() + offset;
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


void Tensor::randomTensor(unsigned long long seed) {
    tensorAssignRandom(data(), numel(), seed);
}


void Tensor::copyFromHost(const float* data, size_t count) {
    if (count > numel())
        throw std::runtime_error("copyFromHost: size exceeds tensor capacity");

    CUDA_CHECK(
        cudaMemcpy(this->data(), data, count * sizeof(float), cudaMemcpyHostToDevice)
    );
}


void Tensor::copyToHost(float* data, size_t count) const {
    if (count > numel())
        throw std::runtime_error("copyToHost: size exceeds tensor capacity");

    CUDA_CHECK(
        cudaMemcpy(data, this->data(), count * sizeof(float), cudaMemcpyDeviceToHost)
    );
}


void Tensor::debug_print(const char* name, size_t elements) const {

    std::cerr << "[Tensor " << name << "] shape=(";

    for (size_t i = 0; i < shape_.size(); ++i)
    {
        if (i)
            std::cerr << ",";

        std::cerr << shape_[i];
    }

    std::cerr << ") numel="
        << numel()
        << " bytes="
        << numel() * sizeof(float)
        << "\n";


    if (elements == 0)
        return;


    elements = std::min(elements, numel());


    std::vector<float> host(elements);

    copyToHost(
        host.data(),
        elements
    );


    std::cout << "Values: ";

    for (size_t i = 0; i < elements; i++)
    {
        std::cout << host[i];

        if (i + 1 < elements)
            std::cout << " ";
    }

    std::cout << "\n";
}