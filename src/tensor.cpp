#include "tensor.hpp"

#include <iostream>
#include <numeric>
#include <functional>

// ============================================================
// Utility Functions
// ============================================================

inline static size_t shape_product(const std::vector<size_t>& shape) {
    if (shape.empty())
        return 0;

    return std::accumulate(
        shape.begin(),
        shape.end(),
        size_t(1),
        std::multiplies<size_t>()
    );
}


inline static std::vector<size_t> compute_strides(const std::vector<size_t>& shape) {
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
Tensor::Tensor(Tensor&& other) noexcept
    : storage_(std::move(other.storage_)), shape_(std::move(other.shape_)), strides_(std::move(other.strides_)) {
}


// Move Assignment
Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        storage_ = std::move(other.storage_);

        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
    }

    return *this;
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
    return storage_.size() > 0;
}


[[nodiscard]]
float* Tensor::data() noexcept {
    return storage_.ptr();
}


[[nodiscard]]
const float* Tensor::data() const noexcept {
    return storage_.ptr();
}


void Tensor::resize(const std::vector<size_t>& new_shape) {
    shape_ = new_shape;

    strides_ = compute_strides(shape_);

    storage_.allocate(numel());
}


[[nodiscard]]
float* Tensor::offset_ptr(size_t offset) {
    if (offset >= storage_.size())
        throw std::out_of_range(
            "Tensor offset out of range"
        );


    return storage_.ptr() + offset;
}


[[nodiscard]]
const float* Tensor::offset_ptr(size_t offset) const {
    if (offset >= storage_.size())
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


[[nodiscard]]
void* Tensor::raw_ptr() noexcept {
    return static_cast<void*>(data());
}


[[nodiscard]]
const void* Tensor::raw_ptr() const noexcept {
    return static_cast<const void*>(data());
}


void Tensor::debug_print(const char* name) const {
    std::cerr << "[Tensor " << name << "] shape=(";

    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i)
            std::cerr << ",";

        std::cerr << shape_[i];
    }

    std::cerr << ") numel=" << numel() << " bytes=" << numel() * sizeof(float) << "\n";
}