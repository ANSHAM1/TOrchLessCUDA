#include "../macros.hpp"
#include "../../include/kernel.cuh"

#include "cuda_runtime.h"

#include <iostream>
#include <vector>
#include <numeric>
#include <stdexcept>


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

class Storage {
private:

    float* ptr_;
    size_t count_;

public:

    Storage() noexcept : ptr_(nullptr), count_(0) {}

    // Move Constructor
    Storage(Storage&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    // Move Assignment
    Storage& operator=(Storage&& other) noexcept {
        if (this != &other) {
            release();

            ptr_ = other.ptr_;
            count_ = other.count_;

            other.ptr_ = nullptr;
            other.count_ = 0;
        }

        return *this;
    }

    void allocate(size_t count) {
        release();

        if (count == 0) return;

        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void**>(&ptr_),count * sizeof(float))
        );

        if (!ptr_) 
            throw std::bad_alloc();

        count_ = count;
    }

    void release() noexcept {
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
    float* ptr() noexcept {
        return ptr_;
    }

    [[nodiscard]]
    const float* ptr() const noexcept {
        return ptr_;
    }

    [[nodiscard]]
    size_t size() const noexcept {
        return count_;
    }

    Storage(const Storage&) = delete;

    Storage& operator=(const Storage&) = delete;

    ~Storage() {
        release();
    }
};




// ============================================================
// Tensor
// ============================================================

class Tensor {
private:

    Storage storage_;

public:

    std::vector<size_t> shape_;
    std::vector<size_t> strides_;

    Tensor() = default;

    explicit Tensor(const std::vector<size_t>& shape, cudaStream_t stream = 0) 
        : shape_(shape), strides_(compute_strides(shape)) {

        size_t n = numel();

        if (n > 0)
            storage_.allocate(n);
    }

    explicit Tensor(const std::vector<size_t>& shape, const float* raw_ptr, bool from_host = true, cuda)
        : shape_(shape), strides_(compute_strides(shape)) {

        if (raw_ptr == nullptr)
            throw std::runtime_error("raw_ptr is null");

        size_t n = numel();
        storage_.allocate(n);

        if (from_host)
            CUDA_CHECK(
                cudaMemcpyAsync(storage_.ptr(), raw_ptr, n * sizeof(float), cudaMemcpyHostToDevice, 0);
            );
        else
            CUDA_CHECK(
                cudaMemcpyAsync(storage_.ptr(), raw_ptr, n * sizeof(float), cudaMemcpyDeviceToDevice, 0);
            );
    }

    // Move Constructor
    Tensor(Tensor&& other) noexcept
        : storage_(std::move(other.storage_)), shape_(std::move(other.shape_)), strides_(std::move(other.strides_)) {

        other.stream_ = 0;
    }

    // Move Assignment
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            storage_ = std::move(other.storage_);

            shape_ = std::move(other.shape_);
            strides_ = std::move(other.strides_);
        }

        return *this;
    }


    Tensor(const Tensor&) = delete;

    Tensor& operator=(const Tensor&) = delete;


    static Tensor zeros(const std::vector<size_t>& shape) {
        Tensor t(shape);

        if (t.numel() == 0)
            return t;

        CUDA_CHECK(
            cudaMemsetAsync(t.data(), 0, t.numel() * sizeof(float), 0);
        );

        return t;
    }


    [[nodiscard]]
    size_t dim() const noexcept {
        return shape_.size();
    }

    [[nodiscard]]
    size_t numel() const noexcept {
        return shape_product(shape_);
    }

    [[nodiscard]]
    bool allocated() const noexcept {
        return storage_.size() > 0;
    }

    [[nodiscard]]
    float* data() noexcept {
        return storage_.ptr();
    }

    [[nodiscard]]
    const float* data() const noexcept {
        return storage_.ptr();
    }


    void resize(const std::vector<size_t>& new_shape) {
        shape_ = new_shape;

        strides_ = compute_strides(shape_);

        storage_.allocate(numel());
    }

    [[nodiscard]]
    float* offset_ptr(size_t offset = 0) {
        if (offset >= storage_.size())
            throw std::out_of_range(
                "Tensor offset out of range"
            );


        return storage_.ptr() + offset;
    }

    [[nodiscard]]
    const float* offset_ptr(size_t offset = 0) const {
        if (offset >= storage_.size())
            throw std::out_of_range(
                "Tensor offset out of range"
            );


        return storage_.ptr() + offset;
    }

    void fill(float value) {
        size_t n = numel();

        if (n == 0)
            return;

        if (value == 0.0f)
            CUDA_CHECK(
                cudaMemsetAsync(data(), 0, n * sizeof(float), 0);
            );
        else
            tensorAssign(data(), value, n);
    }

    void randomTensor(unsigned long long seed) {
        tensorAssignRandom(data(), numel(), seed);
    }

    [[nodiscard]]
    void* raw_ptr() noexcept {
        return static_cast<void*>(data());
    }

    [[nodiscard]]
    const void* raw_ptr() const noexcept {
        return static_cast<const void*>(data());
    }


    void debug_print(const char* name = "") const {
        std::cerr << "[Tensor " << name << "] shape=(";

        for (size_t i = 0; i < shape_.size(); ++i) {
            if (i)
                std::cerr << ",";

            std::cerr << shape_[i];
        }

        std::cerr << ") numel=" << numel() << " bytes=" << numel() * sizeof(float) << "\n";
    }
};