#pragma once

#include "macros.hpp"
#include "kernel.cuh"

#include "cuda_runtime.h"

#include <vector>
#include <utility>
#include <stdexcept>
#include <new>




class Storage {

private:

    float* ptr_;
    size_t count_;

public:

    Storage() noexcept;

    // Move Constructor
    Storage(Storage&& other) noexcept;

    // Move Assignment
    Storage& operator=(Storage&& other) noexcept;

    void allocate(size_t count);

    void release() noexcept;


    /*
        * [[nodiscard]] is a C++17 attribute (also in C++20/23) that tells the compiler:
        * If the return value of this function (or type) is ignored(not assigned to variable), issue a warning.
    */

    [[nodiscard]]
    float* ptr() noexcept;

    [[nodiscard]]
    const float* ptr() const noexcept;

    [[nodiscard]]
    size_t size() const noexcept;

    [[nodiscard]]
    size_t bytes() const noexcept;



    Storage(const Storage&) = delete;

    Storage& operator=(const Storage&) = delete;

    ~Storage();
};




enum class DataType {
    Float32,
    Int32
};




class Tensor {

private:

    Storage storage_;

    float* data_ptr_ = nullptr;

    std::vector<size_t> shape_;
    std::vector<size_t> strides_;

    bool is_view_ = false;


    DataType dtype_ = DataType::Float32;

private:

    explicit Tensor(float* ptr, const std::vector<size_t>& shape, const std::vector<size_t>& strides, DataType dtype = DataType::Float32);

public:

    static size_t shape_product(const std::vector<size_t>& shape);

    static std::vector<size_t> compute_strides(const std::vector<size_t>& shape);


    Tensor() = default;


    explicit Tensor(const std::vector<size_t>& shape, DataType dtype = DataType::Float32);

    explicit Tensor(const std::vector<size_t>& shape, const float* raw_ptr, bool from_host = true);

    // Move Constructor
    Tensor(Tensor&& other) noexcept;

    // Move Assignment
    Tensor& operator=(Tensor&& other) noexcept;

    Tensor(const Tensor&) = delete;

    Tensor& operator=(const Tensor&) = delete;

    void allocate(const std::vector<size_t>& shape);

    static Tensor zeros(const std::vector<size_t>& shape);

    void reshape(const std::vector<size_t>& new_shape);

    void fill(float value);



    [[nodiscard]]
    Tensor view(const std::vector<size_t>& new_shape) const;

    [[nodiscard]]
    const std::vector<size_t>& shape() const;

    [[nodiscard]]
    const std::vector<size_t>& strides() const;

    [[nodiscard]]
    size_t dim() const noexcept;

    [[nodiscard]]
    size_t numel() const noexcept;

    [[nodiscard]]
    bool allocated() const noexcept;

    [[nodiscard]]
    float* data() noexcept;

    [[nodiscard]]
    const float* data() const noexcept;

    [[nodiscard]]
    int* int_data() noexcept;

    [[nodiscard]]
    const int* int_data() const noexcept;

    [[nodiscard]]
    bool is_view() const noexcept;



    void copyFromHost(const float* data, size_t count);

    void copyToHost(float* data, size_t count) const;

    void copyFromHost(const int* data, size_t count);

    void copyToHost(int* data, size_t count) const;

    void debug_print(const char* name, size_t elements) const;

    ~Tensor() = default;
};