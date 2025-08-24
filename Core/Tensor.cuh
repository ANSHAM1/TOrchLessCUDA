#ifndef __TENSOR_CUH__
#define __TENSOR_CUH__

#include "../Utils/Macros.hpp"
#include "../Cuda/Kernels.cuh"
#include "../Cuda/CudaAPIs.cuh"

#include <stdexcept>
#include <concepts>
#include <vector>
#include <iostream>
#include <numeric>

inline size_t shape_product(const std::vector<size_t>& shape) {
    if (shape.empty()) return 0;
    return std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
}

inline std::vector<size_t> compute_strides(const std::vector<size_t>& shape) {
    std::vector<size_t> strides(shape.size());
    if (shape.empty()) return strides;
    strides.back() = 1;
    for (int i = int(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[static_cast<std::vector<size_t, std::allocator<size_t>>::size_type>(i) + 1]
            * shape[static_cast<std::vector<size_t, std::allocator<size_t>>::size_type>(i) + 1];
    }
    return strides;
}

_AM_START

inline void throw_if(bool cond, const char* msg) {
    if (cond) throw std::runtime_error(msg);
}

#ifdef USE_CUDA
#define CUDA_CHECK(call) do {                              \
    cudaError_t _e = (call);                               \
    if (_e != cudaSuccess) {                               \
        std::string s = std::string("CUDA error: ") +      \
            cudaGetErrorString(_e);                        \
        throw std::runtime_error(s);                       \
    }                                                      \
} while(0)
#else
#define CUDA_CHECK(call) do { (void)call; } while(0)
#endif

template<typename T>
concept FloatingTensorType = std::same_as<T, float> || std::same_as<T, double> || std::same_as<T, __half>;

template<FloatingTensorType T>
class Storage {
private:
    void* ptr_;
    size_t count_;

public:
    Storage() noexcept : ptr_(nullptr), count_(0) {}

    Storage(Storage&& o) noexcept : ptr_(o.ptr_), count_(o.count_) {
        o.ptr_ = nullptr; o.count_ = 0;
    }

    Storage& operator=(Storage&& o) noexcept {
        if (this != &o) {
            release();
            ptr_ = o.ptr_;
            count_ = o.count_;
            o.ptr_ = nullptr;
            o.count_ = 0;
        }
        return *this;
    }

    void allocate(size_t count) {
        release();
        if (count == 0) return;
        count_ = count;
        nvtx3::scoped_range allocRange{ "cudaMalloc Storage" };
        CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
        if (!ptr_) throw std::bad_alloc();
    }

    void release() noexcept {
        if (ptr_) {
            cudaError_t e = cudaFree(ptr_);
            if (e != cudaSuccess) {
#ifndef NDEBUG
                fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(e));
#endif
            }
            ptr_ = nullptr;
            count_ = 0;
        }
    }

    /*
        * [[nodiscard]] is a C++17 attribute (also in C++20/23) that tells the compiler:
        * If the return value of this function (or type) is ignored, issue a warning.
    */

    [[nodiscard]] T* ptr() noexcept { return reinterpret_cast<T*>(ptr_); }
    [[nodiscard]] const T* ptr() const noexcept { return reinterpret_cast<const T*>(ptr_); }
    [[nodiscard]] size_t size() const noexcept { return count_; }

    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;

    ~Storage() { release(); }
};


template<FloatingTensorType T>
class Tensor {
    static_assert(std::is_trivially_copyable<T>::value, "Tensor element type must be trivially copyable");

private:
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    Storage<T> storage_;
    cudaStream_t stream_;

    /*
        * offset_ is used to allow for views of the tensor.
        * stream_ is used to specify the CUDA stream for asynchronous operations.
    */

public:
    Tensor() : shape_(), strides_(), stream_(0) {}

    /*
        * `explicit` prevents the compiler from using this constructor
        * for implicit type conversions. Without it, a single-argument
        * constructor can be called automatically (e.g., int -> Tensor).
        * Marking it `explicit` makes construction intentional and avoids
        * subtle bugs from unexpected conversions.
    */

    explicit Tensor(const std::vector<size_t>& shape, cudaStream_t stream = 0)
        : shape_(shape), strides_(compute_strides(shape)), stream_(stream) {
        size_t n = shape_product(shape_);
        if (n > 0) storage_.allocate(n);
    }

    explicit Tensor(const std::vector<size_t>& shape, const T* raw_ptr, bool from_host = true, cudaStream_t stream = 0)
        : shape_(shape), strides_(compute_strides(shape)), stream_(stream) {
        storage_ = Storage<T>{};
        size_t n = shape_product(shape_);
        storage_.allocate(n);

        if (raw_ptr) {
            nvtx3::scoped_range asyncCopyRange{ "cudaMemcpyAsync raw_ptr -> storage" };
            if (from_host)
                CUDA_CHECK(cudaMemcpyAsync(storage_.ptr(), raw_ptr, n * sizeof(T), cudaMemcpyHostToDevice, stream_));
            else
                CUDA_CHECK(cudaMemcpyAsync(storage_.ptr(), raw_ptr, n * sizeof(T), cudaMemcpyDeviceToDevice, stream_));
        }
        else
            throw std::runtime_error("raw_ptr is a null pointer");
    }

    Tensor(Tensor&& o) noexcept
        : shape_(std::move(o.shape_)), strides_(std::move(o.strides_)),
        storage_(std::move(o.storage_)), stream_(o.stream_) {
        o.stream_ = 0;
    }

    Tensor& operator=(Tensor&& o) noexcept {
        if (this != &o) {
            shape_ = std::move(o.shape_);
            strides_ = std::move(o.strides_);
            storage_ = std::move(o.storage_);
            stream_ = o.stream_;
            o.stream_ = 0;
        }
        return *this;
    }

    Tensor(const Tensor& o) = delete;
    Tensor& operator=(const Tensor& o) = delete;

    static Tensor<T> zeros(const std::vector<size_t>& shape) {
        Tensor<T> t(shape);
        size_t n = t.numel();
        if (n == 0) return t;
        nvtx3::scoped_range memsetRange{ "cudaMemsetAsync: zero from zeros method" };
        CUDA_CHECK(cudaMemsetAsync(t.data(), static_cast<T>(0), n * sizeof(T), t.stream_));
        return t;
    }

    cudaStream_t stream() const noexcept { return stream_; }

    const std::vector<size_t>& shape() const noexcept { return shape_; }

    const std::vector<size_t>& strides() const noexcept { return strides_; }

    size_t dim() const noexcept { return shape_.size(); }

    size_t numel() const noexcept { return shape_product(shape_); }

    bool allocated() const noexcept { return storage_.size() > 0; }

    T* data() noexcept { return storage_.ptr() ? storage_.ptr() : nullptr; }
    const T* data() const noexcept { return storage_.ptr() ? storage_.ptr() : nullptr; }

    void resize(const std::vector<size_t>& new_shape) {
        shape_ = new_shape;
        strides_ = compute_strides(shape_);
        size_t n = numel();
        storage_.allocate(n);
    }

    T* offset_ptr(size_t new_offset = 0) {
        throw_if(new_offset >= storage_.size(), "offset_ptr: out-of-bounds");
        return storage_.ptr() + new_offset;
    }

    const T* offset_ptr(size_t new_offset = 0) const {
        throw_if(new_offset >= storage_.size(), "offset_ptr: out-of-bounds");
        return storage_.ptr() + new_offset;
    }

    void fill(T value) {
        size_t n = numel();
        if (n == 0) return;
        if (value == T(0)) {
            nvtx3::scoped_range memsetRange{ "cudaMemsetAsync: zero from fill method" };
            CUDA_CHECK(cudaMemsetAsync(data(), static_cast<T>(0), n * sizeof(T), stream_));
        }
        else {
            dim3 blocks((n + 255) / 256);
            dim3 threads(256);
            ARCH::ExecuteKernel("fill specific values", blocks, threads, 0, stream_, KERNEL::FillTensor<T>, data(), value, n);
        }
    }

    void fillRandom(T llimit, T rlimit, unsigned long long seed) {
        size_t n = numel();
        if (n == 0) return;
        dim3 blocks((n + 255) / 256);
        dim3 threads(256);
        ARCH::ExecuteKernel("fill random values", blocks, threads, 0, stream_, KERNEL::FillTensorRandom<T>, data(), n, llimit, rlimit, seed);
    }

    void fillRandomOptimized(unsigned long long seed) {
		CUDA::FillTensorRandom(data(), numel(), stream_, seed);
    }

    void* raw_ptr() noexcept { return static_cast<void*>(data()); }

    const void* raw_ptr() const noexcept { return static_cast<const void*>(data()); }

    void debug_print(const char* name = "") const {
        std::cerr << "[Tensor " << name << "] shape=(";
        for (size_t i = 0; i < shape_.size(); ++i) {
            if (i) std::cerr << ",";
            std::cerr << shape_[i];
        }
        std::cerr << ") " << " numel=" << numel() << " bytes=" << (numel() * sizeof(T)) << "\n";
    }
};

_AM_END


#endif