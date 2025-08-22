# TorchLessCUDA v1.0

*A minimal deep‑learning modules library in C++/CUDA — no PyTorch, no TensorFlow.*

> **Status:** v1.0 (experimental). v2.0 under active development (templated, faster, safer).
> **OS/Toolchain:** CUDA 11/12+, C++17, NVCC; tested on Windows (MSVC) and Linux (g++).
> **Dataset example:** MNIST (ubyte files).

---

## Table of Contents

* [What is this?](#what-is-this)
* [Directory Layout](#directory-layout)
* [Core ideas](#core-ideas)
* [Supported layers & ops](#supported-layers--ops)
* [Build & Run](#build--run)
* [Quickstart (MNIST)](#quickstart-mnist)
* [API snapshot](#api-snapshot)
* [Memory model](#memory-model)
* [Known issues (v1.0)](#known-issues-v10)
* [Performance notes & easy wins](#performance-notes--easy-wins)
* [Roadmap → v2.0](#roadmap--v20)
* [Contributing](#contributing)
* [License](#license)

---

## What is this?

**TorchLessCUDA** is a small, self‑contained C++/CUDA library for building and training CNNs without any major DL framework. It’s built around a simple **BatchDevice** struct that tracks device pointers and tensor strides, plus lightweight **Layer** classes that:

1. run a `forward()` CUDA kernel, and
2. capture a lambda that knows how to run the matching `backward()` kernels (and update parameters).

It’s intended for learning, experimentation, and as a baseline for a more robust v2.0.

---

## Directory Layout

```
TOrchLessCUDA-1.0/
├─ Model.cpp                     # Example training script (MNIST)
├─ README.md                     # (this file will replace it)
├─ Includes/
│  ├─ Batch.h                    # Host-side BatchWrapper & helpers
│  └─ Cnn.h                      # Layer declarations + Network
├─ Modules/
│  ├─ Batch.cpp                  # Host-side allocation/utilities
│  └─ Cnn.cpp                    # Layer impl + training loop
├─ Cuda/
│  ├─ BatchDevice.cuh            # __device__ tensor view
│  ├─ Kernels.cu / .cuh          # All CUDA kernels (conv, pool, BN, ...)
│  └─ KernelWrappers.cu / .cuh   # Launchers + error checks
└─ Images/ImgIncoder.h           # MNIST ubyte loaders & helpers
```

---

## Core ideas

* **Contiguous 4D tensors** `[B, C, H, W]` with precomputed **strides**; all math stays on device.
* A **BatchWrapper** owns a device‐side `BatchDevice` (DATA, GRAD, STRIDES) and sizes.
  Resets (`resetData/resetGrad`) zero the buffers on GPU.
* Every layer’s `forward()` sets `BWout->backward = [=](lr, r2, clip){ ... }` which:

  * launches the backward kernel(s),
  * recursively calls the previous tensor’s `backward`,
  * updates parameters (via simple SGD wrappers), then
  * clears scratch buffers.
* The `Network` orchestrates minibatching, forward pass construction, loss (`OutputLayer` = softmax+CCE), backward pass, simple metrics.

---

## Supported layers & ops

* **Conv2D** (NCHW, stride/pad) — random init, SGD update
* **Activations**: ReLU, Sigmoid, Tanh
* **Pooling**: Max / Avg / Min (fwd+bwd)
* **BatchNorm** (per‑channel, training path)
* **Dropout** (training path)
* **Dense (Fully Connected)** + Bias
* **Output**: Softmax + Categorical Cross‑Entropy (with gradient copy)
* **Evaluation**: confusion‑matrix like tallies for TP/TN/FP/FN

> Note: v1.0 BN/Dropout do not yet have clean inference modes with frozen stats/masks.

---

## Build & Run

### Prereqs

* CUDA Toolkit 11 or 12 (NVCC in PATH)
* A C++17 compiler (MSVC on Windows; g++/clang on Linux)
* A CUDA‑capable GPU with sufficient memory

### Windows (MSVC + NVCC)

* Create a **CUDA Runtime** project in Visual Studio, add the sources from `TOrchLessCUDA-1.0`.
* Ensure **C/C++ → Language → C++ Language Standard** is at least `/std:c++17`.
* Add include paths to `Includes/` and `Cuda/`.
* Compile `.cu` files with NVCC; `.cpp/.h` with MSVC.

### Linux (g++ + NVCC)

Minimal example build command (no CMake yet):

```bash
nvcc -std=c++17 \
  Cuda/Kernels.cu Cuda/KernelWrappers.cu Modules/Batch.cpp Modules/Cnn.cpp Model.cpp \
  -IIncludes -ICuda -IImages \
  -o torchless_v1
```

Then run:

```bash
./torchless_v1
```

### Dataset

For the MNIST example in `Model.cpp`, place the ubyte files under `Dataset/`:

```
Dataset/
├─ train-images-idx3-ubyte
└─ train-labels-idx1-ubyte
```

---

## Quickstart (MNIST)

`Model.cpp` builds a small CNN and trains for a few epochs:

```cpp
Network Model;
Model.Input(60000, /*B*/ 1, /*C*/ 28, /*H*/ 28, /*W*/ 28, /*miniBatch=*/600);

Model.Conv2D(32, 1, 3, 3, 1, 1);  // (B,32,28,28)
Model.Activation("relu");
Model.Conv2D(64, 32, 3, 3, 1, 1); // (B,64,28,28)
Model.Activation("relu");
Model.Pooling(2, 2, 2, 0, "max"); // (B,64,14,14)
Model.Conv2D(128, 64, 3, 3, 1, 1);
Model.Activation("relu");
Model.Pooling(2, 2, 2, 0, "max");  // (B,128,7,7)
Model.Dropout(0.5f);
Model.Dense(10);                     // logits

int epochs = 4; float lr = 1e-2f; float l2 = 1e-4f; float clip = 1.0f;
Model.Train(inputs, labels, epochs, lr, l2, clip);
Model.Test(inputs, labels);
Model.showEvaluation();
```

---

## API snapshot

> *This is a lightweight view; see `Includes/Cnn.h` for full signatures.*

```cpp
// Topology
void Input(int totalSamples, int C, int H, int W, int miniBatch=600);
void Conv2D(int Cout, int Cin, int kH, int kW, int stride, int pad);
void Activation(const std::string& type); // "relu" | "sigmoid" | "tanh"
void Pooling(int kH, int kW, int stride, int pad, const std::string& type); // "max"|"avg"|"min"
void BatchNorm();
void Dropout(float p);
void Dense(int outFeatures);

// Training / eval
void Train(const Tensor4D& inputs, const Tensor4D& labels, int epochs, float lr, float l2, float gradClip);
void Test(const Tensor4D& inputs, const Tensor4D& labels);
std::vector<int> Predict(const Tensor4D& inputs);
void showEvaluation();
```

**Tensor shape:** Internally represented as `std::vector<std::vector<std::vector<std::vector<float>>>>` (host) → uploaded to device via `batchWrapper(...)`.

---

## Memory model

* `BatchWrapper` allocates device buffers for **DATA**, **GRAD**, and **STRIDES**, plus a device copy of the `BatchDevice` header.
* Layer outputs allocate new `BatchWrapper`s; intermediates are freed when their `shared_ptr` is reset.
* Zeroing (`resetData`/`resetGrad`) uses `cudaMemset` on the device arrays.

**Tip:** For large networks, consider reducing the default minibatch (e.g. 64/128) to limit VRAM usage in v1.0.

---

## Known issues (v1.0)

> These are intentionally documented to help contributors and to guide v2.0.

1. **Dropout backward uses the wrong buffer (bug):** In `DropoutLayer::forward`, the captured `rawBWout` accidentally points to the input `BWi` instead of the layer output. This causes incorrect gradients and unnecessary resets on the input buffer. *Fix:* capture `auto rawBWout = BWout.get();`.
2. **Excessive device synchronizations:** Nearly every wrapper calls `cudaDeviceSynchronize()`, serializing the pipeline and hiding concurrency. *Fix:* remove most synchronizations; rely on stream ordering; only sync at epoch/batch boundaries or when copying to host.
3. **Frequent host↔device header copies:** `reset*()` pulls the `BatchDevice` header back to host just to memset device arrays. *Fix:* keep a cached host mirror of the header or pass the device pointer directly where possible.
4. **Convolution/Dense kernels are naïve:** No tiling, shared memory, vectorization, or tensor‑core paths; weight‑gradients rely on many `atomicAdd`s which will throttle throughput. *Fix:* im2col + GEMM (cuBLAS) or block‑tiling with shared reductions.
5. **BatchNorm lacks running stats for inference:** Only training path is implemented; no moving mean/variance or epsilon parametrization. *Fix:* maintain running stats; add `train()/eval()` modes.
6. **Dropout state allocation per forward:** `curandState*` is `cudaMalloc`’d each call. Although freed, this adds overhead. *Fix:* allocate once and reuse or generate masks via Philox in kernel without persistent state.
7. **Gradient clipping & L2 mixing:** L2 regularization is added to gradient in some kernels and then clipped; the order/targets may not match common practice (clip on parameter/update or on total grad consistently). *Fix:* standardize to `grad = grad + λ*w; grad = clip(grad);` before the update.
8. **No parameter initialization strategy:** Random uniform `[-1,1]` can be unstable. *Fix:* Kaiming/He for ReLU, Glorot/Xavier otherwise.
9. **No bias in Conv2D:** Convolution currently lacks a bias term. *Fix:* add bias buffer + update.
10. **No mixed precision or template types:** Everything is `float`. *Fix (v2.0):* template on `T` (`float`, `double`, `__half`) and add loss‑scaler for FP16.
11. **Evaluation is simplistic:** TP/TN/FP/FR tallies from argmax; no accuracy/precision/recall/F1 reporting utilities. *Fix:* compute metrics and confusion matrix helpers.
12. **Error handling:** Many file and CUDA errors abort via `exit()`; better to return statuses or throw exceptions with context.

---

## Performance notes & easy wins

**Short‑term tweaks (keep v1.0 structure):**

* Drop most `cudaDeviceSynchronize()` calls; check only after critical boundaries.
* Pre‑allocate per‑layer scratch (BN means/vars, Dropout masks/state) and reuse across batches.
* Use **streams**: one default stream per network; optionally separate copy/compute.
* Use better initializers (He/Glorot) to help convergence and avoid large grads → fewer steps.
* Fuse simple ops (Activation after Conv) when convenient to reduce memory traffic.

**Medium‑term (still minimal):**

* Convert Conv/Dense to **im2col + cublasGemmEx** (fast win; enables tensor cores).
* Block‑tiling for custom kernels (shared memory, coalesced loads, `__ldg`, warp‑level reductions).
* Replace heavy `atomicAdd` patterns with CTA‑local reductions followed by a single atomic.
* Add a light **workspace allocator** for temporary buffers.

---

## Roadmap → v2.0

> v2.0 focuses on **templates**, **memory safety**, and **speed**.

* **Templated tensors & layers:** `template<typename T=float>` across `BatchDevice`, kernels, and layers. Optional `__half` + loss scaling.
* **Allocator & RAII:** custom device allocator (arena/pool) + smart handles; no raw `cudaMalloc/free` in layer code; zero host round‑trips for headers.
* **Streams & events:** stage kernels per batch without global sync; optional multi‑stream overlap.
* **Better math backends:** plug‑in BLAS (cuBLAS) for GEMM‑based layers; optional CUTLASS for conv.
* **Initialization policies:** He/Glorot/Orthogonal.
* **Train/Eval modes:** running stats for BN, no‑op Dropout in eval.
* **Optimizers:** SGD, Momentum, Adam; weight decay decoupled from grad (AdamW style).
* **Metrics:** accuracy/precision/recall/F1; confusion matrix util.
* **Testing:** host‑side reference ops (small tensors) + `EXPECT_NEAR` checks; determinism toggles (fixed seeds).
* **CMake project:** cross‑platform build.
* **Docs & examples:** cleaner examples (MNIST, CIFAR‑10), profiling notes (Nsight Systems/Compute).

---

## Contributing

PRs and issues welcome! Ideas to start with:

* Fix the Dropout `rawBWout` capture bug.
* Remove redundant synchronizations; introduce a single network stream.
* Add Conv2D bias + He init.
* Implement running stats for BatchNorm and an `eval()` switch.
* Provide a tiny CMake build.
* Add a CPU reference test for a 1‑layer network to validate kernels.

Please include: reproduction steps, GPU/driver/toolkit versions, and Nsight screenshots if performance‑related.

---

## License

MIT (proposed). Update as appropriate for your project.
