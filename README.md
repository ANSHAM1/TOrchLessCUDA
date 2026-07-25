# TorchLessCUDA

TorchLessCUDA is a deep learning framework built entirely from scratch using Modern C++20 and CUDA, implementing tensor operations, GPU training, and inference without relying on existing machine learning frameworks.

---

#  Overview

Modern deep learning frameworks abstract away GPU programming, memory management, tensor operations, automatic differentiation, and optimization.

TorchLessCUDA focuses on implementing these components manually while maintaining a modular and extensible architecture.

The project currently supports training and inference of neural networks entirely on the GPU using custom CUDA kernels.

---

#  Features

## Tensor Engine

- GPU-resident Tensor abstraction
- Multi-dimensional tensor support
- Dynamic tensor reshaping
- Tensor views (zero-copy)
- Automatic stride computation
- CPU ↔ GPU memory transfer
- Tensor debugging utilities
- Runtime tensor statistics

---

## CUDA Backend

- Custom CUDA kernels
- CUDA runtime API
- GPU memory management
- Ping-pong workspace buffers
- Efficient memory reuse
- Host/device synchronization
- Modern CUDA C++

---

## Neural Network Layers

Currently implemented layers:

- Dense (Fully Connected)
- Conv2D
- Max Pooling
- Flatten
- Activation Layer

Each layer supports:

- Forward propagation
- Backpropagation
- Shape inference
- Parameter management
- Gradient computation

---

## Activation Functions

Implemented activation functions:

- ReLU
- Sigmoid
- Tanh
- Softmax

---

## Loss Functions

Currently implemented:

- Cross Entropy Loss (CCE)

Supports:

- Forward loss computation
- Gradient computation

---

## Optimizers

Implemented optimizers:

- SGD
- Momentum SGD
- AdaGrad
- RMSProp
- Adam

Each optimizer maintains its own internal optimization state.

---

## Training Engine

- Automatic forward propagation
- Automatic backpropagation
- Gradient propagation
- Parameter updates
- Mini-batch training
- Execution context compilation
- Workspace reuse
- Pre-allocated activation buffers
- Pre-allocated gradient buffers

---

## Inference Engine

- Optimized inference pipeline
- Separate inference execution context
- Ping-pong workspace memory
- Zero gradient allocation during inference

---

## Memory Management

TorchLessCUDA minimizes runtime allocations by compiling execution contexts before training or inference.

Features include:

- Workspace reuse
- Activation caching
- Gradient caching
- Reduced memory fragmentation
- Stable memory consumption during training

---

## Dataset Support

Currently supported:

- MNIST Image Dataset
- IDX image loader
- IDX label loader
- Mini-batch data loading

---

## Development Infrastructure

- Modern C++20
- CUDA 12.x
- CMake build system
- Cross-platform project structure
- GitHub Actions CI
- Visual Studio support
- Modular architecture

---

# ⚙️ Requirements

## Hardware

- NVIDIA GPU with CUDA support

## Software

- C++20 compatible compiler
- NVIDIA CUDA Toolkit 12.x
- CMake 3.25+
- Visual Studio 2022 (Windows)

---

# Current Status

Implemented:

- Tensor abstraction
- CUDA backend
- Neural network layers
- Automatic forward propagation
- Automatic backpropagation
- GPU training
- GPU inference
- Optimizers
- MNIST training pipeline

---

# Current Limitations

- Only **Cross Entropy Loss (CCE)** is currently implemented.
- Only the **MNIST dataset** has been tested.
- GPU execution only (no CPU backend).
- No model serialization (save/load) yet.
- No Batch Normalization.
- No Dropout layer.
- No learning rate schedulers.
- No mixed precision (FP16/BF16).
- Single GPU only.
- No distributed training.

---

# Future Work

Planned improvements include:

- Additional loss functions
- Batch Normalization
- Dropout
- Model serialization
- Learning rate schedulers
- Mixed precision training
- Additional datasets
- Additional layer implementations
- Performance profiling and optimization

---

# Motivation

TorchLessCUDA was built as a learning project to understand how modern deep learning frameworks operate internally. Rather than relying on existing machine learning libraries, every major component—including tensor management, CUDA execution, forward propagation, backpropagation, and optimization—has been implemented from scratch using Modern C++ and CUDA.

The project emphasizes clean architecture, modular design, and low-level GPU programming while serving as a foundation for experimenting with deep learning systems.

---

# License

This project is released under the MIT License.
