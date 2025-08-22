#ifndef __MACROS_HPP__
#define __MACROS_HPP__

#ifdef __CUDACC__
#define IF_CUDA_AVAILABLE(code) code
#else
#define IF_CUDA_AVAILABLE(code)
#endif

#define _AM_START namespace ARCH {
#define _AM_END }

#define _HIDDEN_START namespace {
#define _HIDDEN_END }

#define _LAYERS_START namespace LAYER {
#define _LAYERS_END }

#define _KERNELS_START namespace KERNEL {
#define _KERNELS_END }

#define _CUDNN_START namespace CUDA {
#define _CUDNN_END }

#endif// Macros.hpp