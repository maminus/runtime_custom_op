#pragma once

#include <cuda_runtime.h>

namespace custom_kernel
{

template <typename T>
void fma_core(T* out, const T* A, const T* B, const T* C, size_t element_count, cudaStream_t stream);

template <typename T>
void fma_core(T* out, const T* A, const T* B, size_t element_count, cudaStream_t stream);

}	// namespace custom_kernel

