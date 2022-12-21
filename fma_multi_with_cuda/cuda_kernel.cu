#include <cuda_runtime_api.h>
#include "cuda_kernel.cuh"


namespace
{

template <typename T>
__global__ void fma_kernel(T* out, const T* A, const T* B, const T* C, int offset)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x + offset;

	out[index] = A[index] * B[index] + C[index];
}

template <typename T>
__global__ void fma_kernel(T* out, const T* A, const T* B, int offset)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x + offset;

	out[index] = A[index] * B[index];
}


// Maximum number of threads per block
constexpr int MAX_THREADS = 1024;

}		// namespace


namespace custom_kernel
{

template <typename T>
void fma_core(T* out, const T* A, const T* B, const T* C, size_t element_count, cudaStream_t stream)
{
	int offset = 0;
	int num_blocks = element_count / MAX_THREADS;
	int num_threads = MAX_THREADS;
	fma_kernel<T><<<num_blocks, num_threads, 0, stream>>>(out, A, B, C, offset);

	if (element_count % MAX_THREADS) {
		offset = num_blocks * MAX_THREADS;
		num_blocks = 1;
		num_threads = element_count % MAX_THREADS;
		fma_kernel<T><<<num_blocks, num_threads, 0, stream>>>(out, A, B, C, offset);
	}
}

template <typename T>
void fma_core(T* out, const T* A, const T* B, size_t element_count, cudaStream_t stream)
{
	int offset = 0;
	int num_blocks = element_count / MAX_THREADS;
	int num_threads = MAX_THREADS;
	fma_kernel<T><<<num_blocks, num_threads, 0, stream>>>(out, A, B, offset);

	if (element_count % MAX_THREADS) {
		offset = num_blocks * MAX_THREADS;
		num_blocks = 1;
		num_threads = element_count % MAX_THREADS;
		fma_kernel<T><<<num_blocks, num_threads, 0, stream>>>(out, A, B, offset);
	}
}


// explicit instantiation
template void fma_core<float>(float*, const float*, const float*, const float*, size_t, cudaStream_t);
template void fma_core<double>(double*, const double*, const double*, const double*, size_t, cudaStream_t);

template void fma_core<float>(float*, const float*, const float*, size_t, cudaStream_t);
template void fma_core<double>(double*, const double*, const double*, size_t, cudaStream_t);

}	// custom_kernel

