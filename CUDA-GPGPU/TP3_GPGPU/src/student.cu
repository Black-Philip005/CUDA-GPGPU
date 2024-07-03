/*
* TP 3 - Réduction CUDA
* --------------------------
* Mémoire paratagée, synchronisation, optimisation
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"

namespace IMAC
{
	// ==================================================== EX 1
    __global__
    void maxReduce_ex1(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
		extern __shared__ uint sharedMemory[];

		unsigned int tid = threadIdx.x;
		unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
		if(i > size)return;
		sharedMemory[tid] = dev_array[i];
		__syncthreads();
		// do reduction in shared mem
		for(unsigned int s=1; s < blockDim.x; s *= 2) {
			int index = 2 * s * tid;
			if (index < blockDim.x) {
			sharedMemory[index] = umax( sharedMemory[index + s], sharedMemory[index]);
			} else {
				break;
			}
		__syncthreads();
		}
		// write result for this block to global mem
		if (tid == 0) dev_partialMax[blockIdx.x] = sharedMemory[0];
	}
	
	__global__
	void maxReduce_ex2(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
		extern __shared__ uint sharedMemory[];

		unsigned int tid = threadIdx.x;
		unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
		if(i > size)return;
		sharedMemory[tid] = dev_array[i];
		__syncthreads();
		// do reduction in shared mem
		for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
			if (tid < s) {
				sharedMemory[tid] =  umax(sharedMemory[tid], sharedMemory[tid + s]);
			}
			__syncthreads();
		}
		// write result for this block to global mem
		if (tid == 0) dev_partialMax[blockIdx.x] = sharedMemory[0];
	}

	__global__
	void maxReduce_ex3(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
		extern __shared__ uint sharedMemory[];
		unsigned int tid = threadIdx.x;
		unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
		if(i > size)return;
		sharedMemory[tid] = umax(dev_array[i], dev_array[i+blockDim.x]);		__syncthreads();
		// do reduction in shared mem
		for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
			if (tid < s) {
			sharedMemory[tid] =  umax(sharedMemory[tid], sharedMemory[tid + s]);
			}
			__syncthreads();
		}
		// write result for this block to global mem
		if (tid == 0) dev_partialMax[blockIdx.x] = sharedMemory[0];
	}

	__device__ void warpReduce(volatile uint* sdata, int tid) {
		sdata[tid] = umax(sdata[tid], sdata[tid + 32]);
		sdata[tid] = umax(sdata[tid], sdata[tid + 16]);
		sdata[tid] = umax(sdata[tid], sdata[tid + 8]);
		sdata[tid] = umax(sdata[tid], sdata[tid + 4]);
		sdata[tid] = umax(sdata[tid], sdata[tid + 2]);
		sdata[tid] = umax(sdata[tid], sdata[tid + 1]);
		}

	__global__
	void maxReduce_ex4(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
		extern __shared__ uint sharedMemory[];
		unsigned int tid = threadIdx.x;
		unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
		if(i > size)return;
		sharedMemory[tid] = umax(dev_array[i], dev_array[i+blockDim.x]);		__syncthreads();
		// do reduction in shared mem
		for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
			if (tid < s) {
			sharedMemory[tid] =  umax(sharedMemory[tid], sharedMemory[tid + s]);
			}
			__syncthreads();
		}
		if (tid < 32) warpReduce(sharedMemory, tid);
		// write result for this block to global mem
		if (tid == 0) dev_partialMax[blockIdx.x] = sharedMemory[0];
	}
	
	template <uint blockSize>
	__device__ void warpReduceT(volatile uint* sdata, uint tid) {
	if (blockSize >= 64) sdata[tid] = umax( sdata[tid], sdata[tid + 32]);
	if (blockSize >= 32) sdata[tid] = umax( sdata[tid], sdata[tid + 16]);
	if (blockSize >= 16) sdata[tid] = umax( sdata[tid], sdata[tid + 8]);
	if (blockSize >= 8) sdata[tid] = umax( sdata[tid], sdata[tid + 4]);
	if (blockSize >= 4) sdata[tid] = umax( sdata[tid], sdata[tid + 2]);
	if (blockSize >= 2) sdata[tid] = umax( sdata[tid] , sdata[tid + 1]);
	}

	
	template <uint blockSize>
	__global__
	void maxReduce_ex5(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
		extern __shared__ uint sharedMemory[];
		unsigned int tid = threadIdx.x;
		unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
		if(i > size)return;
		sharedMemory[tid] = umax(dev_array[i], dev_array[i+blockDim.x]);		
		__syncthreads();
		// do reduction in shared mem
		if (blockSize >= 1024) {
		if (tid < 512) {sharedMemory[tid] =  umax(sharedMemory[tid] ,sharedMemory[tid + 512]); } __syncthreads(); }
		if (blockSize >= 512) {
		if (tid < 256) { sharedMemory[tid]= umax(sharedMemory[tid] ,sharedMemory[tid + 256]); } __syncthreads(); }
		if (blockSize >= 256) {
		if (tid < 128) {sharedMemory[tid] =  umax(sharedMemory[tid] ,sharedMemory[tid + 128]); } __syncthreads(); }
		if (blockSize >= 128) {
		if (tid < 64) { sharedMemory[tid] = umax(sharedMemory[tid] ,sharedMemory[tid + 64]); } __syncthreads(); }
		if (tid < 32) warpReduceT<blockSize>(sharedMemory, tid);
		// write result for this block to global mem
		if (tid == 0) dev_partialMax[blockIdx.x] = sharedMemory[0];
		
	}
	
	void studentJob(const std::vector<uint> &array, const uint resCPU /* Just for comparison */, const uint nbIterations)
    {
		uint *dev_array = NULL;
        const size_t bytes = array.size() * sizeof(uint);

		// Allocate array on GPU
		HANDLE_ERROR( cudaMalloc( (void**)&dev_array, bytes ) );
		// Copy data from host to device
		HANDLE_ERROR( cudaMemcpy( dev_array, array.data(), bytes, cudaMemcpyHostToDevice ) );

		std::cout << "Test with " << nbIterations << " iterations" << std::endl;

		std::cout << "========== Ex 1 " << std::endl;
		uint res1 = 0; // result
		// Launch reduction and get timing
		float2 timing1 = reduce<KERNEL_EX1>(nbIterations, dev_array, array.size(), res1);
		
        std::cout << " -> Done: ";
        printTiming(timing1);
		compare(res1, resCPU); // Compare results

		std::cout << "========== Ex 2 " << std::endl;
		uint res2 = 0; // result
		// Launch reduction and get timing
		float2 timing2 = reduce<KERNEL_EX2>(nbIterations, dev_array, array.size(), res2);
		
        std::cout << " -> Done: ";
        printTiming(timing2);
		compare(res2, resCPU);

		std::cout << "========== Ex 3 " << std::endl;
		uint res3 = 0; // result
		// Launch reduction and get timing
		float2 timing3 = reduce<KERNEL_EX3>(nbIterations, dev_array, array.size(), res3);
		
        std::cout << " -> Done: ";
        printTiming(timing3);
		compare(res3, resCPU);

		std::cout << "========== Ex 4 " << std::endl;
		uint res4 = 0; // result
		// Launch reduction and get timing
		float2 timing4 = reduce<KERNEL_EX4>(nbIterations, dev_array, array.size(), res4);
		
        std::cout << " -> Done: ";
        printTiming(timing4);
		compare(res4, resCPU);

		std::cout << "========== Ex 5 " << std::endl;
		uint res5 = 0; // result
		// Launch reduction and get timing
		float2 timing5 = reduce<KERNEL_EX5>(nbIterations, dev_array, array.size(), res5);
		
        std::cout << " -> Done: ";
        printTiming(timing5);
		compare(res5, resCPU);

		// Free array on GPU
		cudaFree( dev_array );
    }

	void printTiming(const float2 timing)
	{
		std::cout << ( timing.x < 1.f ? 1e3f * timing.x : timing.x ) << " us on device and ";
		std::cout << ( timing.y < 1.f ? 1e3f * timing.y : timing.y ) << " us on host." << std::endl;
	}

    void compare(const uint resGPU, const uint resCPU)
	{
		if (resGPU == resCPU)
		{
			std::cout << "Well done ! " << resGPU << " == " << resCPU << " !!!" << std::endl;
		}
		else
		{
			std::cout << "You failed ! " << resGPU << " != " << resCPU << " !!!" << std::endl;
		}
	}
}
