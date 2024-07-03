/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 2: Addition de vecteurs
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"
#include "chronoGPU.hpp"

namespace IMAC
{
	__global__ void sumArraysCUDA(const int n, const int *const dev_a, const int *const dev_b, int *const dev_res)
	{
		auto division = n/blockDim.x+1;
		for(int i = 0; i < division; i++)
		{
			dev_res[i+threadIdx.x*division] = dev_a[i+threadIdx.x*division]+dev_b[i+threadIdx.x*division];
		}

	}

    void studentJob(const int size, const int *const a, const int *const b, int *const res)
	{
		ChronoGPU chrGPU;

		// 3 arrays for GPU
		int *dev_a = NULL;
		int *dev_b = NULL;
		int *dev_res = NULL;
		
		// Allocate arrays on device (input and ouput)
		const size_t bytes = size * sizeof(int);

		std::cout 	<< "Allocating input (3 arrays): " 
					<< ( ( 3 * bytes ) >> 20 ) << " MB on Device" << std::endl;
		chrGPU.start();
		
		cudaMalloc(&dev_a, bytes);
		cudaMalloc(&dev_b, bytes);
		cudaMalloc(&dev_res, bytes);

		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from host to device (input arrays) 
    	cudaMemcpy(dev_a,a,bytes,cudaMemcpyHostToDevice);
    	cudaMemcpy(dev_b,b,bytes,cudaMemcpyHostToDevice);
		// Launch kernel
		sumArraysCUDA<<<1, 1024>>>(size, dev_a, dev_b, dev_res);

		// Copy data from device to host (output array)  
		cudaMemcpy(res,dev_res,bytes,cudaMemcpyDeviceToHost);

		// Free arrays on device
		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFree(dev_res);
	}
}

