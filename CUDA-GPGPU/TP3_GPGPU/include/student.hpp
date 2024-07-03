/*
* TP 3 - Réduction CUDA
* --------------------------
* Mémoire paratagée, synchronisation, optimisation
*
* File: student.hpp
* Author: Maxime MARIA
*/

#ifndef __STUDENT_HPP
#define __STUDENT_HPP

#include <vector>

#include "common.hpp"
#include "chronoGPU.hpp"
#include "chronoCPU.hpp"

namespace IMAC
{
	const uint MAX_NB_THREADS = 1024; // En dur, changer si GPU plus ancien ;-)
    const uint DEFAULT_NB_BLOCKS = 32768;

    enum
    {
        KERNEL_EX1 = 0,
        KERNEL_EX2,
        KERNEL_EX3,
        KERNEL_EX4,
        KERNEL_EX5
    };
	
	// ==================================================== EX 1
	__global__
    void maxReduce_ex1(const uint *const dev_array, const uint size, uint *const dev_partialMax);
	__global__
    void maxReduce_ex2(const uint *const dev_array, const uint size, uint *const dev_partialMax);
	__global__
    void maxReduce_ex3(const uint *const dev_array, const uint size, uint *const dev_partialMax);

	__device__ void warpReduce(volatile uint* sdata, int tid) ;
	__global__
    void maxReduce_ex4(const uint *const dev_array, const uint size, uint *const dev_partialMax);

	
	template <uint blockSize>
	__device__ void warpReduceT(volatile uint* sdata, uint tid);

	template <uint blockSize>
	__global__
    void maxReduce_ex5(const uint *const dev_array, const uint size, uint *const dev_partialMax);
	
	// return a uint2 with x: dimBlock / y: dimGrid
    template<uint kernelType>
    uint2 configureKernel(const uint sizeArray)
    {
        uint2 dimBlockGrid; // x: dimBlock / y: dimGrid
		uint newSize = sizeArray/2;
		// Configure number of threads/blocks
		switch(kernelType)
		{
			case KERNEL_EX1:
				dimBlockGrid.x = sizeArray < MAX_NB_THREADS ? min(MAX_NB_THREADS, nextPow2(sizeArray)) : MAX_NB_THREADS; 
                dimBlockGrid.y = 1 + (sizeArray / dimBlockGrid.x);
			break;
			case KERNEL_EX2:
				dimBlockGrid.x = sizeArray < MAX_NB_THREADS ? min(MAX_NB_THREADS, nextPow2(sizeArray)) : MAX_NB_THREADS; 
                dimBlockGrid.y = 1 + (sizeArray / dimBlockGrid.x);
			break;
			case KERNEL_EX3:
                dimBlockGrid.x = newSize < MAX_NB_THREADS ? min(MAX_NB_THREADS, nextPow2(newSize)) : MAX_NB_THREADS; 
                dimBlockGrid.y = 1 + (newSize / dimBlockGrid.x);
                break;
			case KERNEL_EX4:
				dimBlockGrid.x = newSize < MAX_NB_THREADS ? min(MAX_NB_THREADS, nextPow2(newSize)) : MAX_NB_THREADS; 
                dimBlockGrid.y = 1 + (newSize / dimBlockGrid.x);
			break;
			case KERNEL_EX5:
				dimBlockGrid.x = newSize < MAX_NB_THREADS ? min(MAX_NB_THREADS, nextPow2(newSize)) : MAX_NB_THREADS; 
                dimBlockGrid.y = 1 + (newSize / dimBlockGrid.x);
			break;
            default:
                throw std::runtime_error("Error configureKernel: unknown kernel type");
		}
		verifyDimGridBlock( dimBlockGrid.y, dimBlockGrid.x, sizeArray ); // Are you reasonable ?
        
        return dimBlockGrid;
    }

    // Launch kernel number 'kernelType' and return float2 for timing (x:device,y:host)    
    template<uint kernelType>
    float2 reduce(const uint nbIterations, const uint *const dev_array, const uint size, uint &result)
	{
        const uint2 dimBlockGrid = configureKernel<kernelType>(size); // x: dimBlock / y: dimGrid

		// Allocate arrays (host and device) for partial result
		/// TODO
		std::vector<uint> host_partialMax(dimBlockGrid.y); // REPLACE SIZE !
		const size_t bytesPartialMax = host_partialMax.size() * sizeof(uint); // REPLACE BYTES !
		const size_t bytesSharedMem = dimBlockGrid.x * sizeof(uint); // REPLACE BYTES !
		
		uint *dev_partialMax;
		HANDLE_ERROR(cudaMalloc((void**) &dev_partialMax, bytesPartialMax ) );

		std::cout 	<< "Computing on " << dimBlockGrid.y << " block(s) and " 
					<< dimBlockGrid.x << " thread(s) "
					<<"- shared mem size = " << bytesSharedMem << std::endl;

		ChronoGPU chrGPU;
		float2 timing = { 0.f, 0.f }; // x: timing GPU, y: timing CPU
		// Average timing on 'loop' iterations
		for (uint i = 0; i < nbIterations; ++i)
		{
			chrGPU.start();
			switch(kernelType) // Evaluated at compilation time
			{
				case KERNEL_EX1:
					/// TODO EX 2
					maxReduce_ex1<<<dimBlockGrid.y, dimBlockGrid.x, bytesSharedMem>>>(dev_array, size, dev_partialMax);
				break;

				case KERNEL_EX2:
					maxReduce_ex2<<<dimBlockGrid.y, dimBlockGrid.x, bytesSharedMem>>>(dev_array, size, dev_partialMax);
				break;
				case KERNEL_EX3:
					maxReduce_ex3<<<dimBlockGrid.y, dimBlockGrid.x, bytesSharedMem>>>(dev_array, size, dev_partialMax);
				break;
				case KERNEL_EX4:
					maxReduce_ex4<<<dimBlockGrid.y, dimBlockGrid.x, bytesSharedMem>>>(dev_array, size, dev_partialMax);

				break;
				case KERNEL_EX5:
					switch (dimBlockGrid.x)
					{
						case 1024:
							maxReduce_ex5<1024><<<dimBlockGrid.y, dimBlockGrid.x, bytesSharedMem>>>(dev_array, size, dev_partialMax);
							break;
						case 512:
						maxReduce_ex5<512><<< dimBlockGrid.y, dimBlockGrid.x, bytesSharedMem >>>(dev_array, size, dev_partialMax); break;
						case 256:
						maxReduce_ex5<256><<< dimBlockGrid.y, dimBlockGrid.x, bytesSharedMem >>>(dev_array, size, dev_partialMax); break;
						case 128:
						maxReduce_ex5<128><<< dimBlockGrid.y, dimBlockGrid.x, bytesSharedMem >>>(dev_array, size, dev_partialMax); break;
						case 64:
						maxReduce_ex5<64><<< dimBlockGrid.y, dimBlockGrid.x, bytesSharedMem >>>(dev_array, size, dev_partialMax); break;
						case 32:
						maxReduce_ex5<32><<< dimBlockGrid.y, dimBlockGrid.x, bytesSharedMem >>>(dev_array, size, dev_partialMax); break;
						case 16:
						maxReduce_ex5<16><<< dimBlockGrid.y, dimBlockGrid.x, bytesSharedMem >>>(dev_array, size, dev_partialMax); break;
						case 8:
						maxReduce_ex5<8><<< dimBlockGrid.y, dimBlockGrid.x, bytesSharedMem >>>(dev_array, size, dev_partialMax); break;
						case 4:
						maxReduce_ex5<4><<< dimBlockGrid.y, dimBlockGrid.x, bytesSharedMem >>>(dev_array, size, dev_partialMax); break;
						case 2:
						maxReduce_ex5<2><<< dimBlockGrid.y, dimBlockGrid.x, bytesSharedMem >>>(dev_array, size, dev_partialMax); break;
						case 1:
						maxReduce_ex5<1><<< dimBlockGrid.y, dimBlockGrid.x, bytesSharedMem >>>(dev_array, size, dev_partialMax); break;
					}
				break;
                default:
		            cudaFree(dev_partialMax);
                    throw("Error reduce: unknown kernel type.");
			}
			chrGPU.stop();
			timing.x += chrGPU.elapsedTime();
		}
		timing.x /= (float)nbIterations; // Stores time for device

		// Retrieve partial result from device to host
		HANDLE_ERROR(cudaMemcpy(host_partialMax.data(), dev_partialMax, bytesPartialMax, cudaMemcpyDeviceToHost));

		cudaFree(dev_partialMax);

        // Check for error
		cudaDeviceSynchronize();
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			throw std::runtime_error(cudaGetErrorString(err));
		}
		
		ChronoCPU chrCPU;
		chrCPU.start();

		// Finish on host
		for (int i = 0; i < host_partialMax.size(); ++i)
		{
			result = std::max<uint>(result, host_partialMax[i]);
		}
		
		chrCPU.stop();

		timing.y = chrCPU.elapsedTime(); // Stores time for host
		
        return timing;
	}  
    
    void studentJob(const std::vector<uint> &array, const uint resCPU /* Just for comparison */, const uint nbIterations);

    void printTiming(const float2 timing);
    void compare(const uint resGPU, const uint resCPU);
}

#endif
