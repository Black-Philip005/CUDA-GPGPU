/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 3: Filtre d'images sepia
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"
#include "chronoGPU.hpp"

namespace IMAC
{

	__global__ void sepiaCUDA(uchar * input, uchar *output, const uint w, const uint h)
	{
		int idx = threadIdx.x+blockIdx.x*blockDim.x;
		int idy = threadIdx.y+blockIdx.y*blockDim.y;
		int id = (idx*h+(idy));
		if(id > w*h)return;
		uchar inRed = input[id*3];
		uchar inGreen = input[id*3+1];
		uchar inBlue = input[id*3+2];
		output[id*3] = fminf(255, (inRed *0.393f + inGreen *0.769f + inBlue *0.189f));
		output[id*3+1] = fminf(255, (inRed* 0.349f + inGreen *0.686f + inBlue *0.168f));
		output[id*3+2] = fminf(255, (inRed* 0.272f + inGreen *0.534f + inBlue *0.131f));
		
	}

	void studentJob(const std::vector<uchar> &input, const uint width, const uint height, std::vector<uchar> &output)
	{
		ChronoGPU chrGPU;
		size_t bytes = sizeof(uchar)*(input.size());
		// 2 arrays for GPU
		uchar *dev_input = NULL;
		uchar *dev_output = NULL;

		cudaMalloc(&dev_input, bytes);
		cudaMalloc(&dev_output, bytes);
		/// TODOOOOOOOOOOOOOO
		cudaMemcpy(dev_input,input.data(), bytes,cudaMemcpyHostToDevice);
		// Launch kernel
		sepiaCUDA<<<dim3(1+(width/32), 1+(height/32) , 1), dim3(32, 32 , 1)>>>(dev_input, dev_output, width, height);

		// Copy data from device to host (output array)  
	
		cudaMemcpy(output.data(),dev_output,bytes,cudaMemcpyDeviceToHost);
		
		cudaFree(dev_input);
		cudaFree(dev_output);
	}

}
