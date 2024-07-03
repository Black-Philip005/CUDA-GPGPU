/*
* TP 2 - Convolution d'images
* --------------------------
* Mémoire constante et textures
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"
#include "chronoGPU.hpp"
#define clampf(val, min, max) fminf(max, fmaxf(min, val))
#define BLOCK 16
#define THREAD 32
namespace IMAC
{

// ================================================== For image comparison
	std::ostream &operator <<(std::ostream &os, const uchar4 &c)
	{
		os << "[" << uint(c.x) << "," << uint(c.y) << "," << uint(c.z) << "," << uint(c.w) << "]";  
    	return os; 
	}


	void compareImages(const std::vector<uchar4> &a, const std::vector<uchar4> &b)
	{
		bool error = false;
		if (a.size() != b.size())
		{
			std::cout << "Size is different !" << std::endl;
			error = true;
		}
		else
		{
			for (uint i = 0; i < a.size(); ++i)
			{
				// Floating precision can cause small difference between host and device
				if (	std::abs(a[i].x - b[i].x) > 2 || std::abs(a[i].y - b[i].y) > 2 
					|| std::abs(a[i].z - b[i].z) > 2 || std::abs(a[i].w - b[i].w) > 2)
				{
					std::cout << "Error at index " << i << ": a = " << a[i] << " - b = " << b[i] << " - " << std::abs(a[i].x - b[i].x) << std::endl;
					error = true;
					break;
				}
			}
		}
		if (error)
		{
			std::cout << " -> You failed, retry!" << std::endl;
		}
		else
		{
			std::cout << " -> Well done!" << std::endl;
		}
	}
// ==================================================

	__global__ void Cudalution(uchar4 * input, uchar4 *output, const float *convolution, uint convSize, const uint w, const uint h)
	{
		for(uint y = blockIdx.y * blockDim.y + threadIdx.y; y < h; y += gridDim.y * blockDim.y) 
		{
			for(uint x = blockIdx.x * blockDim.x + threadIdx.x; x < w; x += gridDim.x * blockDim.x) 
			{
				float3 sum = make_float3(0.f,0.f,0.f);
				
				// Apply convolution
				for ( uint j = 0; j < convSize; ++j ) 
				{
					for ( uint i = 0; i < convSize; ++i ) 
					{
						int dX = x + i - convSize / 2;
						int dY = y + j - convSize / 2;

						// Handle borders
						if ( dX < 0 ) 
							dX = 0;

						if ( dX >= w ) 
							dX = w - 1;

						if ( dY < 0 ) 
							dY = 0;

						if ( dY >= h ) 
							dY = h - 1;

						const int idMat		= j * convSize + i;
						const int idPixel	= dY * w + dX;
						sum.x += ((float)input[idPixel].x) * convolution[idMat];
						sum.y += (float)input[idPixel].y * convolution[idMat];
						sum.z += (float)input[idPixel].z * convolution[idMat];
					}
				}
				const int idOut = y * w + x;
				output[idOut].x = (uchar)clampf( sum.x, 0.f, 255.f );
				output[idOut].y = (uchar)clampf( sum.y, 0.f, 255.f );
				output[idOut].z = (uchar)clampf( sum.z, 0.f, 255.f );
				output[idOut].w = 255;
			}
		}
		
	}
    void Ex1(const std::vector<uchar4> &inputImg, // Input image
					const uint imgWidth, const uint imgHeight, // Image size
                    const std::vector<float> &matConv, // Convolution matrix (square)
					const uint matSize, // Matrix size (width or height)
					const std::vector<uchar4> &resultCPU, // Just for comparison
                    std::vector<uchar4> &output // Output image
					)
	{
		uint imgSize = imgWidth*imgHeight;
		// 2 arrays for GPU
		uchar4 *device_input = NULL;
		uchar4 *device_output = NULL;
		float * device_convolution = NULL;
		cudaMalloc(&device_input, sizeof(uchar4)*imgSize);
		cudaMalloc(&device_output, sizeof(uchar4)*imgSize);
		cudaMalloc(&device_convolution, sizeof(float)*(matSize*matSize));

		cudaMemcpy(device_input,inputImg.data(), sizeof(uchar4)*imgSize,cudaMemcpyHostToDevice);
		cudaMemcpy(device_convolution,matConv.data(), sizeof(float)*(matSize*matSize),cudaMemcpyHostToDevice);

		const dim3 threads(32, 32); // 32 * 32 = 1024
		const dim3 blocks( ( imgWidth + threads.x - 1 ) / threads.x, ( imgHeight + threads.y - 1 ) / threads.y );

		Cudalution<<<blocks, threads>>>(device_input, device_output, device_convolution, matSize, imgWidth, imgHeight);
		cudaMemcpy(output.data(),device_output,sizeof(uchar4)*imgSize,cudaMemcpyDeviceToHost);

		cudaFree(device_input);
		cudaFree(device_output);
		cudaFree(device_convolution);

	}





	__constant__ float dev_conv_const[2048];

	__global__ void CudalutionConst(uchar4 * input, uchar4 *output, uint convSize, const uint w, const uint h)
	{
		for(uint y = blockIdx.y * blockDim.y + threadIdx.y; y < h; y += gridDim.y * blockDim.y) 
		{
			for(uint x = blockIdx.x * blockDim.x + threadIdx.x; x < w; x += gridDim.x * blockDim.x) 
			{
				float3 sum = make_float3(0.f,0.f,0.f);
				
				// Apply convolution
				for ( uint j = 0; j < convSize; ++j ) 
				{
					for ( uint i = 0; i < convSize; ++i ) 
					{
						int dX = x + i - convSize / 2;
						int dY = y + j - convSize / 2;

						// Handle borders
						if ( dX < 0 ) 
							dX = 0;

						if ( dX >= w ) 
							dX = w - 1;

						if ( dY < 0 ) 
							dY = 0;

						if ( dY >= h ) 
							dY = h - 1;

						const int idMat		= j * convSize + i;
						const int idPixel	= dY * w + dX;
						sum.x += ((float)input[idPixel].x) * dev_conv_const[idMat];
						sum.y += (float)input[idPixel].y * dev_conv_const[idMat];
						sum.z += (float)input[idPixel].z * dev_conv_const[idMat];
					}
				}
				const int idOut = y * w + x;
				output[idOut].x = (uchar)clampf( sum.x, 0.f, 255.f );
				output[idOut].y = (uchar)clampf( sum.y, 0.f, 255.f );
				output[idOut].z = (uchar)clampf( sum.z, 0.f, 255.f );
				output[idOut].w = 255;
			}
		}
		
	}

	void Ex2(const std::vector<uchar4> &inputImg, // Input image
					const uint imgWidth, const uint imgHeight, // Image size
                    const std::vector<float> &matConv, // Convolution matrix (square)
					const uint matSize, // Matrix size (width or height)
					const std::vector<uchar4> &resultCPU, // Just for comparison
                    std::vector<uchar4> &output // Output image
					)
	{
		uint imgSize = imgWidth*imgHeight;
		// 2 arrays for GPU
		uchar4 *device_input = NULL;
		uchar4 *device_output = NULL;
		
		cudaMalloc(&device_input, sizeof(uchar4)*imgSize);
		cudaMalloc(&device_output, sizeof(uchar4)*imgSize);

		cudaMemcpy(device_input,inputImg.data(), sizeof(uchar4)*imgSize,cudaMemcpyHostToDevice);

		cudaMemcpyToSymbol(dev_conv_const,matConv.data(), sizeof(float)*(matSize*matSize));

		const dim3 threads(32, 32); // 32 * 32 = 1024
		const dim3 blocks( ( imgWidth + threads.x - 1 ) / threads.x, ( imgHeight + threads.y - 1 ) / threads.y );

		CudalutionConst<<<blocks, threads>>>(device_input, device_output, matSize, imgWidth, imgHeight);
		cudaMemcpy(output.data(),device_output,sizeof(uchar4)*imgSize,cudaMemcpyDeviceToHost);

		cudaFree(device_input);
		cudaFree(device_output);

	}

	texture<uchar4, 1, cudaReadModeElementType> Texture1D;
	__global__ void CudalutionTexture1d(uchar4 * input, uchar4 *output, uint convSize, const uint w, const uint h)
	{
		for(uint y = blockIdx.y * blockDim.y + threadIdx.y; y < h; y += gridDim.y * blockDim.y) 
		{
			for(uint x = blockIdx.x * blockDim.x + threadIdx.x; x < w; x += gridDim.x * blockDim.x) 
			{
				float3 sum = make_float3(0.f,0.f,0.f);
				
				// Apply convolution
				for ( uint j = 0; j < convSize; ++j ) 
				{
					for ( uint i = 0; i < convSize; ++i ) 
					{
						int dX = x + i - convSize / 2;
						int dY = y + j - convSize / 2;

						// Handle borders
						if ( dX < 0 ) 
							dX = 0;

						if ( dX >= w ) 
							dX = w - 1;

						if ( dY < 0 ) 
							dY = 0;

						if ( dY >= h ) 
							dY = h - 1;

						const int idMat		= j * convSize + i;
						const int idPixel	= dY * w + dX;
						sum.x += (float)(tex1Dfetch(Texture1D, idPixel).x) * dev_conv_const[idMat];
						sum.y += (float)(tex1Dfetch(Texture1D, idPixel).y) * dev_conv_const[idMat];
						sum.z += (float)(tex1Dfetch(Texture1D, idPixel).z) * dev_conv_const[idMat];
					}
				}
				const int idOut = y * w + x;
				output[idOut].x = (uchar)clampf( sum.x, 0.f, 255.f );
				output[idOut].y = (uchar)clampf( sum.y, 0.f, 255.f );
				output[idOut].z = (uchar)clampf( sum.z, 0.f, 255.f );
				output[idOut].w = 255;
			}
		}
		
	}

	void studentJobTexture1D(const std::vector<uchar4> &inputImg, // Input image
					const uint imgWidth, const uint imgHeight, // Image size
                    const std::vector<float> &matConv, // Convolution matrix (square)
					const uint matSize, // Matrix size (width or height)
					const std::vector<uchar4> &resultCPU, // Just for comparison
                    std::vector<uchar4> &output // Output image
					)
	{
		uint imgSize = imgWidth*imgHeight;
		// 2 arrays for GPU
		uchar4 *device_input = NULL;
		uchar4 *device_output = NULL;
		
		cudaMalloc(&device_input, sizeof(uchar4)*imgSize);
		cudaMalloc(&device_output, sizeof(uchar4)*imgSize);

		cudaBindTexture(0, Texture1D, device_input, imgSize*sizeof(uchar4));

		cudaMemcpy(device_input,inputImg.data(), sizeof(uchar4)*imgSize,cudaMemcpyHostToDevice);

		cudaMemcpyToSymbol(dev_conv_const,matConv.data(), sizeof(float)*(matSize*matSize));

		const dim3 threads(32, 32); // 32 * 32 = 1024
		const dim3 blocks( ( imgWidth + threads.x - 1 ) / threads.x, ( imgHeight + threads.y - 1 ) / threads.y );

		CudalutionTexture1d<<<blocks, threads>>>(device_input, device_output, matSize, imgWidth, imgHeight);
		cudaMemcpy(output.data(),device_output,sizeof(uchar4)*imgSize,cudaMemcpyDeviceToHost);

		cudaFree(device_input);
		cudaFree(device_output);

	}

	texture<uchar4, 2, cudaReadModeElementType> Texture2D;
	__global__ void CudalutionTexture2d(uchar4 * input, uchar4 *output, uint convSize, const uint w, const uint h)
	{
		for(uint y = blockIdx.y * blockDim.y + threadIdx.y; y < h; y += gridDim.y * blockDim.y) 
		{
			for(uint x = blockIdx.x * blockDim.x + threadIdx.x; x < w; x += gridDim.x * blockDim.x) 
			{
				float3 sum = make_float3(0.f,0.f,0.f);
				
				// Apply convolution
				for ( uint j = 0; j < convSize; ++j ) 
				{
					for ( uint i = 0; i < convSize; ++i ) 
					{
						int dX = x + i - convSize / 2;
						int dY = y + j - convSize / 2;

						// Handle borders
						if ( dX < 0 ) 
							dX = 0;

						if ( dX >= w ) 
							dX = w - 1;

						if ( dY < 0 ) 
							dY = 0;

						if ( dY >= h ) 
							dY = h - 1;

						const int idMat		= j * convSize + i;
						sum.x += (float)(tex2D(Texture2D, dX, dY).x) * dev_conv_const[idMat];
						sum.y += (float)(tex2D(Texture2D, dX, dY).y) * dev_conv_const[idMat];
						sum.z += (float)(tex2D(Texture2D, dX, dY).z) * dev_conv_const[idMat];
					}
				}
				const int idOut = y * w + x;
				output[idOut].x = (uchar)clampf( sum.x, 0.f, 255.f );
				output[idOut].y = (uchar)clampf( sum.y, 0.f, 255.f );
				output[idOut].z = (uchar)clampf( sum.z, 0.f, 255.f );
				output[idOut].w = 255;
			}
		}
		
	}

	void studentJob(const std::vector<uchar4> &inputImg, // Input image
					const uint imgWidth, const uint imgHeight, // Image size
                    const std::vector<float> &matConv, // Convolution matrix (square)
					const uint matSize, // Matrix size (width or height)
					const std::vector<uchar4> &resultCPU, // Just for comparison
                    std::vector<uchar4> &output // Output image
					)
	{
		uint imgSize = imgWidth*imgHeight;
		// 2 arrays for GPU
		uchar4 *device_input = NULL;
		uchar4 *device_output = NULL;
		size_t pitch;
		cudaMalloc(&device_output, sizeof(uchar4)*imgSize);

		cudaMallocPitch(&device_input, &pitch, imgWidth*sizeof(uchar4), imgHeight);
		cudaMemcpy2D(device_input, pitch, inputImg.data(), imgWidth*sizeof(uchar4), imgWidth*sizeof(uchar4), imgHeight, cudaMemcpyHostToDevice);
		Texture2D.normalized = false;
		cudaBindTexture2D(0, Texture2D, device_input, cudaCreateChannelDesc<uchar4>(), imgWidth, imgHeight, pitch);

		cudaMemcpyToSymbol(dev_conv_const,matConv.data(), sizeof(float)*(matSize*matSize));
		const dim3 threads(32, 32); // 32 * 32 = 1024
		const dim3 blocks( ( imgWidth + threads.x - 1 ) / threads.x, ( imgHeight + threads.y - 1 ) / threads.y );
		CudalutionTexture2d<<<blocks, threads>>>(device_input, device_output, matSize, imgWidth, imgHeight);
		cudaMemcpy(output.data(),device_output,sizeof(uchar4)*imgSize,cudaMemcpyDeviceToHost);
		cudaFree(device_input);
		cudaFree(device_output);

	}
}
