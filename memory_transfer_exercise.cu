#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void memory_transfer(int *d_input, int size)
{
	//gid calculation [notice patterns]
	int blockId = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	int gid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y) 
		+ blockId * (blockDim.x * blockDim.y * blockDim.z);

	if (gid < size)
		printf("\nDevice with gid='%d' contains value '%d'",gid,d_input[gid]);
}

int main()
{
	//Size of array
	int size = 64;

	//Allocating memory in host memory
	int* h_input = (int*)malloc(size * sizeof(int));

	//Allocating memory in device memory
	int* d_input;
	cudaMalloc((void**)&d_input, size * sizeof(int));

	//Random set h_input
	srand(1234);
	for (int i = 0;i < size;i++)
	{
		h_input[i] = rand()&0xff;
	}

	//Copy memory from host to device
	cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice);

	//Call Kernel
	dim3 grid(4, 4, 4);
	dim3 block(2, 2, 2);

	memory_transfer << <grid, block >> > (d_input, size);
	cudaDeviceSynchronize();

	//Cuda memory free-up
	cudaFree(d_input);
	free(h_input);

	cudaDeviceReset();

	return(0);
}