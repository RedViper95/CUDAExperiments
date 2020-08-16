/*#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void memory_transfer(int* input, int array_size)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid < array_size)
		printf("\narray_size = '%d' Input value = '%d' inside device with gid = '%d' threadIdX.x = '%d'", array_size, *(input + gid), gid, threadIdx.x);
}

int main()
{
	//Setting up stream size
	int stream_size = 150;
	int array_size = stream_size * sizeof(int);

	//Allocating memory in host
	int *h_input = (int*)malloc(array_size);

	//Filling up host memory with random values
	srand(1460);
	for (int i = 0; i < stream_size; i++) 
	{
		h_input[i] = (int)(rand() & 0xff);
	}

	//Allocating memory in device
	int *d_input;
	cudaMalloc((void**)&d_input, array_size); //The first argument must be a double pointer 

	//Copy stuff from h_input to d_input
	cudaMemcpy(d_input, h_input, array_size, cudaMemcpyHostToDevice);

	//Launch kernel
	dim3 block(32,1,1); //A single block has 32 threads in x and 1,1 in y and z
	dim3 grid(5,1,1); //A single grid has 5 blocks in x and 1,1 in y and z
	memory_transfer << < grid, block >> > (d_input, stream_size);
	cudaDeviceSynchronize(); //Wait for kernel to launch

	cudaFree(d_input); //Reclaim memory in device
	free(h_input); //Reclaim memory in host

	cudaDeviceReset();
	return 0;
}*/