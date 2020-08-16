#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void helloWorld()
{
	printf("\nHello world executing in Block:'%d,%d,%d' Thread:'%d,%d,%d' GridDim:'%d,%d,%d'",
		blockIdx.x,blockIdx.y,blockIdx.z,threadIdx.x,threadIdx.y,threadIdx.z,gridDim.x,gridDim.y,gridDim.x);
}

int main()
{
	dim3 number_of_blocks_per_grid (1,1,1); //Also defined as grid(2,2,2)
	dim3 number_of_threads_per_block(1,1,1); //Also defined as block(2,2,2)

	//FORMAT: kernel<<<number_of_blocks,number_of_threads_per_block>>>()
	helloWorld << <number_of_blocks_per_grid,number_of_threads_per_block>> > ();
	cudaDeviceSynchronize();
	cudaDeviceReset();

	return 0;
}