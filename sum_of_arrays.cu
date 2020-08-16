#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void sum_of_arrays(int*a,int*b,int*c,int size)
{
	//gid calculation [notice patterns]
	int blockId = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	int gid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y)
		+ blockId * (blockDim.x * blockDim.y * blockDim.z);

	if (gid < size)
		c[gid] = a[gid] + b[gid];
		printf("\nDevice with gid='%d' has input a ='%d' input b ='%d and sum c='%d'", gid,a[gid],b[gid],c[gid]);
}

void sum_of_arrays_cpu(int*a, int*b, int*c_verify, int size)
{
	for (int i = 0;i < size;i++)
	{
		c_verify[i] = a[i] + b[i];
	}
}

void compare_array(int* a, int* b,int size)
{
	for (int i = 0;i < size;i++)
	{
		if (a[i] != b[i])
		{
			printf("\nArrays are not equal!!");
			return;
		}
	}
	printf("\nArrays are same");
}

int main()
{
	//Set array sizes
	int size = 10000;
	int block_size = 128;

	//Allocate memory in host
	int* h_a = (int*)malloc(size * sizeof(int));
	int* h_b = (int*)malloc(size * sizeof(int));
	int* gpu_results = (int*)malloc(size * sizeof(int));

	//Allocate memory in device
	int* d_a, * d_b, * d_c;
	cudaMalloc((int**)&d_a, size * sizeof(int));
	cudaMalloc((int**)&d_b, size * sizeof(int));
	cudaMalloc((int**)&d_c, size * sizeof(int));

	//Fill up h_a, h_b
	time_t t;
	srand((unsigned)&t);
	for (int i = 0;i < size;i++)
	{
		h_a[i] = (int)(rand() & 0xff);
	}
	for (int j = 0;j < size;j++)
	{
		h_b[j] = (int)(rand() & 0xff);
	}

	//Copy inputs from host to device
	cudaMemcpy(d_a, h_a, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size * sizeof(int), cudaMemcpyHostToDevice);

	//Launch Kernel
	dim3 grid(10000/128+1, 1, 1);
	dim3 block(128,1,1);
	sum_of_arrays << <grid, block >> > (d_a,d_b,d_c,size);
	cudaDeviceSynchronize();

	//Copy back d_c from device to host
	cudaMemcpy(gpu_results,d_c,size * sizeof(int), cudaMemcpyDeviceToHost);

	int* c_verify = (int*)malloc(size * sizeof(int));
	sum_of_arrays_cpu(h_a, h_b, c_verify, size);
	compare_array(gpu_results,c_verify,size);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	free(h_a);
	free(h_b);
	free(gpu_results);

	cudaDeviceReset();
	return 0;
}