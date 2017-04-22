#include <stdio.h>
#include <time.h>
#include <math.h>

#include "cpoly.h"

__global__ void kernel_triv_mult(float * A, float * B, float * C, int n, int m)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
  
  if (i >= n || j >= m) return;

  float val = A[i] * B[j];
  atomicAdd(&(C[i + j]), (float)val);

}

float para_triv_mult(CPoly * A, CPoly * B, CPoly * C)
{
	int grid_x = (int)ceil((float)A->m_len / 16.0);
	int grid_y = (int)ceil((float)B->m_len / 32.0);

	dim3 dim_block(16, 32);
	dim3 dim_grid(grid_x, grid_y);

	cudaEvent_t start;
	cudaEvent_t stop;
	float elapsed_time = 0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float * dev_a;
	float * dev_b;
	float * dev_c;

	cudaMalloc((void**)&dev_a, A->m_len * sizeof(float));
	cudaMalloc((void**)&dev_b, B->m_len * sizeof(float));
	cudaMalloc((void**)&dev_c, C->m_len * sizeof(float));

	cudaMemcpy(dev_a, (float *)A->m_coefs, A->m_len * sizeof(float),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, (float *)B->m_coefs, B->m_len * sizeof(float),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_c, (float *)C->m_coefs, C->m_len * sizeof(float),
		cudaMemcpyHostToDevice);

	cudaEventRecord(start, 0);
	kernel_triv_mult<<<dim_grid, dim_block>>>(dev_a, dev_b, dev_c, A->m_len, B->m_len);
	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);

	cudaMemcpy(C->m_coefs, dev_c, C->m_len * sizeof(float), cudaMemcpyDeviceToHost);

	return elapsed_time;
}

int main(int argc, char * argv[])
{
	int n_devices = 0;
	cudaGetDeviceCount(&n_devices);
	srand(time(NULL));

	for(int i = 0; i < n_devices; i++)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device number: %d\n", i);
		printf("Device name: %s\n", prop.name);
		printf("CC: %d.%d\n", prop.major, prop.minor);
	}
	printf("\n");

	int n = 10;
	if(argc > 1)
		n = atoi(argv[1]);

	CPoly A(n);
	A.randomize();
	CPoly B(n);
	B.randomize();

	clock_t start_triv_t = 0;
    clock_t end_triv_t = 0;
    double total_triv_t = 0;

	CPoly res_cuda(2 * n);

	start_triv_t = clock();
	CPoly * res = CPoly::triv_mult(&A, &B);
	end_triv_t = clock();

	total_triv_t = (double)(end_triv_t - start_triv_t) / (CLOCKS_PER_SEC);

	float total_time_cuda = para_triv_mult(&A, &B, &res_cuda);

	if(CPoly::compare(res, &res_cuda))
        printf("OK\n");
    else
        printf("Error\n");
    printf("\n");

	//res->print_poly();
	printf("Total time seq: %f ms\n", total_triv_t * 1000);
	printf("Total time cuda: %f ms\n", total_time_cuda);

	if(res->m_len < 20){
		res->print_poly();
	}
	return 0;
}