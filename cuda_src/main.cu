#include <stdio.h>

#include "cpoly.h"

__global__ void kernel_triv_mult(float * A, float * B, float * C, int n, int m)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
    if(i > n || j > m)
    	return;

    C[i+j] = 6;

}

void para_triv_mult(CPoly * A, CPoly * B, CPoly * C)
{
	dim3 dim_block(1, 1);
	dim3 dim_grid(A->m_len, B->m_len);

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

	kernel_triv_mult<<<dim_grid, dim_block>>>(dev_a, dev_b, dev_c, A->m_len, B->m_len);
	cudaThreadSynchronize();

	cudaMemcpy(C->m_coefs, dev_c, C->m_len * sizeof(float), cudaMemcpyDeviceToHost);

}

int main(int argc, char * argv[])
{
	int n_devices = 0;
	cudaGetDeviceCount(&n_devices);

	for(int i = 0; i < n_devices; i++)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device number: %d\n", i);
		printf("Device name: %s\n", prop.name);
		printf("CC: %d.%d\n", prop.major, prop.minor);
	}

	CPoly A(10);
	A.randomize();
	CPoly B(10);
	B.randomize();

	CPoly res_cuda(20);
	CPoly * res = CPoly::triv_mult(&A, &B);

	para_triv_mult(&A, &B, &res_cuda);
	res_cuda.print_poly();

	//res->print_poly();

	return 0;
}