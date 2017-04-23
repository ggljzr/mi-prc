#ifndef CUDA_CARA_H
#define CUDA_CARA_H

#include <math.h>
#include <stdio.h>
#include <time.h>

#include "cpoly.h"

__global__ void kernel_grid_test(int n, float * S, float * T)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= 2 * n - 2 || i == 0)
    return;

	printf("kernele num = %d dev_s = %f dev_d = %f \n", i, S[i], T[i]);
}

__global__ void kernel_kara_st(float *A, float *B, float *D, int n, float *S,
                               float *T) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i >= 2 * n - 2 || i == 0)
    return;

  for (int s = 0; s < i; s++) {
    int t = i - s;
    if (s >= t)
      break;
    if (t >= n)
      continue;

    float as = A[s];
    float bs = B[s];

    float at = A[t];
    float bt = B[t];

    atomicAdd(&(T[i]), (float)(D[s] + D[t]));
    atomicAdd(&(S[i]), (float)((as + at) * (bs + bt)));
  }
}

__global__ void kernel_kara_res(float *S, float *T, float *D, int n, float *res) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= 2 * n - 2 || i == 0)
    return;

  if ((i & 0x01) == 1) {
    res[i] = S[i] - T[i];
  } else {
    res[i] = S[i] - T[i] + D[i / 2];
  }
}

float para_kara_mult(CPoly *A, CPoly *B, CPoly *C) {
  
	int n = A->m_len;
	int grid_x = (int)ceil((float) (2 * n - 2) / 16.0);

	dim3 dim_block(512);
	dim3 dim_grid(grid_x);

	cudaEvent_t start;
	cudaEvent_t stop;
	float elapsed_time = 0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float *D = new float[n];

	float *dev_A;
	float *dev_B;
	float *dev_C;

	float *dev_S;
	float *dev_T;
	float *dev_D;

	cudaMalloc((void**)&dev_A, A->m_len * sizeof(float));
	cudaMalloc((void**)&dev_B, B->m_len * sizeof(float));

	cudaMalloc((void**)&dev_S, (2 * n - 1) * sizeof(float));
	cudaMalloc((void**)&dev_T, (2 * n - 1) * sizeof(float));
	cudaMalloc((void**)&dev_D, n * sizeof(float));

	for(int i = 0; i < n; i++)
			D[i] = A->m_coefs[i] * B->m_coefs[i];

	cudaMemcpy(dev_A, (float *)A->m_coefs, A->m_len * sizeof(float),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, (float *)B->m_coefs, B->m_len * sizeof(float),
		cudaMemcpyHostToDevice);

	cudaMemset(dev_S, 0, (2 * n - 1) * sizeof(float));
	cudaMemset(dev_T, 0, (2 * n - 1) * sizeof(float));
	cudaMemcpy(dev_D, (float *)D, n * sizeof(float),
		cudaMemcpyHostToDevice);

	cudaEventRecord(start, 0);
	kernel_kara_st<<<dim_grid, dim_block>>>(dev_A, dev_B, dev_D, n, dev_S, dev_T);
	cudaThreadSynchronize();

	cudaMalloc((void**)&dev_C, C->m_len * sizeof(float));
	cudaMemset(dev_C, 0, C->m_len * sizeof(float));

	kernel_kara_res<<<dim_grid, dim_block>>>(dev_S, dev_T, dev_D, n, dev_C);
	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);

	cudaMemcpy(C->m_coefs, dev_C, C->m_len * sizeof(float), cudaMemcpyDeviceToHost);

	C->m_coefs[0] = D[0];
  C->m_coefs[2 * n - 2] = D[n - 1];

  delete D;

  return elapsed_time;
}

#endif