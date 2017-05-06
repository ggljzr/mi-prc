#include <math.h>
#include <stdio.h>
#include <time.h>

#include "cpoly.h"
#include "cuda_mult.h"

int main(int argc, char* argv[]) {
  int n_devices = 0;
  cudaGetDeviceCount(&n_devices);
  srand(time(NULL));

  for (int i = 0; i < n_devices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    fprintf(stderr, "Device number: %d\n", i);
    fprintf(stderr, "Device name: %s\n", prop.name);
    fprintf(stderr, "CC: %d.%d\n", prop.major, prop.minor);
  }

  int n = 10;
  if (argc > 1) n = atoi(argv[1]);

  CPoly A(n);
  A.randomize();
  CPoly B(n);
  B.randomize();

  clock_t start_triv_t = 0;
  clock_t end_triv_t = 0;
  double total_triv_t = 0;

  CPoly res_cuda(2 * n);

  start_triv_t = clock();
  //CPoly* res = CPoly::triv_mult(&A, &B);
  CPoly* res = CPoly::triv_mult(&A, &B);
  end_triv_t = clock();

  total_triv_t = (double)(end_triv_t - start_triv_t) / (CLOCKS_PER_SEC);
  total_triv_t *= 1000;

  float total_cuda_triv_t = para_triv_mult(&A, &B, &res_cuda);
  float total_cuda_kara_t = para_kara_mult(&A, &B, &res_cuda);

  if (CPoly::compare(res, &res_cuda))
    fprintf(stderr, "OK\n");
  else
    fprintf(stderr, "Error\n");

  if (res->m_len < 20) {
    res->print_poly();
    res_cuda.print_poly();
  }

 
  printf("%d %f %f %f\n", n, total_triv_t, total_cuda_triv_t, total_cuda_kara_t);

  delete res;

  return 0;
}
