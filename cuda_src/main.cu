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
    printf("Device number: %d\n", i);
    printf("Device name: %s\n", prop.name);
    printf("CC: %d.%d\n", prop.major, prop.minor);
  }
  printf("\n");

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
  CPoly* res = CPoly::triv_mult(&A, &B);
  end_triv_t = clock();

  total_triv_t = (double)(end_triv_t - start_triv_t) / (CLOCKS_PER_SEC);

  float total_time_cuda = para_triv_mult(&A, &B, &res_cuda);

  if (CPoly::compare(res, &res_cuda))
    printf("OK\n");
  else
    printf("Error\n");
  printf("\n");

  // res->print_poly();
  printf("Total time seq: %f ms\n", total_triv_t * 1000);
  printf("Total time cuda: %f ms\n", total_time_cuda);

  if (res->m_len < 20) {
    res->print_poly();
  }

  para_kara_mult(NULL, NULL, NULL);

  return 0;
}