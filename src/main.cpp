#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <time.h>

#include "cpoly.hpp"

using namespace std;

int main(int argc, char * argv[])
{

    srand(time(NULL));

    int pol_deg = 10000;

    if(argc > 1)
        pol_deg = atoi(argv[1]);

    printf("Polynomial degree = %d\n\n", pol_deg);

    clock_t start_triv_t = 0;
    clock_t end_triv_t = 0;

    clock_t start_kara_t = 0;
    clock_t end_kara_t = 0;

    double total_kara_t = 0;
    double total_triv_t = 0;

    CPoly * pa = new CPoly(pol_deg);
    CPoly * pb = new CPoly(pol_deg);

    pa->randomize();
    pb->randomize();

    start_triv_t = clock();
    CPoly * res_triv = CPoly::triv_mult(pa, pb);
    end_triv_t = clock();

    start_kara_t = clock();
    CPoly * res = CPoly::karatsuba(pa, pb);
    end_kara_t = clock();

    total_triv_t = (double)(end_triv_t - start_triv_t) / (CLOCKS_PER_SEC);
    total_kara_t = (double)(end_kara_t - start_kara_t) / (CLOCKS_PER_SEC);

    if(CPoly::compare(res, res_triv))
        printf("OK\n");
    else
        printf("Error\n");

    printf("\n");

    printf("Time triv: %f\n", total_triv_t);
    printf("Time kara: %f\n", total_kara_t);

    delete res;
    delete res_triv;
    delete pa;
    delete pb;

    return 0;
}
