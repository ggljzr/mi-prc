#include <cstdio>
#include <iostream>
#include <cstdlib>
#include "cpoly.hpp"

using namespace std;

int main(int argc, char * argv[])
{

    srand(1);

    CPoly * pa = new CPoly(10000);
    CPoly * pb = new CPoly(10000);

    pa->randomize();
    pb->randomize();

    CPoly * res_triv = CPoly::triv_mult(pa, pb);
    CPoly * res = CPoly::karatsuba(pa, pb);

    if(CPoly::compare(res, res_triv))
        printf("OK\n");
    else
        printf("Error\n");

    delete res;
    delete res_triv;

    return 0;
}
