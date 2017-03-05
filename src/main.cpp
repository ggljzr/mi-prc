#include <cstdio>
#include <iostream>
#include <cstdlib>
#include "cpoly.hpp"

using namespace std;

int main(int argc, char * argv[])
{

    srand(1);

    double a[7] = {32, 8, 7, 8, 12, 5, 3};
    double b[7] = {12, 0, 8, 4, 25, 8, 7};

    CPoly * pa = new CPoly(a, 6);
    CPoly * pb = new CPoly(b, 6);
    CPoly * pc = new CPoly(10000);

    pc->randomize();
    //pc->print_poly();

    CPoly * res_triv = CPoly::triv_mult(pc, pc);
    printf("TRIV:\n");
    //res_triv->print_poly();

    CPoly * res = CPoly::karatsuba(pc, pc);
    printf("KARATSUBA:\n");
    //res->print_poly();

    if(CPoly::compare(res, res_triv))
        printf("OK");
    else
        printf("Error");

    delete res;
    delete res_triv;

    return 0;
}
