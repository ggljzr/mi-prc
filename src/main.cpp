#include <cstdio>
#include <iostream>
#include "cpoly.hpp"

using namespace std;

double * triv_mult(double * a, double * b, int deg_a, int deg_b)
{
    double * res = new double[deg_a + deg_b];
    
    for(int i = 0; i < deg_a + deg_b; i++)
        res[i] = 0;

    for(int i = 0; i <= deg_a; i++)
    {
        for(int j = 0; j <= deg_b; j++)
        {
            res[i + j] += a[i] * b[j];
        }
    }

    return res;
}

int main(int argc, char * argv[])
{

    double a[5] = {23,41,35,87,11};
    double b[7] = {11,15,18,12,3,7,9};

    CPoly * pa = new CPoly(a, 4);
    CPoly * pb = new CPoly(b, 6);

    pa->print_poly();
    pb->print_poly();

    CPoly * res = CPoly::triv_mult(pa, pb);

    res->print_poly();

    return 0;
}
