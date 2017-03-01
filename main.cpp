#include <cstdio>
#include <iostream>

using namespace std;

void print_poly(double * poly, int deg)
{
    for(int i = 0; i < deg; i++)
        printf("%.02f x^%d ", poly[i], deg - i);
    printf("%f x^0 \n", poly[deg]);
}

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

    double * res = triv_mult(a,b, 4, 6);
    print_poly(a, 4);
    print_poly(b, 6);
    print_poly(res, 10);

    return 0;
}
