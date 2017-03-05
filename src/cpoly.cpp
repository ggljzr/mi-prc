#ifndef CPOLY_H_
#define CPOLY_H_

#include <cstdio>
#include <cstdlib>
#include <math.h>

#include "cpoly.hpp"

#define EPSILON 0.001

double frand(double min, double max)
{
    double f = (double)rand() / RAND_MAX;
    return min + f * (max - min);
}

bool compare_double(double a, double b)
{
    return fabs(a - b) < EPSILON;
}

CPoly::CPoly(double * coefs, int deg)
{
	m_deg = deg;
	m_len = deg + 1;
	m_coefs = new double[m_len];
	for(int i = 0; i < m_len; i++)
		m_coefs[i] = coefs[i];
}

CPoly::CPoly(int deg)
{
	m_deg = deg;
	m_len = deg + 1;
	m_coefs = new double[m_len];
	for(int i = 0; i < m_len; i++)
		m_coefs[i] = 0;
}

CPoly::~CPoly()
{
	delete [] m_coefs;
}

bool CPoly::compare(CPoly * a, CPoly * b)
{
	if(a->m_deg != b->m_deg)
		return false;

	for(int i = 0; i < a->m_len; i++)
	{
		if(!compare_double(a->m_coefs[i], b->m_coefs[i]))
			return false;
	}

	return true;
}

CPoly * CPoly::triv_mult(CPoly * a, CPoly * b)
{
	
    CPoly * res = new CPoly(a->m_deg + b->m_deg);
    long long counter = 0;

    for(int i = 0; i <= a->m_deg; i++)
    {
        for(int j = 0; j <= b->m_deg; j++)
        {
            res->m_coefs[i + j] += a->m_coefs[i] * b->m_coefs[j];
            counter++;
        }
    }

    printf("triv flops = %lld\n", counter);

    return res;
}

/*
Karatsuba's iterative algorithm. 
Implementation based on:
https://eprint.iacr.org/2006/224.pdf
*/
CPoly * CPoly::karatsuba(CPoly * a, CPoly * b)
{
	CPoly * res = new CPoly(a->m_deg + b->m_deg);

	int n = a->m_len;

	double * D = new double[n];
	for(int i = 0; i < n; i++)
		D[i] = a->m_coefs[i] * b->m_coefs[i];

	double * S = new double[2*n - 1];
	double * T = new double[2*n - 1];

	long long counter = 0;

	for(int i = 1; i < 2*n - 2; i++)
	{
		for(int s = 0; s < i; s++)
		{
			int t = i - s;
			if(s >= t) break;

			if(t < a->m_len){
				double as = a->m_coefs[s];
				double bs = b->m_coefs[s];

				double at = a->m_coefs[t];
				double bt = b->m_coefs[t];

				S[i] += (as + at) * (bs + bt);
				T[i] += D[s] + D[t];
				counter++;
			}
		}
	}

	res->m_coefs[0] = D[0];
	res->m_coefs[2*n - 2] = D[n - 1];

	for(int i = 1; i < 2*n - 2; i++)
	{
		if(i % 2 == 1)
		{
			res->m_coefs[i] = S[i] - T[i];
		}
		else
		{
			res->m_coefs[i] = S[i] - T[i] + D[i / 2];
		}
	}

	printf("karatsuba flops = %lld\n", counter);

	delete [] D;
	delete [] S;
	delete [] T;
	return res;
}

void CPoly::randomize()
{
	for(int i = 0; i < m_len; i++)
		m_coefs[i] = frand(-100,100);
}

void CPoly::print_poly()
{
	for(int i = m_deg; i > 0; i--)
        printf("%.02f x^%d ", m_coefs[i], i);
    printf("%f x^0 \n", m_coefs[0]);
}

#endif