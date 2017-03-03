#ifndef CPOLY_H_
#define CPOLY_H_

#include <cstdio>

#include "cpoly.hpp"

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

CPoly * CPoly::triv_mult(CPoly * a, CPoly * b)
{
	
    CPoly * res = new CPoly(a->m_deg + b->m_deg);

    for(int i = 0; i <= a->m_deg; i++)
    {
        for(int j = 0; j <= b->m_deg; j++)
        {
            res->m_coefs[i + j] += a->m_coefs[i] * b->m_coefs[j];
        }
    }

    return res;
}

void CPoly::print_poly()
{
	for(int i = 0; i < m_deg; i++)
        printf("%.02f x^%d ", m_coefs[i], m_deg - i);
    printf("%f x^0 \n", m_coefs[m_deg]);
}

#endif