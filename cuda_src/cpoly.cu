#ifndef CPOLY_H_
#define CPOLY_H_

#include <math.h>
#include <cstdio>
#include <cstdlib>

#include "cpoly.h"

#define EPSILON 0.01

#define MUH_MAX 2147483647

float frand(float min, float max) {
  float f = (float)rand() / RAND_MAX;
  return min + f * (max - min);
}

bool compare_float(float a, float b) { return fabs(a - b) < EPSILON; }

CPoly::CPoly(float* coefs, int deg) {
  m_deg = deg;
  m_len = deg + 1;
  m_coefs = new float[m_len];
  for (int i = 0; i < m_len; i++) m_coefs[i] = coefs[i];
}

CPoly::CPoly(int deg) {
  m_deg = deg;
  m_len = deg + 1;
  m_coefs = new float[m_len];
  for (int i = 0; i < m_len; i++) m_coefs[i] = 0;
}

CPoly::~CPoly() { delete[] m_coefs; }

bool CPoly::compare(CPoly* a, CPoly* b) {
  if (a == NULL || b == NULL) return false;

  if (a->m_deg != b->m_deg) return false;

  for (int i = 0; i < a->m_len; i++) {
    if (!compare_float(a->m_coefs[i], b->m_coefs[i])) {
      printf("A: %f B:%f (x^%d)\n", a->m_coefs[i], b->m_coefs[i], i);
      return false;
    }
  }

  return true;
}

CPoly* CPoly::triv_mult(CPoly* a, CPoly* b) {
  CPoly* res = new CPoly(a->m_deg + b->m_deg);

  for (int i = 0; i <= a->m_deg; i++) {
    for (int j = 0; j <= b->m_deg; j++) {
      res->m_coefs[i + j] += a->m_coefs[i] * b->m_coefs[j];
    }
  }

  return res;
}

/*
Karatsuba's iterative algorithm.
Implementation based on:
https://eprint.iacr.org/2006/224.pdf
*/
CPoly* CPoly::karatsuba(CPoly* a, CPoly* b) {
  CPoly* res = new CPoly(a->m_deg + b->m_deg);

  int n = a->m_len;

  float* D = new float[n];
  for (int i = 0; i < n; i++) D[i] = a->m_coefs[i] * b->m_coefs[i];

  float* S = new float[2 * n - 1];
  float* T = new float[2 * n - 1];

  for(int i = 0; i < 2 * n - 1; i++)
  {
    S[i] = 0;
    T[i] = 0;
  }

  for (int i = 1; i < 2 * n - 2; i++) {
    for (int s = 0; s < i; s++) {
      int t = i - s;
      if (s >= t) break;
      if (t >= n) continue;

      float as = a->m_coefs[s];
      float bs = b->m_coefs[s];

      float at = a->m_coefs[t];
      float bt = b->m_coefs[t];

      T[i] += D[s] + D[t];

      S[i] += (as + at) * (bs + bt);
    }
  }

  res->m_coefs[0] = D[0];
  res->m_coefs[2 * n - 2] = D[n - 1];

  for (int i = 1; i < 2 * n - 2; i++) {
    if ((i & 0x01) == 1) {
      res->m_coefs[i] = S[i] - T[i];
    } else {
      res->m_coefs[i] = S[i] - T[i] + D[i / 2];
    }
  }

  delete[] D;
  delete[] S;
  delete[] T;
  return res;
}

void CPoly::randomize(float min, float max) {
  for (int i = 0; i < m_len; i++) {
    m_coefs[i] = frand(min, max);
  }
}

void CPoly::print_poly() {
  for (int i = m_deg; i > 0; i--) printf("%.02f x^%d ", m_coefs[i], i);
  printf("%f x^0 \n", m_coefs[0]);
}

#endif
