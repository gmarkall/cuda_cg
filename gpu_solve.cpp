// Copyright (c) 2019 Graham Markall
// All rights reserved.
//
// pcg_solve.cpp
//
// Implements a Jacobi preconditioned sparse conjugate gradient solver, similar
// to the HPCG benchmark.

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <sys/time.h>

#include "gpu_solve.h"
#include "opcount.h"

// Solver parameters - relative tolerance and maximum iterations
#define epsilon 1e-7
#define IMAX 40000

// Flag for counting operations
bool opcount = true;

// For timing solver
double utime() {
  struct timeval tv;

  gettimeofday(&tv, NULL);

  return (tv.tv_sec + tv.tv_usec * 1e-6);
}

// Creates a diagonal matrix stored in a vector pcmat, from the CSR matrix.
//. n is the matrix size.
void create_jac(int n, int *row_ptr, int *col_idx, double *val,
                           double *pcmat) {
  for (int i = 0; i < n; i++)
    for (int k = row_ptr[i] - 1; k < row_ptr[i + 1] - 1; k++)
      if (col_idx[k] == i + 1)
        pcmat[i] = 1.0 / val[k];
}

// Multiplies diagonal matrix mat stored as a vector by the vector src, storing
// result in dest. n is the vector length.
void diag_spmv(int n, double *mat, double *src, double *dest) {
  for (int i = 0; i < n; i++)
    dest[i] = mat[i] * src[i];
}

// Sets the length-n vector vec to the zero vector.
void veczero(int n, double *vec) {
  for (int i = 0; i < n; i++)
    vec[i] = 0;
}

// Multiply a CSR matrix by src and store the rresult in dest.
// n is the matrix size/vector length.
void csr_spmv(int n, double *src, double *dest, int *row_ptr, int *col_idx, double *val) {
  opcount_start_kernel("csr_spmv");

  int row;

  // Count row = 0 in for loop
  opcount_add_mem_write(&row);
  // Assume n held in register, so don't count mem read for it
  for (row = 0; row < n; row++) {
    opcount_start_loop_iteration();
    // Assume dest and row in registers. Count read of dest[row]
    opcount_add_mem_read(&dest[row]);
    dest[row] = 0;
    // Assume row_ptr and row in registers. Count reads of row_ptr[row,row+1]
    opcount_add_mem_read(&row_ptr[row]);
    opcount_add_mem_read(&row_ptr[row+1]);
    int a = row_ptr[row];
    int b = row_ptr[row + 1];

    int k;
    // Assume a went to a register, write k to memory.
    opcount_add_mem_write(&k);

    for (int k = a; k < b; k++) {
      opcount_start_loop_iteration();

      // Read of dest[row]
      opcount_add_mem_read(&dest[row]);
      // Read val[k-1]
      opcount_add_mem_read(&val[k-1]);
      // Read col_idx[k-1]
      opcount_add_mem_read(&col_idx[k-1]);
      // Read src[col_idx[k-1]-1]
      opcount_add_mem_read(&src[col_idx[k-1]]);
      // Write dest[row]
      opcount_add_mem_write(&dest[row]);
      dest[row] += val[k - 1] * src[col_idx[k - 1] - 1];

      opcount_finish_loop_iteration();
    }

    opcount_finish_loop_iteration();
  }

  opcount_finish_kernel("csr_spmv");
}

// Computes the dot product of length-n vectors vec1 and vec2.
double vecdot(int n, double *vec1, double *vec2) {
  double res = 0;

  for (int i = 0; i < n; i ++)
    res += vec1[i] * vec2[i];

  return res;
}

// Computes r= a*x+y for n-length vectors x and y, and scalar a.
void axpy(int n, double a, double *x, double *y, double *r) {
  for (int i = 0; i < n; i++)
    r[i] = y[i] + a * x[i];
}

// Computes y= y-a*x for n-length vectors x and y, and scalar a.
void ymax(int n, double a, double *x, double *y) {
  for (int i = 0; i < n; i++)
    y[i] = y[i] - a * x[i];
}

// Sets dest=src for n-length vectors on the GPU.
void vecassign(double *dest, double *src, int n) {
  memcpy(dest, src, sizeof(double) * n);
}

// Main solver entry point. Solves Ax = b for x.
// CSR matrix (A) stores the row and column pointer information like Fortran
// indices (i.e. begin at 1) findrm_p     : matrix row pointer colm_p       :
// matrix column pointer matrix_val_p : matrix values size*        : size of
// each vector b_p          : pointer to RHS vector x_p          : solution (x)
// is returned here
void pcg_solve(int n, int *row_ptr, int *col_idx, double *val, double *x, double *b)
{
  // Temporary vectors
  double *r, *d, *q, *s;
  r = (double*)malloc(sizeof(double) * n);
  d = (double*)malloc(sizeof(double) * n);
  q = (double*)malloc(sizeof(double) * n);
  s = (double*)malloc(sizeof(double) * n);

  // Diagonal matrix for preconditioning (stored as a vector)
  double *jac;
  jac = (double*)malloc(sizeof(double) * n);

  // Temporary scalars
  double t, alpha, snew, beta, sold, s0;

  int iterations = 0;

  // Begin timing
  t = -utime();

  // Create diagonal preconditioning matrix (J = 1/diag(M))
  create_jac(n, row_ptr, col_idx, val, jac);

  // Initialise result vector (x=0)
  veczero(n, x);

  // r=b-Ax (r=b since x=0), and d=M^(-1)r
  memcpy(r, b, sizeof(double) * n);
  diag_spmv(n, jac, r, d);

  // s0 = r.d
  s0 = vecdot(n, r, d);
  // snew = s0
  snew = s0;

  // While i < imax and snew > epsilon^2*s0
  while (iterations < IMAX && snew > epsilon * epsilon * s0) {
    // q = Ad
    csr_spmv(n, d, q, row_ptr, col_idx, val);
    // alpha = snew/(d.q)
    alpha = vecdot(n, d, q);
    alpha = snew / alpha;
    // x = x + alpha*d
    axpy(n, alpha, d, x, x);
    // r = r - alpha*q
    ymax(n, alpha, q, r);
    // s = M^(-1)r
    diag_spmv(n, jac, r, s);
    // sold = snew
    sold = snew;
    // snew = r.s
    snew = vecdot(n, r, s);
    // beta = snew/sold
    beta = snew / sold;
    // d = s + beta*d
    axpy(n, beta, d, s, d);
    // i = i+1
    iterations++;
  }

  // Clean up
  free(r);
  free(d);
  free(q);
  free(jac);

  // End timing
  t += utime();

  // Interesting information
  printf("Iterations: %d \n", iterations);
  printf("Solve time: %f seconds\n", t);
}
