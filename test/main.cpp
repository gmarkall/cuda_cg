#include "sparse_io.h"
#include "vec.h"
#include "gpu_solve.h"
#include <iostream>

#define MAX_ERROR 1.0e-6

using namespace std;

template <class IndexType, class ValueType>
void cg(csr_matrix<IndexType, ValueType> A, vec<IndexType, ValueType> x, 
        vec<IndexType, ValueType> b)
{
  int size_b = A.num_rows;
  int size_row_ptr = A.num_rows+1;
  int size_col_idx = A.num_nonzeros;
  gpucg_solve_(A.Ap, &size_row_ptr, A.Aj, &size_col_idx, A.Ax, &size_col_idx, 
               b.val, &size_b, x.val);
}

int main(int argc, char **argv)
{
  cout << "CUDA CG Solver test." << endl << endl;

  cout << "Reading in matrix" << endl;
  csr_matrix<int,double> A = read_csr_matrix<int,double>("input/A", true);
  csr_c_to_fortran(A);
  cout << "Reading in RHS vector" << endl;
  vec<int,double> b = read_vec<int,double>("input/b");
  cout << "Reading in reference solution vector" << endl;
  vec<int,double> x = read_vec<int,double>("input/x");

  cout << "Calling solver" << endl;
  vec<int,double> res = new_vec<int,double>(x.len);
  cg(A, res, b);
  double l2norm = l2_norm(res, x);
  cout << "L2 Norm is " << l2norm << endl;

  if (l2norm > MAX_ERROR)
  {
    printf("L2 Norm is too large - did something go wrong?\n");
    exit(1);
  }

  cout << "Deleting matrix" << endl;
  delete_csr_matrix(A, HOST_MEMORY);
  cout << "Deleting RHS vector" << endl;
  delete_vec(b);
  cout << "Deleting solution vector" << endl;
  delete_vec(res);
  cout << "Deleting reference solution vector" << endl;
  delete_vec(x);

  cout << endl << "Test ran successfully." << endl;
}
