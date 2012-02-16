#include "sparse_io.h"
#include "vector_io.h"
#include <iostream>

using namespace std;

int main(int argc, char **argv)
{
  cout << "CUDA CG Solver test." << endl << endl;

  cout << "Reading in matrix" << endl;
  csr_matrix<int,double> A = read_csr_matrix<int,double>("input/A", true);
  cout << "Reading in RHS vector" << endl;
  vec<int,double> b = read_vec<int,double>("input/b");
  cout << "Reading in reference solution vector" << endl;
  vec<int,double> x = read_vec<int,double>("input/x");

  // ...

  cout << "Deleting matrix" << endl;
  delete_csr_matrix(A, HOST_MEMORY);
  cout << "Deleting RHS vector" << endl;
  delete_vec(b);
  cout << "Deleting reference solution vector" << endl;
  delete_vec(x);
}
