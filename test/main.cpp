#include "sparse_io.h"
#include <iostream>

using namespace std;

int main(int argc, char **argv)
{
  cout << "CUDA CG Solver test." << endl;

  csr_matrix<int,double> A = read_csr_matrix<int,double>("inputs/A", true);
}
