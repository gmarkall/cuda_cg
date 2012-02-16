extern "C" 
{
  // Main solver entry point. Solves Ax = b for x.
  // CSR matrix (A) stores the row and column pointer information like Fortran 
  // indices (i.e. begin at 1)
  // findrm_p     : matrix row pointer
  // colm_p       : matrix column pointer
  // matrix_val_p : matrix values
  // size*        : size of each vector
  // b_p          : pointer to RHS vector
  // x_p          : solution (x) is returned here
  void gpucg_solve_(int* findrm_p, int *size_findrm, int* colm_p, int* size_colm, 
                    double* matrix_val_p, int *matrix_val_size, double* b_p, 
                    int* rhs_val_size, double *x_p);
}

