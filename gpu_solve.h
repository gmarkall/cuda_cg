// Main solver entry point. Solves Ax = b for x.
// CSR matrix (A) stores the row and column pointer information like Fortran
// indices (i.e. begin at 1)
// findrm_p     : matrix row pointer
// colm_p       : matrix column pointer
// matrix_val_p : matrix values
// size*        : size of each vector
// b_p          : pointer to RHS vector
// x_p          : solution (x) is returned here
void pcg_solve(int n, int *row_ptr, int *col_idx, double *val, double *x, double *b);
