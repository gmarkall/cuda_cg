
#ifndef VECTOR_IO_H
#define VECTOR_IO_H

#include <math.h>

template<typename IndexType, typename ValueType> 
struct vec {
  IndexType len;
  ValueType *val;
};


template<class IndexType, class ValueType>
vec<IndexType, ValueType> new_vec(const int len)
{
  vec<IndexType, ValueType> v;
  v.len = len;
  v.val = (ValueType*)malloc(sizeof(ValueType)*len);
  return v;
}

template<class IndexType, class ValueType>
vec<IndexType, ValueType> print_vec(vec<IndexType, ValueType> v)
{
  for (int i=0; i<v.len; ++i)
  {
    printf("%f\n", v.val[i]);
  }
}

template<class IndexType, class ValueType>
ValueType l2_norm(vec<IndexType, ValueType> v1,
                                  vec<IndexType, ValueType> v2)
{
  if (v1.len != v2.len)
  {
    printf("L2 Norm: different vector lengths make no sense.\n");
    exit(1);
  }

  ValueType norm = 0.0;

  for (int i=0; i<v1.len; ++i)
  {
    ValueType d = v1.val[i] - v2.val[i];
    norm = norm + (d*d);
    //printf("V1: %f       V2: %f        Norm: %f\n", v1.val, v2.val, norm);
  }

  return sqrt(norm);
}

template<class IndexType, class ValueType>
vec<IndexType, ValueType> read_vec(const char *filename)
{
  FILE *fd = fopen(filename, "r");
    
  if (fd == NULL){
      printf("Unable to open file %s\n", filename);
      exit(1);
  }

  printf("Reading vector from file (%s)\n",filename);
  
  IndexType count = 0;
  if (fscanf(fd, "%%%%Length: %d", &count) != 1)
  {
    printf("Error reading vector length.\n");
    exit(1);
  }
  
  printf("Vector length: %d\n", count);

  ValueType *val = (ValueType*)malloc(sizeof(ValueType)*count);

  for (int i=0; i<count; ++i)
  {
    if (fscanf(fd, "%lf\n", &val[i]) != 1)
    {
      printf("Error reading value %d\n", i);
      exit(1);
    }
  }

  fclose(fd);

  printf("Finished reading vector.\n");

  vec<IndexType, ValueType> v;
  v.len = count;
  v.val = val;
  return v;
}

template <class IndexType, class ValueType>
void delete_vec(vec<IndexType, ValueType> v)
{
  free(v.val);
}

#endif
