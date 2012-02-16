
#ifndef VECTOR_IO_H
#define VECTOR_IO_H

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
