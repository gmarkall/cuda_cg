#include <iostream>
#include "opcount.h"

using std::cerr;
using std::endl;
using std::hex;
using std::dec;

static int loop_nest_level = 0;

template<typename T>
void opcount_add_mem_read(T* addr)
{
  cerr << "Read of size " << dec << sizeof(T) << " at 0x" << hex << addr << endl;
}

template<typename T>
void opcount_add_mem_write(T* addr)
{
  cerr << "Write of size " << dec << sizeof(T) << " at 0x" << hex << addr << endl;
}

template void opcount_add_mem_read<double>(double* addr);
template void opcount_add_mem_write<double>(double* addr);
template void opcount_add_mem_read<int>(int* addr);
template void opcount_add_mem_write<int>(int* addr);

void opcount_start_kernel(const char* name)
{
  cerr << "Entering kernel " << name << endl;
}

void opcount_finish_kernel(const char* name)
{
  cerr << "Leaving kernel " << name << endl;
}

void opcount_start_loop_iteration()
{
  loop_nest_level += 1;
  cerr << "Entering loop nest level " << dec << loop_nest_level << endl;
}

void opcount_finish_loop_iteration()
{
  cerr << "Leaving loop nest level " << dec << loop_nest_level << endl;
  loop_nest_level -= 1;
}


