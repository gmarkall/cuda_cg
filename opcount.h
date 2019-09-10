#ifndef _OPCOUNT_H
#define _OPCOUNT_H

template<typename T>
void opcount_add_mem_read(T* addr);

template<typename T>
void opcount_add_mem_write(T* addr);

void opcount_start_kernel(const char* name);
void opcount_finish_kernel(const char* name);

void opcount_start_loop_iteration();
void opcount_finish_loop_iteration();

#endif // _OPCOUNT_H
