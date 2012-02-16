OBJ=main.o mmio.o gpu_solve.o
TEST=test

CFLAGS=-g
CXXFLAGS=$(CFLAGS)
NVCCFLAGS=-arch sm_20
LDFLAGS=-lcudart -L/usr/local/cuda/lib64 -g
CXX=g++
CC=gcc
NVCC=nvcc


%.o:	%.cpp
	$(CXX) $(CXXFLAGS) -c $^

%.o:    %.c
	$(CC) $(CFLAGS) -c $^

%.o:    %.cu
	$(NVCC) $(NVCCFLAGS) -c $^

all: $(OBJ)
	$(CXX) $(LDFLAGS) $(OBJ) -o $(TEST)

clean:
	rm -f *.o $(TEST)
