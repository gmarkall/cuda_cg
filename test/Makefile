OBJ=main.o mmio.o gpu_solve.o
TEST=test

CFLAGS=-g -Wall -Werror -Wextra -pedantic
CXXFLAGS=$(CFLAGS)
LDFLAGS=-g
CXX=g++
CC=gcc


%.o:	%.cpp
	$(CXX) $(CXXFLAGS) -c $^

%.o:    %.c
	$(CC) $(CFLAGS) -c $^

all: $(OBJ)
	$(CXX) $(LDFLAGS) $(OBJ) -o $(TEST)

clean:
	rm -f *.o $(TEST)
