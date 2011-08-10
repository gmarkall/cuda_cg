OBJ=main.o mmio.o
TEST=test

CXX=g++
CC=gcc

%.o:	%.cpp
	$(CXX) -c $^

%.o:    %.c
	$(CC) -c $^

all: $(OBJ)
	$(CXX) $(OBJ) -o $(TEST)

clean:
	rm -f *.o $(TEST)
