CC1 = gcc
CC2 = g++
SRC = src/
CFLAGS = -O3 -Ofast -funroll-loops -mavx
.DEFAULT_GOAL = all
all: MDseq.exe MDpar.exe

MDseq.exe: $(SRC)/MDseq.cpp
	module load gcc/11.2.0;
	$(CC1) $(CFLAGS) $(SRC)MDseq.cpp -lm -o MDseq.exe
MDpar.exe: $(SRC)/MDpar.cpp
	module load gcc/11.2.0;
	$(CC2) $(CFLAGS) $(SRC)MDpar.cpp -lm -fopenmp -o MDpar.exe

clean:
	rm ./MD*.exe

runseq:
	./MDseq.exe < inputdata.txt
runpar:
	./MDpar.exe < inputdata.txt