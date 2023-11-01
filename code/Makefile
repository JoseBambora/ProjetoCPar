CC = g++
SRC = src/
CFLAGS = -O3 -Ofast -pg -funroll-loops -fopenmp -mavx

.DEFAULT_GOAL = MD.exe

MD.exe: $(SRC)/MD.cpp
	$(CC) $(CFLAGS) $(SRC)MD.cpp -lm -o MD.exe

clean:
	rm ./MD.exe

run:
	./MD.exe < inputdata.txt
