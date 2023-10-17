module load gcc/9.3.0
make clean
make
srun --partition=cpar perf stat -e L1-dcache-load-misses -M cpi ./MD.exe < inputdata.txt