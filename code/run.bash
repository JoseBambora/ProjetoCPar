make clean
make
sudo perf stat -e L1-dcache-load-misses -M cpi ./MD.exe < inputdata.txt