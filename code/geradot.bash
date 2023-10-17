gprof MD.exe gmon.out > aux.txt
gprof2dot -n0 -e0 aux.txt -o MDFinal.dot
rm aux.txt