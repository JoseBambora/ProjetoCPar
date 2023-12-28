#!/bin/bash
#SBATCH --ntasks=40
#SBATCH --time=00:10:00
#SBATCH --partition=cpar
#SBATCH --exclusive
#SBATCH --constraint=k20

echo "Versão Sequencial:"
time `make runseq > lixo`
echo "Versão Paralela (Open-MP):"
time `make runpar > lixo`
echo "Versão Paralela (CUDA):"
time `make run > lixo`
rm lixo