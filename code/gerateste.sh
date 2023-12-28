#!/bin/bash
#SBATCH --ntasks=40
#SBATCH --time=00:10:00
#SBATCH --partition=cpar
#SBATCH --exclusive
#SBATCH --constraint=k20

# Output file name
output_file="timing_results.csv"

# Clear existing CSV file and add headers
echo "seq;par;cuda" > "$output_file"

# Loop to run and measure each version 10 times
for i in {1..20}; do
    echo "Run $i:"
    seq_time=$( { time make runseq > lixo; } 2>&1 | grep real | awk '{print $2}' )
    par_time=$( { time make runpar > lixo; } 2>&1 | grep real | awk '{print $2}' )
    cuda_time=$( { time make run > lixo; }   2>&1 | grep real | awk '{print $2}' )
    echo "$seq_time;$par_time;$cuda_time" >> "$output_file"
done

# Display a message indicating the completion of the script
echo "Timing results have been saved to $output_file"
