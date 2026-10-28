#!/bin/bash
#
#SBATCH --job-name=VPMXrun
#SBATCH --partition=gpu-a100
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=education-ae-msc-ae

# Load modules
module load 2026 gpu
module load cuda/12.9

# Define runtime output file
RUNTIME_FILE="runtime_${SLURM_JOB_ID}.txt"

# Record start time
echo "Job ID: ${SLURM_JOB_ID}" > "$RUNTIME_FILE"
echo "Start: $(date)" >> "$RUNTIME_FILE"

# Run simulation and measure execution time
srun /usr/bin/time -v -a -o "$RUNTIME_FILE" \
    ./runCase_cuda.sh

# Record exit status
EXIT_CODE=$?

# Record end time
echo "End: $(date)" >> "$RUNTIME_FILE"
echo "Exit code: ${EXIT_CODE}" >> "$RUNTIME_FILE"

exit "$EXIT_CODE"