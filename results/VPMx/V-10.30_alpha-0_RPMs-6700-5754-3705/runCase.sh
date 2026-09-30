#!/bin/bash
#
#SBATCH --job-name=VPMXtest
#SBATCH --partition=gpu-a100
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=education-ae-msc-ae

module load 2026 gpu
module load cuda/12.1

srun ./runCase_cuda.sh
