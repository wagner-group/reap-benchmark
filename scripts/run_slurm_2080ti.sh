#!/bin/bash
#SBATCH --job-name=apb
#SBATCH --account=fc_wagner
#SBATCH --partition=savio3_gpu
# Number of nodes:
#SBATCH --nodes=1
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH --cpus-per-task=2
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --time=24:00:00
#SBATCH --output slurm-%j-10x20-all-signs.out
## Command(s) to run:
source /global/home/users/$USER/.bash_profile
module purge
module load python
source activate /global/scratch/users/$USER/apb

sh scripts/test_detectron.sh
