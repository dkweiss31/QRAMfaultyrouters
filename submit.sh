#!/bin/bash
#SBATCH --partition=scavenge
#SBATCH --job-name=qramfaulty
#SBATCH -o out/output-%a.txt -e out/errors-%a.txt
#SBATCH --array=0-99
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.weiss@yale.edu

################################## modify params here
IDX=$SLURM_ARRAY_TASK_ID
n=5

num_eps=10
num_instance_scans=10
eps_idx=$((SLURM_ARRAY_TASK_ID / num_instance_scans))
eps_seq=($(seq 1 1 $num_eps))
eps=$((${eps_seq[$eps_idx]} / 100))

num_instances=1000
rng_seed=$((IDX * 3567483))
############################

module load miniconda
conda activate qramfaultyrouters
python run_faulty_routers.py \
  --idx=$IDX \
  --n=$n \
  --eps=$eps \
  --num_instances=$num_instances \
  --rng_seed=$rng_seed
