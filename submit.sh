#!/bin/bash
#SBATCH --partition=day
#SBATCH --job-name=qramfaulty
#SBATCH -o out/output-%a.txt -e out/errors-%a.txt
#SBATCH --array=0-10
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.weiss@yale.edu

################################## modify params here
n_start=6
n_end=$((SLURM_ARRAY_TASK_COUNT + n_start))
n_seq=($(seq $n_start 1 $n_end))

IDX=$SLURM_ARRAY_TASK_ID
n=${n_seq[$SLURM_ARRAY_TASK_ID]}
eps=0.01
num_instances=100000
rng_seed=$((IDX * n * 3567483))
############################

module load miniconda
conda activate qramfaultyrouters
python run_faulty_routers.py \
  --idx=$IDX \
  --n=$n \
  --eps=$eps \
  --num_instances=$num_instances \
  --rng_seed=$rng_seed
