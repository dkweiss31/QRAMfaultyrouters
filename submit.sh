#!/bin/bash
#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --job-name=qramfaulty
#SBATCH -o out/output-%a.txt -e out/errors-%a.txt
#SBATCH --array=0-99
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4G
#SBATCH --time=08:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.weiss@yale.edu

################################## modify params here
n=12
top_three_functioning=1
# num_eps * num_instance_scans should equal number of array jobs
num_eps=10
num_instance_scans=10
eps_idx=$((SLURM_ARRAY_TASK_ID / num_instance_scans))
eps_seq=($(seq 1 1 $num_eps))
eps=$(echo "scale=2; ${eps_seq[$eps_idx]} / 100" | bc)

num_instances=100000
rng_seed=$((SLURM_ARRAY_TASK_ID * n * 3567483 + 23784 * eps_idx + 5627))
############################

module load miniconda
conda activate qramfaultyrouters
python run_faulty_routers.py \
  --idx=$SLURM_ARRAY_TASK_ID \
  --n=$n \
  --eps=$eps \
  --num_instances=$num_instances \
  --rng_seed=$rng_seed \
  --top_three_functioning=$top_three_functioning \
  --num_cpus=$SLURM_CPUS_PER_TASK