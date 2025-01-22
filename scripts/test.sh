#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=0-10:00:00
#SBATCH --mail-user=wyyadd@gmail.com
#SBATCH --mail-type=ALL

cd $project/kGPT
module purge
module load python/3.12.4
source ../agents/bin/activate

srun python3 train_k_gpt.py \
--mode="val" \
--root="$project/kGPT/data" \
--train_processed_dir="$SLURM_TMPDIR/processed" \
--num_workers=4 \
--accelerator="auto" \
--devices=-1 \
--num_nodes=$SLURM_NNODES \
--train_batch_size=4 \
--val_batch_size=4 \
--test_batch_size=8 \
--submission_dir="./data/pkl_files" \
--simulation_times=32 \
--ckpt_path=""
