#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=0-15:00:00
#SBATCH --mail-user=wyyadd@gmail.com
#SBATCH --mail-type=ALL

cd $project/kGPT
module purge module load python/3.11
source ~/agents/bin/activate

pip3 install -r requirements.txt

srun python3 train_k_gpt.py \
--mode="val" \
--root="$project/kGPT/data" \
--train_processed_dir="$SLURM_TMPDIR/processed" \
--num_workers=$SLURM_CPUS_PER_TASK \
--accelerator="auto" \
--devices=-1 \
--num_nodes=$SLURM_NNODES \
--train_batch_size=8 \
--val_batch_size=8 \
--test_batch_size=16 \
--submission_dir="pkl_files" \
--simulation_times=32 \
--ckpt_path=""
