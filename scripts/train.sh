#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=2-12:00:00
#SBATCH --mail-user=wyyadd@gmail.com
#SBATCH --mail-type=ALL

cd $project/kGPT
module purge
module load python/3.12.4
source ../agents/bin/activate

pip3 install -r requirements.txt

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
tar -I pigz -xf $project/kGPT/data/training/processed.tar.gz -C $SLURM_TMPDIR

srun python3 train_k_gpt.py \
--root="$project/kGPT/data" \
--train_processed_dir="$SLURM_TMPDIR/processed" \
--num_workers=4 \
--accelerator="auto" \
--devices=-1 \
--num_nodes=$SLURM_NNODES \
--train_batch_size=5 \
--val_batch_size=8 \
--test_batch_size=8 \
--lr=1e-3 \
--grad_batch_size=1
