#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=32G
#SBATCH --time=0-10:00:00
#SBATCH --mail-user=wyyadd@gmail.com
#SBATCH --mail-type=ALL

cd $project/kGPT
module purge
module load python/3.12.4
source ../agents/bin/activate

pip3 install -r requirements.txt

python3 generate_dataset.py \
--root="$project/kGPT/data" \
--train_processed_dir="$SLURM_TMPDIR/processed"

cd $SLURM_TMPDIR
tar -I pigz -cf $project/kGPT/data/training/processed.tar.gz processed

