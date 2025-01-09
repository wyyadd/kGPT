#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=256G
#SBATCH --time=0-02:00:00
#SBATCH --mail-user=wyyadd@gmail.com
#SBATCH --mail-type=ALL

cd $project/kGPT
module purge module load python/3.12.4
source ../agents/bin/activate

python3 generate_submission.py \
--root="./data/pkl_files/scenarios" \
--submission_dir="./data/pkl_files/sub"
