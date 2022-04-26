#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -t 0-00:20                         # Runtime in D-HH:MM format
#SBATCH -p gpu                           # Partition to run in
#SBATCH --mem=40G                         # Memory total in MiB (for all cores)
#SBATCH -o train_simclr_%j.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e train_simclr_%j.err                 # File to which STDERR will be written, including job ID (%j)
                                           # You can change the filenames given with -o and -e to any filenames you'd like
source ~/.bash_profile
module load conda2/4.2.13
module load gcc/6.2.0 cuda/10.2
conda activate test
python train_model.py
