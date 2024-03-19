#!/bin/sh
#$ -cwd
#$ -l q_node=1
#$ -l h_rt=06:00:00
#$ -o ./logs/
#$ -e ./logs/

source ~/.bashrc
conda activate vae_cuda11.6 

export PATH="/apps/t3/sles12sp2/uge/latest/bin/lx-amd64/:$PATH"
python -u make_csv.py

