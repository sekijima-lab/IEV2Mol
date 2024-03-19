#!/bin/sh
#$ -cwd
#$ -l q_node=1
#$ -l h_rt=04:00:00
#$ -o ./logs/
#$ -e ./logs/

source ~/.bashrc
conda activate iev_vae_env

export PATH="/apps/t3/sles12sp2/uge/latest/bin/lx-amd64/:$PATH"
python -u make_csv.py
