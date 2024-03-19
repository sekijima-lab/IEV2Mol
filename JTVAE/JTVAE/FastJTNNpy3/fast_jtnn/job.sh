#!/bin/bash
#$ -cwd
#$ -N make_my-pretrain-dataset_vocab
#$ -l f_node=1
#$ -l h_rt=24:0:0
#$ -V


. /etc/profile.d/modules.sh
module load cuda/11.6.2 cudnn/8.3

source  ~/.bashrc
conda activate vae_cuda11.6
export PYTHONPATH="/gs/hs0/tga-science/ozawa/M2/JTVAE/JTVAE/FastJTNNpy3/:$PYTHONPATH"
export PATH="/gs/hs0/tga-science/ozawa/scripts/:$PATH"

slacknotice python -u ../fast_jtnn/mol_tree.py -i ./../data/pretrain_druglike_canonical.txt -v ./../data/pretrain_druglike_canonical_vocab.txt
