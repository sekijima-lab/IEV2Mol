#!/bin/sh
#$ -cwd
#$ -l q_node=1
#$ -l h_rt=00:40:00
#$ -o ./logs_out/
#$ -e ./logs_out/

source ~/.bashrc

conda activate

export SCHROD_LICENSE_FILE="///path-to-licence-file///"
export SCHRODINGER="///path-to-schrodinger///"
export SIEVE="../../../SIEVE-Score"


rm prepred_out_smiles.maegz
rm out_smiles_HTVS_pv.maegz
rm out_smiles_HTVS_pv.interaction
rm out_smiles_HTVS_pv_max.interaction
$SCHRODINGER/ligprep -ismi out_smiles.smi -omae prepred_out_smiles.maegz -WAIT -NJOBS 2 -TMPDIR $TMPDIR
$SCHRODINGER/glide out_smiles_HTVS.in -OVERWRITE -NJOBS 2 -HOST "localhost:2" -TMPDIR $TMPDIR -ATTACHED -WAIT
$SCHRODINGER/run python3 $SIEVE/SIEVE-Score.py -m interaction -i ./out_smiles_HTVS_pv.maegz -l sieve-score.log
python3 rest_max.py out_smiles_HTVS_pv.interaction 

