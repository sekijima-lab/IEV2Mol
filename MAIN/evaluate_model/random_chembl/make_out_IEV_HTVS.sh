export SCHROD_LICENSE_FILE="/opt/schrodinger/licenses/80_client_2024-01-21_192.168.0.32.lic"
# export SCHROD_LICENSE_FILE="/opt/schrodinger2018-3/licenses/license_key_t3.txt"
export SIEVE="/home/m.ozawa/M2/SIEVE-Score"
export SCHRODINGER="/opt/schrodinger2024-1"
export TMPDIR="/home/m.ozawa/M2/no_dot/evaluate_model/random_chembl/tmp"


rm prepred_out_smiles.maegz 
rm out_smiles_HTVS_pv.maegz
rm out_smiles_HTVS_pv.interaction
rm out_smiles_HTVS_pv_max.interaction
$SCHRODINGER/ligprep -ismi out_smiles.smi -omae prepred_out_smiles.maegz -WAIT -NJOBS 10 -TMPDIR $TMPDIR &
$SCHRODINGER/glide out_smiles_HTVS.in -OVERWRITE -NJOBS 10 -HOST "localhost:10" -TMPDIR $TMPDIR -ATTACHED -WAIT &
$SCHRODINGER/run python3 $SIEVE/SIEVE-Score.py -m interaction -i ./out_smiles_HTVS_pv.maegz -l sieve-score.log & 
python3 rest_max.py out_smiles_HTVS_pv.interaction &

