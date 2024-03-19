#!/bin/sh

# DRD2のtrain_smilesについてmoses-processedを作る
nohup python -u preprocess.py --train ../../../../MAIN/data/drd2_train_smiles_no_dot.smi --split 1 --jobs 40 --output ./drd2_train_smiles_no_dot_moses-processed >drd2_train_smiles_no_dot_moses-processed.log 2>drd2_train_smiles_no_dot_moses-processed.err &


# DRD2ののtest_smilesについてmoses-processedを作る
nohup python -u preprocess.py --train ../../../../MAIN/data/drd2_test_smiles_no_dot.smi --split 1 --jobs 40 --output ./drd2_test_smiles_no_dot_moses-processed >drd2_test_smiles_no_dot_moses-processed.log 2>drd2_test_smiles_no_dot_moses-processed.err &
