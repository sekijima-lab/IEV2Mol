"""
density graphをプロットする
10のテストデータ個別のプロットと，全体のプロットを行う．
対象モデル：
    NO-IEVLOSS100・JT-VAE・IFP-RNN・ランダムCHEMBL33
メトリクス：
    IEVCos類似度・Tanimoto類似度・Dockingscoreの分布
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

sys.path.append("../data")  
from Smiles_Vector_Dataset import Smiles_Vector_Dataset

ievcos_model3 = []
ievcos_jt_vae = []
ievcos_ifp_rnn = []
ievcos_random_chembl33 = []

tanimoto_model3 = []
tanimoto_jt_vae = []
tanimoto_ifp_rnn = []
tanimoto_random_chembl33 = []

dscore_model3 = []
dscore_jt_vae = []
dscore_ifp_rnn = []
dscore_random_chembl33 = []

test_dataset = torch.load("../data/drd2_test_dataset_no_dot.pt")

for i in range(10):
    seed_mol = Chem.MolFromSmiles(test_dataset[i][0])
    seed_fp = AllChem.GetMorganFingerprintAsBitVect(seed_mol,2)

    # IEV2Mol
    df_model3 = pd.read_csv(f'results/test{i}/raw_csv/iev2mol.csv')
    ievcos_model3 = ievcos_model3 + list(df_model3['ievcos'])
    
    mols = [Chem.MolFromSmiles(smile) for smile in df_model3['smiles']]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m,2) for m in mols if m is not None]
    tmp_tanimoto_model3 = DataStructs.BulkTanimotoSimilarity(seed_fp,fps)
    tanimoto_model3 = tanimoto_model3 + list(tmp_tanimoto_model3)

    dscore_model3 = dscore_model3 + list(df_model3['dscore'])

    # JT-VAE
    df_jtvae = pd.read_csv(f'results/test{i}/raw_csv/jt-vae.csv')
    ievcos_jt_vae = ievcos_jt_vae + list(df_jtvae['ievcos'])

    mols = [Chem.MolFromSmiles(smile) for smile in df_jtvae['smiles']]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m,2) for m in mols if m is not None]
    tmp_tanimoto_jtvae = DataStructs.BulkTanimotoSimilarity(seed_fp,fps)
    tanimoto_jt_vae = tanimoto_jt_vae + list(tmp_tanimoto_jtvae)

    dscore_jt_vae = dscore_jt_vae + list(df_jtvae['dscore'])

    # IFP-RNN
    df_ifprnn = pd.read_csv(f'results/test{i}/raw_csv/ifp-rnn_0.csv')
    ievcos_ifp_rnn = ievcos_ifp_rnn + list(df_ifprnn['ievcos'])

    mols = [Chem.MolFromSmiles(smile) for smile in df_ifprnn['smiles'] if type(smile) == str]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m,2) for m in mols if m is not None]
    tmp_tanimoto_ifprnn = DataStructs.BulkTanimotoSimilarity(seed_fp,fps)
    tanimoto_ifp_rnn = tanimoto_ifp_rnn + list(tmp_tanimoto_ifprnn)

    dscore_ifp_rnn = dscore_ifp_rnn + list(df_ifprnn['dscore'])

    # ランダムCHEMBL33
    df_random = pd.read_csv(f'results/test{i}/raw_csv/random_chembl33.csv')
    ievcos_random_chembl33 = ievcos_random_chembl33 + list(df_random['ievcos'])

    mols = [Chem.MolFromSmiles(smile) for smile in df_random['smiles']]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m,2) for m in mols if m is not None]
    seed_fp = AllChem.GetMorganFingerprintAsBitVect(seed_mol,2)
    tmp_tanimoto_random = DataStructs.BulkTanimotoSimilarity(seed_fp,fps)
    tanimoto_random_chembl33 = tanimoto_random_chembl33 + list(tmp_tanimoto_random)

    dscore_random_chembl33 = dscore_random_chembl33 + list(df_random['dscore'])

    # 各テストデータごとの分布のプロット
    # 1. IEVCos類似度の分布をプロット
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)

    sns.kdeplot(df_model3['ievcos'], ax=ax, label='Our model')
    sns.kdeplot(df_jtvae['ievcos'], ax=ax, label='JT-VAE')
    sns.kdeplot(df_ifprnn['ievcos'], ax=ax, label='IFP-RNN')
    sns.kdeplot(df_random['ievcos'], ax=ax, label='Random ChEMBL')
    ax.legend(fontsize=18)
    ax.tick_params(labelsize=18)
    ax.set_xlabel("IEV cosine similarity", fontsize=20, labelpad=0)
    ax.set_ylabel("Density", fontsize=20, labelpad=0)
    fig.savefig(f'results/test{i}/density_ievcos.jpeg')

    # 2. Tanimoto類似度の分布をプロット
    plt.clf()
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)

    sns.kdeplot(tmp_tanimoto_model3, ax=ax, label='Our model')
    sns.kdeplot(tmp_tanimoto_jtvae, ax=ax, label='JT-VAE')
    sns.kdeplot(tmp_tanimoto_ifprnn, ax=ax, label='IFP-RNN')
    sns.kdeplot(tmp_tanimoto_random, ax=ax, label='Random ChEMBL')
    ax.legend(fontsize=18)
    ax.tick_params(labelsize=18)
    ax.set_xlabel("Tanimoto", fontsize=20, labelpad=0)
    ax.set_ylabel("Density", fontsize=20, labelpad=0)
    fig.savefig(f'results/test{i}/density_tanimoto.jpeg')

    # 3. ドッキングスコアの分布をプロット
    plt.clf()
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)

    sns.kdeplot(df_model3['dscore'], ax=ax, label='Our model')
    sns.kdeplot(df_jtvae['dscore'], ax=ax, label='JT-VAE')
    sns.kdeplot(df_ifprnn['dscore'], ax=ax, label='IFP-RNN')
    sns.kdeplot(df_random['dscore'], ax=ax, label='Random ChEMBL')
    ax.legend(fontsize=18)
    ax.tick_params(labelsize=18)
    ax.set_xlabel("Docking score", fontsize=20, labelpad=0)
    ax.set_ylabel("Density", fontsize=20, labelpad=0)
    fig.savefig(f'results/test{i}/density_dscore.jpeg')

# 10のテストデータ全ての分布のプロット
# 1. IEVCos類似度の分布をプロット
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)

sns.kdeplot(ievcos_model3, ax=ax, label='Our model')
sns.kdeplot(ievcos_jt_vae, ax=ax, label='JT-VAE')
sns.kdeplot(ievcos_ifp_rnn, ax=ax, label='IFP-RNN')
sns.kdeplot(ievcos_random_chembl33, ax=ax, label='Random ChEMBL')
ax.legend(fontsize=18)
ax.tick_params(labelsize=18)
ax.set_xlabel("IEV cosine similarity", fontsize=20, labelpad=0)
ax.set_ylabel("Density", fontsize=20, labelpad=0)
fig.savefig(f'results/density_ievcos.jpeg')

# 2. Tanimoto類似度の分布をプロット
plt.clf()
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)

sns.kdeplot(tanimoto_model3, ax=ax, label='Our model')
sns.kdeplot(tanimoto_jt_vae, ax=ax, label='JT-VAE')
sns.kdeplot(tanimoto_ifp_rnn, ax=ax, label='IFP-RNN')
sns.kdeplot(tanimoto_random_chembl33, ax=ax, label='Random ChEMBL')
ax.legend(fontsize=18)
ax.tick_params(labelsize=18)
ax.set_xlabel("Tanimoto", fontsize=20, labelpad=0)
ax.set_ylabel("Density", fontsize=20, labelpad=0)
fig.savefig(f'results/density_tanimoto.jpeg')

# 3. ドッキングスコアの分布をプロット
plt.clf()
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)

sns.kdeplot(dscore_model3, ax=ax, label='Our model')
sns.kdeplot(dscore_jt_vae, ax=ax, label='JT-VAE')
sns.kdeplot(dscore_ifp_rnn, ax=ax, label='IFP-RNN')
sns.kdeplot(dscore_random_chembl33, ax=ax, label='Random ChEMBL')
ax.legend(fontsize=18)
ax.tick_params(labelsize=18)
ax.set_xlabel("Docking score", fontsize=20, labelpad=0)
ax.set_ylabel("Density", fontsize=20, labelpad=0)
fig.savefig(f'results/density_dscore.jpeg')
