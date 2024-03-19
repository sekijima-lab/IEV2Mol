"""
化学空間をPCAでプロットする
10のテストデータ全てを使う
対象データ：
    シード化合物・NO-IEVLOSS100が各シードに対して生成した分子・dmqp1m_no_dot_dup・drd2active_no_dot
"""

import os
import pickle
import random
import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

sys.path.append("../data")  
from Smiles_Vector_Dataset import Smiles_Vector_Dataset

if os.path.exists("../data/dmqp1m_no_dot_dup_fps.pt"):
    print("Loading dmqp1m_no_dot_dup_fps.pt")
    with open("../data/dmqp1m_no_dot_dup_fps.pt", "rb") as f:
        dmqp1m_fps = pickle.load(f)
else:
    print("Creating dmqp1m_no_dot_dup_fps.pt")
    dmqp1m_mols = [Chem.MolFromSmiles(line) for line in open("../data/Druglike_million_canonical_no_dot_dup.smi")]
    dmqp1m_fps = [AllChem.GetMorganFingerprintAsBitVect(m,2) for m in dmqp1m_mols if m is not None]
    pickle.dump(dmqp1m_fps, open("../data/dmqp1m_no_dot_dup_fps.pt", "wb"))

if os.path.exists("../data/drd2active_no_dot_fps.pt"):
    print("Loading drd2active_no_dot_fps.pt")
    with open("../data/drd2active_no_dot_fps.pt", "rb") as f:
        drd2active_fps = pickle.load(f)
else:
    print("Creating drd2active_no_dot_fps.pt")
    drd2active_mols = [Chem.MolFromSmiles(line.split(" ")[0]) for line in open("../data/unique_ChEMBL_DRD2_Ki_and_IC50_no_dot.smi")]
    drd2active_fps = [AllChem.GetMorganFingerprintAsBitVect(m,2) for m in drd2active_mols if m is not None]
    pickle.dump(drd2active_fps, open("../data/drd2active_no_dot_fps.pt", "wb"))

if os.path.exists("../data/fitted_pca2d.pt"):
    print("Loading fitted_pca2d.pt")
    with open("../data/fitted_pca2d.pt", "rb") as f:
        pca = pickle.load(f)
else:
    print("Creating fitted_pca2d.pt")
    pca = PCA(n_components=2)
    pca.fit(dmqp1m_fps + drd2active_fps)
    pickle.dump(pca, open("../data/fitted_pca2d.pt", "wb"))

if os.path.exists("../data/dmqp1m_pca2d.pt"):
    print("Loading dmqp1m_pca2d.pt")
    with open("../data/dmqp1m_pca2d.pt", "rb") as f:
        dmqp1m_pca2d = pickle.load(f)
else:
    print("Creating dmqp1m_pca2d.pt")
    dmqp1m_pca2d = pca.transform(dmqp1m_fps)
    pickle.dump(dmqp1m_pca2d, open("../data/dmqp1m_pca2d.pt", "wb"))

# dmqp1mは大きすぎるので10000個に絞る
random.seed(10) # ランダムシードの固定
dmqp1m_pca2d = dmqp1m_pca2d[random.sample(range(len(dmqp1m_pca2d)), 10000)]

if os.path.exists("../data/drd2active_pca2d.pt"):
    print("Loading drd2active_pca2d.pt")
    with open("../data/drd2active_pca2d.pt", "rb") as f:
        drd2active_pca2d = pickle.load(f)
else:
    print("Creating drd2active_pca2d.pt")
    drd2active_pca2d = pca.transform(drd2active_fps)
    pickle.dump(drd2active_pca2d, open("../data/drd2active_pca2d.pt", "wb"))

chemicalspace = pd.DataFrame({"x": list(dmqp1m_pca2d[:,0]) + list(drd2active_pca2d[:,0]),
                        "y": list(dmqp1m_pca2d[:,1]) + list(drd2active_pca2d[:,1]),
                        "label": ["dmqp1m" for i in range(len(dmqp1m_pca2d))] + ["drd2active" for i in range(len(drd2active_pca2d))]
                        })


test_dataset = torch.load("../data/drd2_test_dataset_no_dot.pt")

for i in range(10):
    print("Processing test", i)

    # シード化合物のPCA
    print(" PCA of seed molecule")
    seed_mol = Chem.MolFromSmiles(test_dataset[i][0])
    seed_fp = AllChem.GetMorganFingerprintAsBitVect(seed_mol,2)
    seed_pca2d = pca.transform(np.array(seed_fp).reshape(1,-1))

    # IEV2Mol
    print(" PCA of model3")
    df = pd.read_csv(f'results/test{i}/raw_csv/iev2mol.csv')
    mols = [Chem.MolFromSmiles(smile) for smile in df['smiles']]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m,2) for m in mols if m is not None]
    pca2d = pca.transform(fps)

    plt.clf()

    fig, ax = plt.subplots()
    
    data_sets = [dmqp1m_pca2d, drd2active_pca2d]
    alpha = 0.5  # 塗りつぶしの透明度
    colors = ['Blues', 'Reds']  # 各データセットのカラーマップ
    _min, _max = -3, 3  # x, yの範囲

    # 1. dmqp1mとdrd2Activeの分布をプロット
    for data, color in zip(data_sets, colors):
        x = data[:, 0]
        y = data[:, 1]
        
        # カーネル密度推定
        values = np.vstack([x, y])
        kernel = gaussian_kde(values)
        
        # 格子点を生成
        X, Y = np.mgrid[_min:_max:100j, _min:_max:100j]
        
        # 格子点上での密度を計算
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kernel(positions).T, X.shape)
        Z = (Z - Z.min()) / (Z.max() - Z.min())

        # 等高線プロット
        cs = ax.contourf(X, Y, Z, levels=8, cmap=color, alpha=alpha, extend='min')
        cs.cmap.set_under('white')
        cs.changed()

    # 2. モデル3をプロット
    ax.scatter(pca2d[:,0], pca2d[:,1], c="blue", s=20, marker="o", label="generated")

    # 3. シード化合物をプロット
    ax.scatter(seed_pca2d[:,0], seed_pca2d[:,1], c="red", s=50, marker="x", label="seed")

    ax.set_xlim(_min, _max)
    ax.set_ylim(_min, _max)
    ax.legend()
    
    plt.title("Kernel density of train dataset space and points of generated molecules")

    fig.savefig(f'results/test{i}/chemicalspace.jpeg')