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

protein = 'AKT1'

if os.path.exists(f"../data/{protein}/dmqp1m_no_dot_dup_fps.pt"):
    print("Loading dmqp1m_no_dot_dup_fps.pt")
    with open(f"../data/{protein}/dmqp1m_no_dot_dup_fps.pt", "rb") as f:
        dmqp1m_fps = pickle.load(f)
else:
    print("Creating dmqp1m_no_dot_dup_fps.pt")
    dmqp1m_mols = [Chem.MolFromSmiles(line) for line in open("../data/Druglike_million_canonical_no_dot_dup.smi")]
    dmqp1m_fps = [AllChem.GetMorganFingerprintAsBitVect(m,2) for m in dmqp1m_mols if m is not None]
    pickle.dump(dmqp1m_fps, open("../data/dmqp1m_no_dot_dup_fps.pt", "wb"))

if os.path.exists("../data/drd2active_no_dot_fps.pt"):
    print("Loading drd2active_no_dot_fps.pt")
    with open("../data/drd2active_no_dot_fps.pt", "rb") as f:
        active_fps = pickle.load(f)
else:
    print("Creating fingerprints...")
    active_mols = [Chem.MolFromSmiles(line.split(" ")[0]) for line in open(f"../data/{protein}/{protein}_ligands.smi")]
    active_fps = [AllChem.GetMorganFingerprintAsBitVect(m,2) for m in active_mols if m is not None]
    pickle.dump(active_fps, open(f"../data/{protein}/{protein}_fps.pt", "wb"))

if os.path.exists(f"../data/{protein}/fitted_pca2d.pt"):
    print("Loading fitted_pca2d.pt")
    with open(f"../data/{protein}/fitted_pca2d.pt", "rb") as f:
        pca = pickle.load(f)
else:
    print("Creating fitted_pca2d.pt")
    pca = PCA(n_components=2)
    pca.fit(dmqp1m_fps + active_fps)
    pickle.dump(pca, open(f"../data/{protein}/fitted_pca2d.pt", "wb"))

if os.path.exists(f"../data/{protein}/dmqp1m_pca2d.pt"):
    print("Loading dmqp1m_pca2d.pt")
    with open(f"../data/{protein}/dmqp1m_pca2d.pt", "rb") as f:
        dmqp1m_pca2d = pickle.load(f)
else:
    print("Creating dmqp1m_pca2d.pt")
    dmqp1m_pca2d = pca.transform(dmqp1m_fps)
    pickle.dump(dmqp1m_pca2d, open(f"../data/{protein}/dmqp1m_pca2d.pt", "wb"))

random.seed(10)
dmqp1m_pca2d = dmqp1m_pca2d[random.sample(range(len(dmqp1m_pca2d)), 10000)]

if os.path.exists(f"../data/{protein}/active_pca2d.pt"):
    print("Loading active_pca2d.pt")
    with open(f"../data/{protein}/active_pca2d.pt", "rb") as f:
        active_pca2d = pickle.load(f)
else:
    print("Creating active_pca2d.pt")
    active_pca2d = pca.transform(active_fps)
    pickle.dump(active_pca2d, open(f"../data/{protein}/active_pca2d.pt", "wb"))

chemicalspace = pd.DataFrame({"x": list(dmqp1m_pca2d[:,0]) + list(active_pca2d[:,0]),
                        "y": list(dmqp1m_pca2d[:,1]) + list(active_pca2d[:,1]),
                        "label": ["dmqp1m" for i in range(len(dmqp1m_pca2d))] + ["active" for i in range(len(active_pca2d))]
                        })


test_dataset = torch.load(f"../data/{protein}/{protein}_test.pt")

for i in range(100):
    print("Processing test", i)

    # シード化合物のPCA
    print(" PCA of seed molecule")
    seed_mol = Chem.MolFromSmiles(test_dataset[i][0])
    seed_fp = AllChem.GetMorganFingerprintAsBitVect(seed_mol,2)
    seed_pca2d = pca.transform(np.array(seed_fp).reshape(1,-1))

    # IEV2Mol
    print(" PCA of IEV2Mol")
    df = pd.read_csv(f'results/{protein}/test{i}/raw_csv/iev2mol.csv')
    mols = [Chem.MolFromSmiles(smile) for smile in df['smiles']]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m,2) for m in mols if m is not None]
    pca2d = pca.transform(fps)

    plt.clf()

    fig, ax = plt.subplots()
    
    data_sets = [dmqp1m_pca2d, active_pca2d]
    alpha = 0.5 
    colors = ['Blues', 'Reds']  
    _min, _max = -3, 3 

    for data, color in zip(data_sets, colors):
        x = data[:, 0]
        y = data[:, 1]
        
        values = np.vstack([x, y])
        kernel = gaussian_kde(values)
        
        X, Y = np.mgrid[_min:_max:100j, _min:_max:100j]
        
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kernel(positions).T, X.shape)
        Z = (Z - Z.min()) / (Z.max() - Z.min())

        cs = ax.contourf(X, Y, Z, levels=8, cmap=color, alpha=alpha, extend='min')
        cs.cmap.set_under('white')
        cs.changed()
    ax.scatter(pca2d[:,0], pca2d[:,1], c="blue", s=20, marker="o", label="generated")
    ax.scatter(seed_pca2d[:,0], seed_pca2d[:,1], c="red", s=50, marker="x", label="seed")
    ax.set_xlim(_min, _max)
    ax.set_ylim(_min, _max)
    ax.legend()
    fig.savefig(f'results/{protein}/test{i}/chemicalspace.jpeg')