import os
import pandas as pd
import numpy as np
from rdkit import Chem
import matplotlib.pyplot as plt
from rdkit.Chem import Draw

protein = 'DRD2'

for i in range(100):
    print("test", i)
    dir = f"results/{protein}/test{i}/raw_csv"

    for file in sorted(os.listdir(dir)):
        if file.endswith(".csv"):
            model_name = file.split(".")[0]

            df = pd.read_csv(f"{dir}/{file}")

            valid_smiles = [smiles for smiles in df["smiles"] if smiles is not np.nan and Chem.MolFromSmiles(smiles)is not None]

            if not os.path.exists(f"results/{protein}/test{i}/mol_from_rawcsv"):
                os.mkdir(f"results/{protein}/test{i}/mol_from_rawcsv")

            mols = [Chem.MolFromSmiles(valid_smile) for valid_smile in valid_smiles]
            img = Draw.MolsToGridImage(
                mols, molsPerRow=8, subImgSize=(400, 400)
                )
            img.save(f"results/{protein}/test{i}/mol_from_rawcsv/{model_name}.jpg")

            valid_iev_num = len(df.dropna(subset=["ievcos"]))
            
            docking_score_list = df["dscore"].tolist()
            cos_sim_list = df["ievcos"].tolist()
