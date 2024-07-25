import os
import sys
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw

sys.path.append("../data")  
from Smiles_Vector_Dataset import Smiles_Vector_Dataset

tanimoto_threshold = 0.5
ievcos_threshold = 0.7

test_len = 100
model_dict = {}
model_dict["ifp-rnn_mean_len"] = {}

protein = 'AA2AR'

test_dataset = torch.load(f"../data/{protein}/{protein}_test.pt")

for i in range(test_len):
    print("test", i)
    dir = f"results/{protein}/test{i}/raw_csv"

    test_smiles = test_dataset[i][0]
    test_iev = test_dataset[i][1]
    
    num_ifprnn_pose = 0

    for file in sorted(os.listdir(dir)):
        if file.endswith(".csv"):
            model_name = file.split(".")[0]
            filtered_mol_list = []

            if "ifp-rnn" in model_name:
                model_name = "ifp-rnn"
                num_ifprnn_pose += 1

            if model_name not in model_dict.keys():
                model_dict[model_name] = {}
        
            if model_name == "ifp-rnn":
                if not i in model_dict[model_name].keys():
                    model_dict[model_name][i] = {}
            
            else:
                model_dict[model_name][i] = []
        
            df = pd.read_csv(f"{dir}/{file}")
            df.dropna()

            iev_valid_smiles_list = []
            ievcos_list = []
            dscore_list = []

            for _, row in df.iterrows():
                try:
                    if row["smiles"] is not np.nan and Chem.MolFromSmiles(row["smiles"]) is not None:
                        iev_valid_smiles_list.append(row["smiles"])
                        ievcos_list.append(row["ievcos"])
                        dscore_list.append(row["dscore"])
                except:
                    pass

            mols = [Chem.MolFromSmiles(smile) for smile in iev_valid_smiles_list]
            fps = [AllChem.GetMorganFingerprintAsBitVect(m,2) for m in mols]

            test_smiles_mol = Chem.MolFromSmiles(test_smiles)
            test_smiles_fp = AllChem.GetMorganFingerprintAsBitVect(test_smiles_mol,2)

            tanimoto_list = DataStructs.BulkTanimotoSimilarity(test_smiles_fp,fps)

            for smiles, tanimoto, iev_cos, dscore in zip(iev_valid_smiles_list, tanimoto_list, ievcos_list, dscore_list):
                if tanimoto <= tanimoto_threshold and iev_cos >= ievcos_threshold:
                    filtered_mol_list.append([smiles, tanimoto, iev_cos, dscore])
                    
            if model_name == "ifp-rnn":
                model_dict[model_name][i][num_ifprnn_pose-1] = filtered_mol_list
            else:
                model_dict[model_name][i] = filtered_mol_list
        if "ifp-rnn" in model_dict.keys() and i in model_dict["ifp-rnn"].keys():
            model_dict["ifp-rnn_mean_len"][i] = sum([len(value) for value in model_dict["ifp-rnn"][i].values()])/len(model_dict["ifp-rnn"][i].keys())
        
                
# テストごとの平均をとる
for model in model_dict.keys():
    if model == "ifp-rnn":
        model_dict[model]["MEAN"] = sum([i for i in model_dict["ifp-rnn_mean_len"].values()])/test_len
    elif model == "ifp-rnn_mean_len":
        continue
    else:
        model_dict[model]["MEAN"] = sum([len(value) for value in model_dict[model].values()])/test_len

models = ["iev2mol", "jt-vae", "ifp-rnn", "random_chembl33"]
japanese_models = ["IEV2Mol", "JT-VAE", "IFP-RNN", "Random_ChEMBL"]
 
for test_index in range(100):
    print(f"test{test_index}")
    for model, japanese_model in zip(models, japanese_models):
        if model == "ifp-rnn":
            try:
                print(f"{japanese_model}, {model_dict['ifp-rnn_mean_len'][test_index]}")
            except:
                pass
        else:
            print(f"{japanese_model}, {len(model_dict[model][test_index])}")
    print()
print()

print('MEAN')
for model, japanese_model in zip(models, japanese_models):
    print(f"{japanese_model}, {model_dict[model]['MEAN']}")
with open(f'./results/{protein}/tanimoto{int(tanimoto_threshold*10)}e-1_ievcos{int(ievcos_threshold*10)}e-1.csv', 'w') as f:
    f.write("model, num\n")
    for model, japanese_model in zip(models, japanese_models):
        f.write(f"{japanese_model}, {model_dict[model]['MEAN']}\n")
    f.close()

for i in range(100):
    if not os.path.exists(f"results/{protein}/test{i}/tanimoto{int(tanimoto_threshold*10)}e-1_ievcos{int(ievcos_threshold*10)}e-1"):
        os.mkdir(f"results/{protein}/test{i}/tanimoto{int(tanimoto_threshold*10)}e-1_ievcos{int(ievcos_threshold*10)}e-1")
    
    dock_smiles_list = []
    options = Draw.MolDrawOptions()
    options.legendFontSize = 55
    options.legendFraction = 0.4


    for model in models:
        if model == "ifp-rnn":
            for j in model_dict['ifp-rnn'][i].keys():
                if len(model_dict['ifp-rnn'][i][j]) == 0:
                    print("None")
                    continue
                smiles = [model_dict['ifp-rnn'][i][j][k][0] for k in range(len(model_dict['ifp-rnn'][i][j]))]
                tanimoto_list = [float(model_dict['ifp-rnn'][i][j][k][1]) for k in range(len(model_dict['ifp-rnn'][i][j]))]
                ievcos_list = [float(model_dict['ifp-rnn'][i][j][k][2]) for k in range(len(model_dict['ifp-rnn'][i][j]))]

                ievcos_list, tanimoto_list, smiles = zip(*sorted(zip(ievcos_list, tanimoto_list, smiles), reverse=True)) # IEV類似度が高い順に並び替え


                mols = [Chem.MolFromSmiles(smiles[0])]
                img = Draw.MolsToGridImage(
                    mols, molsPerRow=1, subImgSize=(400, 400), drawOptions=options, 
                    legends=[f"Tanimoto:{float(Decimal(str(tanimoto_list[0])).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))},  IEVCos:{float(Decimal(str(ievcos_list[0])).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))}"]
                )
                img.save(f"results/{protein}/test{i}/tanimoto{int(tanimoto_threshold*10)}e-1_ievcos{int(ievcos_threshold*10)}e-1/{model}_{j}.jpg")
                
                dock_smiles_list.append([smiles[0], f"{model}_{j}"])

                mols = [Chem.MolFromSmiles(smile) for smile in smiles]
                img = Draw.MolsToGridImage(
                    mols, molsPerRow=5, subImgSize=(400, 400), drawOptions=options, 
                    legends=[f"Tanimoto:{float(Decimal(str(tanimoto)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))},  IEVCos:{float(Decimal(str(ievcos)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))}" for smiles, tanimoto, ievcos in zip(smiles, tanimoto_list, ievcos_list)]
                )

                img.save(f"results/{protein}/test{i}/tanimoto{int(tanimoto_threshold*10)}e-1_ievcos{int(ievcos_threshold*10)}e-1/{model}_{j}_all.jpg")
        else:
            if len(model_dict[model][i]) == 0:
                print("None")
                continue
            smiles = [model_dict[model][i][j][0] for j in range(len(model_dict[model][i]))]
            tanimoto_list = [float(model_dict[model][i][j][1]) for j in range(len(model_dict[model][i]))]
            ievcos_list = [float(model_dict[model][i][j][2]) for j in range(len(model_dict[model][i]))]

            ievcos_list, tanimoto_list, smiles = zip(*sorted(zip(ievcos_list, tanimoto_list, smiles), reverse=True)) # IEV類似度が高い順に並び替え

            mols = [Chem.MolFromSmiles(smiles[0])]
            img = Draw.MolsToGridImage(
                mols, molsPerRow=1, subImgSize=(400, 400), drawOptions=options, 
                legends=[f"Tanimoto:{float(Decimal(str(tanimoto_list[0])).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))},  IEVCos:{float(Decimal(str(ievcos_list[0])).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))}"]
            )
            img.save(f"results/{protein}/test{i}/tanimoto{int(tanimoto_threshold*10)}e-1_ievcos{int(ievcos_threshold*10)}e-1/{model}.jpg")
            dock_smiles_list.append([smiles[0], f"{model}"])

            mols = [Chem.MolFromSmiles(smile) for smile in smiles]
            img = Draw.MolsToGridImage(
                mols, molsPerRow=5, subImgSize=(400, 400), drawOptions=options, 
                legends=[f"Tanimoto:{float(Decimal(str(tanimoto)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))},  IEVCos:{float(Decimal(str(ievcos)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))}" for smiles, tanimoto, ievcos in zip(smiles, tanimoto_list, ievcos_list)]
            )
            img.save(f"results/{protein}/test{i}/tanimoto{int(tanimoto_threshold*10)}e-1_ievcos{int(ievcos_threshold*10)}e-1/{model}_all.jpg")

    with open(f"results/{protein}/test{i}/tanimoto{int(tanimoto_threshold*10)}e-1_ievcos{int(ievcos_threshold*10)}e-1/dock_smiles.smi", "w") as f:
        f.write(f"{test_dataset[i][0]} seed\n")
        for smiles, model in dock_smiles_list:
            f.write(f"{smiles} {model}\n")