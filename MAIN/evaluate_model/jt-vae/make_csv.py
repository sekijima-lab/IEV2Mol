import os
import sys
import torch
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
import pandas as pd
import subprocess

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw

import time

sys.path.append("../../data")
from Smiles_Vector_Dataset import Smiles_Vector_Dataset

sys.path.append("../../../JTVAE/JTVAE/FastJTNNpy3")
from fast_jtnn import *

cos_sim = nn.CosineSimilarity()

# test_datasetの10化合物から，各100個ずつzをサンプリングして生成する
# 生成した化合物のIEVを計算する
# 生成した化合物のSMILES，IEVCos，ドッキングスコア，IEVをcsvファイルに書き込む
# 全て jt-vae.* という名前で保存する



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)
print("num: ", torch.cuda.device_count())


test_dataset_path = "../../data/drd2_test_dataset_no_dot.pt"

vocab_path = "../../data/vocab_drd2_train_no_dot.txt"

vocab = [x.strip("\r\n ") for x in open(vocab_path)]
vocab = Vocab(vocab)

jtvae_path =  "../../model/jtvae_drd2_no_dot.pt"
depthT = 20
depthG = 3
latent_size = 56
hidden_size = 450


jtvae  = JTNNVAE(vocab, hidden_size, latent_size, depthT, depthG)
print(f"{jtvae_path}を読み込みます。")

jtvae.load_state_dict(torch.load(jtvae_path))
jtvae = jtvae.to(device)


# 実験
print("\n実験を開始します。")
jtvae.eval()
num_sample = 100
cos_sim = nn.CosineSimilarity()


with torch.no_grad():
    if os.path.exists(test_dataset_path):
        test_dataset = torch.load(test_dataset_path)
        print(f"test_datasetを{test_dataset_path}からロードしました。")
        print(f"test_datasetのサイズ: {len(test_dataset)}")
    else:
        print(f"test_datasetがないよ")
        exit()

    for i, (inp_smi, inp_vec) in enumerate(test_dataset):

        if not os.path.exists(f"../results/test{i}"):
            os.mkdir(f"../results/test{i}")
            os.mkdir(f"../results/test{i}/raw_csv")


        # num_sample個の分子を生成
        
        inp_vec = torch.tensor(inp_vec).reshape(1,-1).to(torch.float32).to(device) #shape=[1, 189]
        out_smis = []
       
        z = jtvae.encode_from_smiles([inp_smi])
        z = z.to(device)
        [latent_trees, latent_mols] = torch.chunk(z, 2, dim=-1)

        for j in range(num_sample):
            z_tree_vecs, tree_kl = jtvae.rsample(latent_trees, jtvae.T_mean, jtvae.T_var)
            z_mol_vecs, mol_kl = jtvae.rsample(latent_mols, jtvae.G_mean, jtvae.G_var)

            for z_mol_vec, z_tree_vec in zip(z_mol_vecs, z_tree_vecs):
                out_smis.append(
                    jtvae.decode(
                        z_tree_vec.unsqueeze(0),
                        z_mol_vec.unsqueeze(0),
                        prob_decode=False,
                    )
                )

        print(f"{i}th input_smi: {inp_smi}")
        print(f"{i}th", file=sys.stderr)

        # out_smisのうち有効なもののみをファイルに書き込み
        subprocess.run(["rm", "out_smiles.smi"])
        valid_smiles = []
        with open("out_smiles.smi", "w") as f:
            for j, smi in enumerate(out_smis):
                mol = Chem.MolFromSmiles(smi)

                if mol is not None:
                    smi = Chem.MolToSmiles(mol)
                    f.write(f"{smi} {j}\n")
                    valid_smiles.append(smi)
        
        subprocess.run(["rm", "prepred_out_smiles.maegz"])
        subprocess.run(["rm", "out_smiles_HTVS_pv.maegz"])
        subprocess.run(["rm", "out_smiles_HTVS_pv.interaction"])
        subprocess.run(["rm", "out_smiles_HTVS_pv_max.interaction"])

        # # TSUBAMEで動かす場合
        # subprocess.run(["qsub", "-g", "///group name///", "make_out_IEV_HTVS.sh"])

        # ローカルでシュレディンガーが動かせる場合は以下
        SCHROD_LICENSE_FILE = "///path-to-licence-file///"
        SIEVE = "../../../SIEVE-Score"
        SCHRODINGER = "///path-to-schrodinger///"
        TMPDIR = "/tmp"
        subprocess.run(["slacknotice", f"{SCHRODINGER}/ligprep", "-ismi", "out_smiles.smi", "-omae", "prepred_out_smiles.maegz", "-WAIT", "-NJOBS", "10", "-TMPDIR", f"{TMPDIR}"])
        subprocess.run(["slacknotice", f"{SCHRODINGER}/glide", "out_smiles_HTVS.in", "-OVERWRITE", "-NJOBS", "10", "-HOST", "localhost:10", "-TMPDIR", f"{TMPDIR}", "-ATTACHED", "-WAIT"])
        subprocess.run([f"{SCHRODINGER}/run", "python3", f"{SIEVE}/SIEVE-Score.py", "-m", "interaction", "-i", "./out_smiles_HTVS_pv.maegz", "-l", "sieve-score.log"])
        subprocess.run(["python3", "rest_max.py", "out_smiles_HTVS_pv.interaction"])

        print(" IEV計算開始", file=sys.stderr)

        # そのすきにDiversityの計算
        mols = [Chem.MolFromSmiles(smile) for smile in valid_smiles]
        fps = [AllChem.GetMorganFingerprintAsBitVect(m,2) for m in mols]
        sum = 0
        for j in range(len(fps)):
            tanimotos = DataStructs.BulkTanimotoSimilarity(fps[j],fps)
            sum += np.sum(tanimotos)

        print(f" Diversity: {1 - sum/(len(fps)**2)}")

        # さらにUniquenessの計算
        unique_smiles = list(set(valid_smiles))
        print(f" Uniqueness: {len(unique_smiles)/len(valid_smiles)}")

        loop = 0
        for j in range(90):
            time.sleep(60)
            print(f" {j}分経過", file=sys.stderr)
            loop += 1
            if os.path.exists("out_smiles_HTVS_pv_max.interaction"):
                break
        if os.path.exists("out_smiles_HTVS_pv_max.interaction"):
            print(" IEV計算完了", file=sys.stderr)
        else:
            print(" 待ち時間が90分を超えたため中止します", file=sys.stderr)
            exit()
        
        print(" IEV計算結果読み込み開始", file=sys.stderr)
        out_iev = pd.read_csv("out_smiles_HTVS_pv_max.interaction", index_col=0)
        valid_iev_index = list(out_iev.index)
        print(" IEVが有効なSMILES数: ", len(valid_iev_index))


        column = ["smiles", "ievcos", "dscore"]
        column.extend(list(out_iev.columns[1:-1]))

        df = pd.DataFrame(data=None, index = range(100), columns=column)
        df["smiles"] = out_smis
        

        cos_sim_list = []
        docking_score_list = []
        for index, entry in out_iev.iterrows():
            iev_tensor = torch.tensor(entry[1:-1], requires_grad=False).to(device)
            cos_sim_list.append(cos_sim(inp_vec.reshape(1,-1,1), iev_tensor.reshape(1,-1,1))[0].item())
            docking_score_list.append(entry[-1])

            df.iloc[index,1:3] = [cos_sim_list[-1], docking_score_list[-1]]
            df.iloc[index,3:] = iev_tensor.reshape(-1).cpu().detach().numpy()
        
        df.to_csv(f"../results/test{i}/raw_csv/jt-vae.csv")

