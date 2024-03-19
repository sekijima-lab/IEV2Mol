"""
chembl_33_no_dot.smiからランダムに100個*10セットのsmilesを選ぶ
raw_csvを計算し，保存する
"""

import os
import sys
import torch
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import subprocess
from tqdm import tqdm

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw

import time

sys.path.append("../../data")
from Smiles_Vector_Dataset import Smiles_Vector_Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)
print("num: ", torch.cuda.device_count())


# 実験
print("\n実験を開始します。")

num_sample = 100
cos_sim = nn.CosineSimilarity()
test_dataset_path = "../../data/drd2_test_dataset_no_dot.pt"

if os.path.exists(test_dataset_path):
    test_dataset = torch.load(test_dataset_path)
    print(f"test_datasetを{test_dataset_path}からロードしました。")
    print(f"test_datasetのサイズ: {len(test_dataset)}")
else:
    print(f"test_datasetがないよ")
    exit()

# ランダムデータがない場合，作成
if not os.path.exists(f"chembl33_random100_9.smi"):
    with open("chembl33_random100_0.smi", "w") as f:
        all_smiles = [line.split(" ")[0] for line in open("../../data/chembl_33_no_dot.smi")]
    for i in tqdm(range(10)):
        random_smiles = np.random.choice(all_smiles, num_sample, replace=False)
        with open(f"chembl33_random100_{i}.smi", "w") as f:
            for index, smi in enumerate(random_smiles):
                f.write(f"{smi} {index}\n")

# ランダムデータを使ってIEV計算
for i, (inp_smi, inp_vec) in enumerate(test_dataset):

    if not os.path.exists(f"../results/test{i}"):
        os.mkdir(f"../results/test{i}")
        os.mkdir(f"../results/test{i}/raw_csv")

    inp_vec = torch.tensor(inp_vec).reshape(1,-1).to(torch.float32).to(device) #shape=[1, 189]
    
    subprocess.run(["cp", f"chembl33_random100_{i}.smi", "out_smiles.smi"])
        
    subprocess.run(["rm", "prepred_out_smiles.maegz"])
    subprocess.run(["rm", "out_smiles_HTVS_pv.maegz"])
    subprocess.run(["rm", "out_smiles_HTVS_pv.interaction"])
    subprocess.run(["rm", "out_smiles_HTVS_pv_max.interaction"])

    SCHROD_LICENSE_FILE = "path-to-licence-file"
    SIEVE = "../../../SIEVE-Score"
    SCHRODINGER = "path-to-schrodinger"
    TMPDIR = "/tmp"
    subprocess.run(["slacknotice", f"{SCHRODINGER}/ligprep", "-ismi", "out_smiles.smi", "-omae", "prepred_out_smiles.maegz", "-WAIT", "-NJOBS", "10", "-TMPDIR", f"{TMPDIR}"])
    subprocess.run(["slacknotice", f"{SCHRODINGER}/glide", "out_smiles_HTVS.in", "-OVERWRITE", "-NJOBS", "10", "-HOST", "localhost:10", "-TMPDIR", f"{TMPDIR}", "-ATTACHED", "-WAIT"])
    subprocess.run([f"{SCHRODINGER}/run", "python3", f"{SIEVE}/SIEVE-Score.py", "-m", "interaction", "-i", "./out_smiles_HTVS_pv.maegz", "-l", "sieve-score.log"])
    subprocess.run(["python3", "rest_max.py", "out_smiles_HTVS_pv.interaction"])
    

    print(" IEV計算開始", file=sys.stderr)

    loop = 0
    for j in range(120):
        time.sleep(10)
        print(f" {j*10}秒経過", file=sys.stderr)
        loop += 1
        if os.path.exists("out_smiles_HTVS_pv_max.interaction"):
            break
    if os.path.exists("out_smiles_HTVS_pv_max.interaction"):
        print(" IEV計算完了", file=sys.stderr)
    else:
        print(" 待ち時間が1200秒を超えたため中止します", file=sys.stderr)
        exit()
    
    print(" IEV計算結果読み込み開始", file=sys.stderr)
    out_iev = pd.read_csv("out_smiles_HTVS_pv_max.interaction", index_col=0)
    valid_iev_index = list(out_iev.index)
    print(" IEVが有効なSMILES数: ", len(valid_iev_index))


    column = ["smiles", "ievcos", "dscore"]
    column.extend(list(out_iev.columns[1:-1]))

    df = pd.DataFrame(data=None, index = range(100), columns=column)
    out_smis = [line.split(" ")[0] for line in open("out_smiles.smi")]
    df["smiles"] = out_smis
    

    cos_sim_list = []
    docking_score_list = []
    for index, entry in out_iev.iterrows():
        iev_tensor = torch.tensor(entry[1:-1], requires_grad=False).to(device)
        cos_sim_list.append(cos_sim(inp_vec.reshape(1,-1,1), iev_tensor.reshape(1,-1,1))[0].item())
        docking_score_list.append(entry[-1])
        df.iloc[index,1:3] = [cos_sim_list[-1], docking_score_list[-1]]
        df.iloc[index,3:] = iev_tensor.reshape(-1).cpu().detach().numpy()
    
    df.to_csv(f"../results/test{i}/raw_csv/random_chembl33.csv")

