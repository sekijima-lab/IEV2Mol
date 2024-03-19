import os
import sys
import torch
import torch.nn as nn
import torch.utils.data
import os
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





# Lynxで生成したIFP-RNN（DRD2のIFPで学習ずみ）に対してテストのみを行う
# 生成した化合物のSMILES，IEVCos，ドッキングスコア，IEVをcsvファイルに書き込む
# これで作られるcsvはSMILESがValidなものだけになっているため，長さに注意する（indexは100あるが，SMILESがNoneの列がある）



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)
print("num: ", torch.cuda.device_count())

valid_smiles_vector_dataset_path = "../../data/drd2_test_dataset_no_dot.pt"

# 実験
print("\n実験を開始します。")
cos_sim = nn.CosineSimilarity()

valid_smiles_vector_dataset = torch.load(valid_smiles_vector_dataset_path)


for test_index in range(10):
    print(f"{test_index}番目のテストデータの処理を開始します")

    inp_smi = valid_smiles_vector_dataset[test_index][0]
    inp_vec = valid_smiles_vector_dataset[test_index][1]

    print(" Seed SMILES: ", inp_smi)
    
    

    # 各テストデータ内のポーズを取得（１つとは限らない）
    pose_index = -1
    for smi_file_path in sorted(os.listdir(f"test{test_index}")):
        if smi_file_path.endswith(".smi"):
            pose_index += 1
            print(f" test{test_index}/{smi_file_path}")
            file_head = smi_file_path.split(".")[0]
            print(f" {file_head} を処理します")

            if not os.path.exists(f"../results/test{test_index}"):
                os.mkdir(f"../results/test{test_index}")
                os.mkdir(f"../results/test{test_index}/raw_csv")


            # テストデータのSMILESを読み込む
            valid_smiles_df = pd.read_csv(f"test{test_index}/{file_head}.smi", sep=" ", header=0) # header: [Seed SMILES, "Seed"]
            

            valid_smiles = valid_smiles_df.iloc[:, 0].tolist()

            # out_smiles.smiに書き込む
            subprocess.run(["rm", "out_smiles.smi"])
            time.sleep(5)
            with open(f"out_smiles.smi", "w") as f:
                for idx_smi, smi in enumerate(valid_smiles):
                    smi = smi.split("\t")[0]
                    f.write(f"{smi} {idx_smi}\n")
            print("  out_smiles.smiに書き込みました")


            # テストデータのIEVを計算する
            print("  テストデータのIEVを計算します")
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

            print("  IEV計算開始", file=sys.stderr)
            loop = 0
            for t in range(90):
                time.sleep(60)
                print(f"  {t}分経過", file=sys.stderr)
                loop += 1
                if os.path.exists("out_smiles_HTVS_pv_max.interaction"):
                    break
            if os.path.exists("out_smiles_HTVS_pv_max.interaction"):
                print("  IEV計算完了", file=sys.stderr)
            else:
                print("  待ち時間が90分を超えたため中止します", file=sys.stderr)
                exit()

            out_iev = pd.read_csv("out_smiles_HTVS_pv_max.interaction", index_col=0)
        
            valid_iev_index = list(out_iev.index)
            print(" IEVが有効なSMILES数: ", len(valid_iev_index))


            column = ["smiles", "ievcos", "dscore"]
            column.extend(list(out_iev.columns[1:-1]))

            df = pd.DataFrame(data=None, index = range(100), columns=column)
            df["smiles"] = valid_smiles + [None for i in range(100-len(valid_smiles))]
            

            cos_sim_list = []
            docking_score_list = []
            for index, entry in out_iev.iterrows():
                iev_tensor = torch.tensor(entry[1:-1], requires_grad=False).to(device)
                cos_sim_list.append(cos_sim(inp_vec.reshape(1,-1,1).to(device), iev_tensor.reshape(1,-1,1))[0].item())
                docking_score_list.append(entry[-1])

                df.iloc[index,1:3] = [cos_sim_list[-1], docking_score_list[-1]]
                df.iloc[index,3:] = iev_tensor.reshape(-1).cpu().detach().numpy()
            
            df.to_csv(f"../results/test{test_index}/raw_csv/ifp-rnn_{pose_index}.csv")
