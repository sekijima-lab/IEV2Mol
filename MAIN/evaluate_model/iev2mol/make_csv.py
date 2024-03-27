import os
import pickle
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw

sys.path.append("../../model")
from iev2mol import CVAE
from inter_vae_0110 import InteractionVAE
from smiles_vae_20231004 import SmilesVAE, make_vocab, read_smiles

cos_sim = nn.CosineSimilarity()


# test_datasetの10化合物から，各100個ずつzをサンプリングして生成する
# 生成した化合物のIEVを計算する
# 生成した化合物のSMILES，IEVCos，ドッキングスコア，IEVをcsvファイルに書き込む
# 全て iev2mol.* という名前で保存する




batch_size = 128
max_smiles_length = 201

smiles_config = {
    "encoder_hidden_size": 256,  # encoderのGRUの隠れ層の次元数h
    "encoder_num_layers": 1,  # encoderのGRUの層数
    "bidirectional": True,  # Trueなら双方向，Falseなら単方向
    "encoder_dropout": 0.5,  # encoderのGRUのdropout率
    "latent_size": 128,  # 潜在変数の次元数z
    "decoder_hidden_size": 512,  # decoderのGRUの隠れ層の次元数h
    "decoder_num_layers": 3,  # decoderのGRUの層数
    "decoder_dropout": 0,  # decoderのGRUのdropout率
    "n_batch": 512,  # バッチサイズ
    "clip_grad": 50,
    "kl_start": 0, # KL_Annearningの開始epoch
    "kl_w_start": 0, # KL_Annearningの重みの開始値
    "kl_w_end": 0.05,
    "lr_start": 3 * 1e-4,
    "lr_n_period": 10,
    "lr_n_restarts": 10,
    "lr_n_mult": 1, # epoch数 = sum(lr_n_period * lr_n_mult**i), i=range(lr_n_restarts)
    "lr_end": 3 * 1e-4,
    "n_last": 1000,
    "n_jobs": 1,
    "n_workers": 1,
    "model_save": None,
    "save_frequency": 10,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)
print("num: ", torch.cuda.device_count())

inter_vae_path = "../../model/inter_vae_drd2_no_dot.pt"                 # DRD2の阻害剤データで事前学習したVAE
smiles_vae_path  = "../../model/smiles_vae_dmqp1m_no_dot_dup.pt" # Drug_likeなSMILESで事前学習したVAE

smiles_pretrain_data_path = "../../data/Druglike_million_canonical_no_dot_dup.smi"

test_dataset_path = "../../data/smitensor_iev_test_dataset_tensorized_by_smiles_vae_dmqp1m_no_dot_dup.pt"

trained_cvae_path = "../../model/iev2mol_no_dot.pt"

# データの準備
if os.path.exists(test_dataset_path):
    test_dataset = torch.load(test_dataset_path)
    print(f"test_dataset is loaded from {test_dataset_path}")
    print(f"test_dataset size: {len(test_dataset)}")
else:
    print(f"test_dataset is not exist at {test_dataset_path}")
    exit()

vec_length = test_dataset[0][1].shape[-1]


# 事前学習済みモデルを用意
if not os.path.exists(inter_vae_path):
    print(f"trained Interaction VAE is not exists")
    exit()

elif not os.path.exists(smiles_vae_path):
    print(f"trained SMILES VAE is not exists")
    exit()

elif not os.path.exists(trained_cvae_path):
    print(f"trained IEV2Mol is not exists at {trained_cvae_path}")
    exit()

inter_vae = InteractionVAE(device=device, vec_length=vec_length)
inter_vae.load_state_dict(torch.load(inter_vae_path))
print(f"trained Interaction VAE is loaded from {inter_vae_path}")

smiles_vae = SmilesVAE(device=device, vocab=make_vocab(read_smiles(smiles_pretrain_data_path)), config=smiles_config).to(device)
smiles_vae.load_state_dict(torch.load(smiles_vae_path))
model = CVAE(device=device, pretrained_smiles_vae=smiles_vae, pretrained_inter_vae=inter_vae).to(device)
model.load_state_dict(torch.load(trained_cvae_path))
print(f"trained IEV2Mol is loaded from {trained_cvae_path}")



# 実験


with torch.no_grad():
    
    model.eval()
    fig = plt.figure(figsize=(40, 50))
    bar_width = 0.5
    out_vecs = []

    if not os.path.exists("../results"):
        os.mkdir("../results")

    for i, (inp_smi, inp_vec) in enumerate(test_dataset):

        if not os.path.exists(f"../results/test{i}"):
            os.mkdir(f"../results/test{i}")
            os.mkdir(f"../results/test{i}/raw_csv")

        inp_vec = inp_vec.reshape(1,-1).to(device)
        
        inp_vec = torch.tensor(inp_vec).reshape(1,-1).to(torch.float32).to(device) #shape=[1, 189]
        out_smis = []
        
        for j in range(100):
            # IEV VAEによるconditionの生成

            _, condition, _ = inter_vae.forward(inp_vec) # shape=[1,56]
            condition = condition.reshape(-1) # shape=[56]

            # 分子の生成
            out_smis.extend(model.sample_smiles_with_IEV(condition, batch_size=1, max_length=201))


        
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
        subprocess.run([f"{SCHRODINGER}/ligprep", "-ismi", "out_smiles.smi", "-omae", "prepred_out_smiles.maegz", "-WAIT", "-NJOBS", "10", "-TMPDIR", f"{TMPDIR}"])
        subprocess.run([f"{SCHRODINGER}/glide", "out_smiles_HTVS.in", "-OVERWRITE", "-NJOBS", "10", "-HOST", "localhost:10", "-TMPDIR", f"{TMPDIR}", "-ATTACHED", "-WAIT"])
        subprocess.run([f"{SCHRODINGER}/run", "python3", f"{SIEVE}/SIEVE-Score.py", "-m", "interaction", "-i", "./out_smiles_HTVS_pv.maegz", "-l", "sieve-score.log"])
        subprocess.run(["python3", "rest_max.py", "out_smiles_HTVS_pv.interaction"])


        print(" start calculating IEV", file=sys.stderr)

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
            print(f" {j} min passed", file=sys.stderr)
            loop += 1
            if os.path.exists("out_smiles_HTVS_pv_max.interaction"):
                break
        if os.path.exists("out_smiles_HTVS_pv_max.interaction"):
            print(" finish calusulating IEV", file=sys.stderr)
        else:
            print(" waiting over 90 min. stop.", file=sys.stderr)
            exit()
        
        print(" loading IEV", file=sys.stderr)
        out_iev = pd.read_csv("out_smiles_HTVS_pv_max.interaction", index_col=0)
        valid_iev_index = list(out_iev.index)
        print(" num of SMILES whose IEV is valid: ", len(valid_iev_index))


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
        
        df.to_csv(f"../results/test{i}/raw_csv/iev2mol.csv")
