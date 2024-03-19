import os
import pickle
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from iev2mol import CVAE
from inter_vae_0110 import InteractionVAE
from matplotlib import pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from smiles_vae_20231004 import SmilesVAE, make_vocab, read_smiles

sys.path.append('../data')
from Smiles_Vector_Dataset import Smiles_Vector_Dataset

cos_sim = nn.CosineSimilarity()

##############################ヘッダ#######################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)
print("num: ", torch.cuda.device_count())


def torch_fix_seed(seed=1):
    os.environ['PYTHONHASHSEED'] = str(seed)
    # # Python random
    # random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

############################################################################
# smilesvaeが違うのでtensorizeしたデータセットも変わる．

inter_vae_path = "inter_vae_drd2_no_dot.pt"          # DRD2の阻害剤データで事前学習したVAE
smiles_vae_path  = "smiles_vae_dmqp1m_no_dot_dup.pt"      # 被りのないDMQP1MのSMILESで事前学習したVAE

smiles_pretrain_data_path = "../data/Druglike_million_canonical_no_dot_dup.smi"
trained_vae_path = "iev2mol_no_dot.pt"

train_dataset_path = "../data/smitensor_iev_train_dataset_tensorized_by_smiles_vae_dmqp1m_no_dot_dup.pt" # 被りのないDMQP1MのSMILESで事前学習したSMILES-VAEでtensorizeしたデータセット
test_dataset_path = "../data/smitensor_iev_test_dataset_tensorized_by_smiles_vae_dmqp1m_no_dot_dup.pt"

train_raw_dataset_path = "../data/drd2_train_dataset_no_dot.pt"
test_raw_dataset_path = "../data/drd2_test_dataset_no_dot.pt"


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


# 学習済みモデルを用意
if not os.path.exists(inter_vae_path):
    print(f"Interaction VAEの学習済みモデルがないよ")
    exit()

elif not os.path.exists(smiles_vae_path):
    print(f"SMILES VAEの学習済みモデルがないよ")
    exit()

else:
    # inter_vae = InteractionVAE(device=device, vec_length=189, latent_dim=inter_config["latent_dim"], hidden_dim=inter_config["hidden_dim"])
    inter_vae = InteractionVAE(device=device, vec_length=189)
    inter_vae.load_state_dict(torch.load(inter_vae_path))
    print(f"学習ずみInteraction VAEとして{inter_vae_path}をロードしました")

    smiles_vae = SmilesVAE(device=device, vocab=make_vocab(read_smiles(smiles_pretrain_data_path)), config=smiles_config).to(device)
    # print("学習前:\n", smiles_vae.sample_smiles(10, max_length=201))
    smiles_vae.load_state_dict(torch.load(smiles_vae_path))
    # print("学習後:\n", smiles_vae.sample_smiles(10, max_length=201))
    torch_fix_seed()
    cvae = CVAE(device=device, pretrained_smiles_vae=smiles_vae, pretrained_inter_vae=inter_vae).to(device)


# データの準備
if os.path.exists(test_dataset_path) and os.path.exists(train_dataset_path):
    test_dataset = torch.load(test_dataset_path)
    train_dataset = torch.load(train_dataset_path)
    print(f"test_datasetとtrain_datasetを{test_dataset_path}と{train_dataset_path}からロードしました。")
else:
    # trainデータのSMILESをtensorize
    smiles_tensor_list = []
    inter_tensor_list = []

    for smi, iev in torch.load(train_raw_dataset_path):
        smiles_tensor = smiles_vae.string2tensor(smi)
        smiles_tensor_list.append(smiles_tensor)
        inter_tensor_list.append(iev.to(torch.float32))

    length_smiles = [len(smiles_tensor) for smiles_tensor in smiles_tensor_list]


    zip_data = zip(length_smiles, smiles_tensor_list, inter_tensor_list)
    zip_data = sorted(zip_data, key=lambda x: x[0], reverse=True) # length_smilesで降順ソート
    length_smiles, smiles_tensor_list, inter_tensor_list = zip(*zip_data)

    smiles_tensor_list = torch.nn.utils.rnn.pack_sequence(smiles_tensor_list)
    smiles_tensor_list, length_smiles_tensor = torch.nn.utils.rnn.pad_packed_sequence(smiles_tensor_list, batch_first=True, padding_value=smiles_vae.PAD, total_length=max_smiles_length)


    train_dataset = torch.utils.data.TensorDataset(smiles_tensor_list, torch.stack(inter_tensor_list))
    torch.save(train_dataset, train_dataset_path)


    # testデータのSMILESをtensorize
    smiles_tensor_list = []
    inter_tensor_list = []
    for smi, iev in torch.load(test_raw_dataset_path):
        smiles_tensor = smiles_vae.string2tensor(smi)
        smiles_tensor_list.append(smiles_tensor)
        inter_tensor_list.append(iev.to(torch.float32))

    length_smiles = [len(smiles_tensor) for smiles_tensor in smiles_tensor_list]


    zip_data = zip(length_smiles, smiles_tensor_list, inter_tensor_list)
    zip_data = sorted(zip_data, key=lambda x: x[0], reverse=True) # length_smilesで降順ソート
    length_smiles, smiles_tensor_list, inter_tensor_list = zip(*zip_data)

    smiles_tensor_list = torch.nn.utils.rnn.pack_sequence(smiles_tensor_list)
    smiles_tensor_list, length_smiles_tensor = torch.nn.utils.rnn.pad_packed_sequence(smiles_tensor_list, batch_first=True, padding_value=smiles_vae.PAD, total_length=max_smiles_length)

    test_dataset = torch.utils.data.TensorDataset(smiles_tensor_list, torch.stack(inter_tensor_list))
    torch.save(test_dataset, test_dataset_path)
    print(f"test_datasetとtrain_datasetを{test_dataset_path}と{train_dataset_path}に保存しました。")

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

# VAEの学習
if os.path.exists(trained_vae_path):
    print(f"{trained_vae_path}がすでに存在します")
else:
    cvae.finetune(epochs=100, loader=train_dataloader, lr=1e-4, milestones=[i*20 for i in range(1,5)], gamma=0.8)
    cvae.save(trained_vae_path)
    print(f"{trained_vae_path}を保存しました")
