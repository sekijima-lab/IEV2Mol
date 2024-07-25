import os
import sys
from inter_vae_0110 import InteractionVAE
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train', '-t', type=str, help='path to train dataset')
parser.add_argument('--valid', '-v', type=str, help='path to valid dataset')
parser.add_argument('--out', '-o', type=str, help='path to save')
args = parser.parse_args()

sys.path.append("../data")
from Smiles_Vector_Dataset import Smiles_Vector_Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)
print("num: ", torch.cuda.device_count())

def check_files_exist(directory):
    for root, dirs, files in os.walk(directory):
        if files:
            return True
    return False

def fit_inter_vae(device, tensor_len, train_dataset, valid_dataset, save_path):
    print("Start fit_inter_vae()")

    # Interaction VAEの学習
    print(" Interaction VAEの学習を開始します。")
    interaction_vae = InteractionVAE(device, tensor_len)

    interaction_vae = interaction_vae.to(device)
    
    interaction_vae.fit(
        train_dataset,
        valid_dataset,
        save_path,
        batch_size=128,
        lr=1e-3,
        warmup=4000,
        anneal_iter=4000,
        kl_anneal_iter=100,
        epochs=200,
        beta=1.0,
        step_beta=0,
    )
    return interaction_vae


if __name__ == "__main__":
    
    # # trainデータで学習したInteraction VAEを用意
    # trained_inter_vae_path = "inter_vae_drd2_no_dot.pt"
    # smiles_inter_train_dataset_path = "../data/drd2_train_dataset_no_dot.pt"
    
    # trained_inter_vae_path = args.out
    # smiles_train_path = args.train
    # smiles_valid_path = args.valid
    
    smiles_train_path = '../data/AA2AR/AA2AR_train.pt'
    smiles_valid_path = '../data/AA2AR/AA2AR_test.pt'
    trained_inter_vae_path = './AA2AR/'
    
    train_data = torch.load(smiles_train_path)
    val_data = torch.load(smiles_valid_path)
    
    if check_files_exist(trained_inter_vae_path):
        print(f"{trained_inter_vae_path}がすでに存在しています．")
    else:
        print(f"学習を開始します．")
        print(f'len_train:{len(train_data.smiles)}')
        print(f'len_valid:{len(val_data.smiles)}')
        print(f'tensor_len:{len(train_data[0][1])}')
        interaction_vae = fit_inter_vae(device=device, tensor_len=len(train_data[0][1]),
                                        train_dataset=train_data, valid_dataset=val_data, save_path=trained_inter_vae_path)