import os
import sys
from inter_vae_0110 import InteractionVAE
import torch

sys.path.append("../data")
from Smiles_Vector_Dataset import Smiles_Vector_Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)
print("num: ", torch.cuda.device_count())

def fit_inter_vae(device, tensor_len, smiles_inter_dataset, save_path):
    print("Start fit_inter_vae()")

    # Interaction VAEの学習
    # print(" Interaction VAEの学習を開始します。")
    interaction_vae = InteractionVAE(device, tensor_len)

    interaction_vae = interaction_vae.to(device)
    
    interaction_vae.fit(
        smiles_inter_dataset,
        save_path,
        batch_size=128,
        lr=1e-3,
        warmup=4000,
        anneal_iter=4000,
        kl_anneal_iter=100,
        epochs=100,
        beta=1.0,
        step_beta=0,
    )
    del smi_inter_dataset
    return interaction_vae


if __name__ == "__main__":
    
    # trainデータで学習したInteraction VAEを用意
    trained_inter_vae_path = "inter_vae_drd2_no_dot.pt"
    smiles_inter_train_dataset_path = "../data/drd2_train_dataset_no_dot.pt"

    smiles_inter_dataset = torch.load(smiles_inter_train_dataset_path)

    if os.path.exists(trained_inter_vae_path):
        print(f"{trained_inter_vae_path} is already exists")
    else:
        print(f"start training")
        print(len(smiles_inter_dataset[0][1]))
        interaction_vae = fit_inter_vae(device=device, tensor_len=len(smiles_inter_dataset[0][1]), smiles_inter_dataset=smiles_inter_dataset, save_path=trained_inter_vae_path)
