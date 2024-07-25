"""
- SMILES VAEのデコーダ入力部分にDense層を3つ付けて入力を拡張したものを学習する
- Lossとして Reconのみ を利用する
"""


import os
import subprocess
import sys
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from inter_vae_0110 import InteractionVAE
from smiles_vae_20231004 import SmilesVAE, Trainer, make_vocab, read_smiles
from tqdm.auto import tqdm

# ちゃんと凍結できているかの確認は必要そう


class CVAE(nn.Module):
    def __init__(
        self,
        device,
        pretrained_smiles_vae: SmilesVAE,
        pretrained_inter_vae: InteractionVAE,
    ):
        super().__init__()
        self.device = device
        self.pretrained_smiles_vae = pretrained_smiles_vae
        self.pretrained_inter_vae = pretrained_inter_vae

        # デコーダ以外の重みを凍結しておく
        for param in self.pretrained_smiles_vae.parameters():
            param.requires_grad = False
        for param in self.pretrained_inter_vae.parameters():
            param.requires_grad = False
        for param in self.pretrained_smiles_vae.decoder.parameters():
            param.requires_grad = True

        # コンディションをconcatした後の次元数
        self.concat_latent_dim = (
            self.pretrained_smiles_vae.config["latent_size"]
            + self.pretrained_inter_vae.latent_dim
        )

        # SMILES VAEのデコーダ入力部分にDense層を3つ付けて入力の次元数を拡張する
        decrease_dim = self.pretrained_inter_vae.latent_dim // 3
        self.new_smiles_decoder_head = nn.Sequential(
            nn.Linear(self.concat_latent_dim, self.concat_latent_dim - decrease_dim),
            nn.Linear(
                self.concat_latent_dim - decrease_dim,
                self.concat_latent_dim - decrease_dim * 2,
            ),
            nn.Linear(
                self.concat_latent_dim - decrease_dim * 2,
                self.pretrained_smiles_vae.config["latent_size"],
            ),
        )

    def sample_smiles_with_IEV(self, condition, batch_size, max_length=201):
        """
        - condition: torch.tensor (latent_dim)
        - batch_size: int
        - max_length: int
        - return: list
        """
        self.eval()
        with torch.no_grad():
            random_smiles_latent = self.pretrained_smiles_vae.sample_z_prior(
                batch_size
            ).to(
                self.device
            )  # (batch, latent_dim)
            condition = (
                condition.unsqueeze(0).repeat(batch_size, 1).to(self.device)
            )  # (batch, latent_dim)

            z = torch.cat(
                (random_smiles_latent, condition), dim=1
            )  # (batch, concat_latent_dim)
            z = self.new_smiles_decoder_head(z)  # (batch, latent_dim)

            return self.pretrained_smiles_vae.sample_smiles_from_z(
                z, max_length=max_length
            )

    def forward(self, dataset):
        """
        - input: torch.utils.data.TensorDataset (smiles, interaction)
        - SMILES VAEのデコーダ入力部分にDense層を3つ付けて入力を拡張したものを学習する
        - 拡張部分以外は重みを凍結しておく
        - return: invalid_per, IEV_loss
        """
        self.train()
        self.zero_grad()

        smiles, interaction = dataset

        # SMILESを潜在変数にエンコード
        smiles_latent, _= self.pretrained_smiles_vae.forward_encoder(smiles)

        # interactionを潜在変数にエンコード
        interaction = interaction.to(torch.float32).to(self.device)
        _, inter_latent, _ = self.pretrained_inter_vae.forward(interaction)  # (batch, latent_dim)

        # SMILESの潜在変数とinteractionの潜在変数をconcat
        concat_latent = torch.cat(
            (smiles_latent, inter_latent), dim=1
        )  # (batch, concat_latent_dim)

        z = self.new_smiles_decoder_head(concat_latent)  # (batch, latent_dim)
        
        each_recon_loss = self.pretrained_smiles_vae.forward_decoder_eachloss(smiles, z)
        gen_smiles = self.pretrained_smiles_vae.sample_smiles_from_z(z)
        
        
        loss = sum(each_recon_loss)/len(each_recon_loss)

            
        print(f"loss: {loss}")

        wandb.log({
            "loss": loss.item()})
        
        return loss
    

    def finetune(self, epochs, loader, lr, milestones, gamma):
        self.train()
        self.zero_grad()

        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )

        wandb.init(project="IEV2Mol",
                   config={"epochs": epochs,
                            "lr": lr,
                            "milestones": milestones,
                            "gamma": gamma,
                            "batch_size": loader.batch_size,
                            "concat_latent_dim": self.concat_latent_dim,
                            "smiles_vae_latent_dim": self.pretrained_smiles_vae.config["latent_size"],
                            "inter_vae_latent_dim": self.pretrained_inter_vae.latent_dim
                            })

        for epoch in range(epochs):
            print(f"epoch {epoch} start!")
            tqdm_data = tqdm(loader, desc="Training (epoch #{})".format(epoch))

            for batch in tqdm_data:
                optimizer.zero_grad()
                loss = self.forward(batch)
                loss.backward()
                optimizer.step()
                tqdm_data.set_postfix(
                    {
                        "loss": loss.item(),
                        "lr": scheduler.get_last_lr()[0],
                    }
                )

            scheduler.step()

    def save(self, path):
        torch.save(self.state_dict(), path)
        print(f"モデルを{path}に保存しました")
