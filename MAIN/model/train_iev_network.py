import os
import sys
import numpy as np
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm

# sys.path.append("../../JTVAE/JTVAE/FastJTNNpy3")
# sys.path.append("../../../JTVAE/JTVAE/FastJTNNpy3")
# from fast_jtnn import *


# サンプリングの時0.1をかけない

class InteractionVAE(nn.Module):
    def __init__(
        self,
        device,
        vec_length,
        latent_dim=56,
        hidden_dim=450,
    ):
        super(InteractionVAE, self).__init__()

        self.device = device
        self.vec_length = vec_length
        self.latent_dim = latent_dim

        # encoder
        self.conv_1 = nn.Conv1d(
            1, 10, stride=3, kernel_size=3
        )  # out: (batch, 10, kernel_size/3)
        self.enc_batchnorm_1 = nn.BatchNorm1d(10)
        self.enc_linear_1 = nn.Linear(10 * (vec_length // 3), hidden_dim)
        self.enc_batchnorm_2 = nn.BatchNorm1d(hidden_dim)
        self.enc_linear_mean1 = nn.Linear(hidden_dim, latent_dim // 2)
        self.enc_linear_logvar1 = nn.Linear(hidden_dim, latent_dim // 2)
        self.enc_linear_mean2 = nn.Linear(hidden_dim, latent_dim // 2)
        self.enc_linear_logvar2 = nn.Linear(hidden_dim, latent_dim // 2)

        # decoder
        self.dec_linear_2 = nn.Linear(latent_dim, latent_dim)
        self.dec_batchnorm_3 = nn.BatchNorm1d(latent_dim)
        self.dec_linear_3 = nn.Linear(latent_dim, hidden_dim)
        self.dec_batchnorm_4 = nn.BatchNorm1d(hidden_dim)
        self.dec_linear_4 = nn.Linear(hidden_dim, 10 * (vec_length // 3))
        self.dec_batchnorm_5 = nn.BatchNorm1d(10 * (vec_length // 3))
        self.dec_transcnn = nn.ConvTranspose1d(10, 1, stride=3, kernel_size=3)
        self.dec_batchnorm_6 = nn.BatchNorm1d(vec_length)
        self.dec_linear_5 = nn.Linear(vec_length, vec_length)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        self.encoder_layers = [
            "enc_linear_1.weight",
            "enc_linear_1.bias",
            "enc_linear_mean1.weight",
            "enc_linear_mean1.bias",
            "enc_linear_mean2.weight",
            "enc_linear_mean2.bias",
            "enc_enc_linear_logvar1.weight",
            "enc_enc_linear_logvar1.bias",
            "enc_enc_linear_logvar2.weight",
            "enc_enc_linear_logvar2.bias",
            "enc_batchnorm_1.weight",
            "enc_batchnorm_1.bias",
            "enc_batchnorm_2.weight",
            "enc_batchnorm_2.bias",
            "enc_conv_1.weight",
            "enc_conv_1.bias",
        ]

        self.decoder_layers = [
            "dec_linear_2.weight",
            "dec_linear_2.bias",
            "dec_linear_3.weight",
            "dec_linear_3.bias",
            "dec_linear_4.weight",
            "dec_linear_4.bias",
            "dec_linear_5.weight",
            "dec_linear_5.bias",
            "dec_batchnorm_3.weight",
            "dec_batchnorm_3.bias",
            "dec_batchnorm_4.weight",
            "dec_batchnorm_4.bias",
            "dec_batchnorm_5.weight",
            "dec_batchnorm_5.bias",
            "dec_batchnorm_6.weight",
            "dec_batchnorm_6.bias",
            "dec_transcnn.weight",
            "dec_transcnn.bias",
        ]

    def kl_loss(self, z_mean, z_logvar):
        batch_size = z_mean.size(0)
        return (
            -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp()) / batch_size
        )

    def sampling(self, z_mean, z_logvar):
        # epsilon = 1e-1 * torch.randn_like(z_logvar)
        epsilon = torch.randn_like(z_logvar)
        return torch.exp(0.5 * z_logvar) * epsilon + z_mean

    def encode(self, x):
        x = x.to(torch.float32)
        x = x.view(x.size(0), 1, x.size(-1))  # (batch, 1, vec_length)
        x = self.relu(self.conv_1(x))  # (batch, 10, vec_length/3)
        x = self.enc_batchnorm_1(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # (batch, 10*(vec_length/3))
        x = F.selu(self.enc_linear_1(x))  # (batch, hidden_dim)
        logit = self.enc_batchnorm_2(x)
        return logit

    def rsample(self, logit):
        z_mean_1 = self.enc_linear_mean1(logit)
        z_logvar_1 = self.enc_linear_logvar1(logit)  # tree_vecsに当たる

        z_mean_2 = self.enc_linear_mean2(logit)
        z_logvar_2 = self.enc_linear_logvar2(logit)  # mol_vecsに当たる

        return torch.cat(
            [self.sampling(z_mean_1, z_logvar_1), self.sampling(z_mean_2, z_logvar_2)],
            dim=-1,
        ), self.kl_loss(z_mean_1, z_logvar_1) + self.kl_loss(z_mean_2, z_logvar_2)

    def decode(self, z):
        batch_size = z.size(0)
        # z.shape = (batch, latent_dim)
        z = F.selu(self.dec_linear_2(z))  # (batch, latent_dim)
        z = self.dec_batchnorm_3(z)
        z = self.dropout(z)
        z = F.selu(self.dec_linear_3(z))  # (batch, hidden_dim)
        z = self.dec_batchnorm_4(z)
        z = self.dropout(z)
        z = F.selu(self.dec_linear_4(z))  # (batch, 10*(vec_length/3))
        z = self.dec_batchnorm_5(z)
        z = self.dropout(z)
        z = z.contiguous().view(batch_size, 10, -1)  # (batch, 10, vec_length/3)
        z = self.dec_transcnn(z)  # (batch, 1, vec_length)
        z = z.view(batch_size, -1)  # (batch, vec_length)
        z = self.dec_batchnorm_6(z)
        y = self.dec_linear_5(z)  # (batch, vec_length)
        return y

    def forward(self, x):
        x = x.to(torch.float32)
        logit = self.encode(x)
        z, kl_loss = self.rsample(logit)
        return self.decode(z), z, kl_loss

    def inter_reconstruction_loss(self, inter_sampled, inter):
        batch_size = inter.size(0)
        return nn.L1Loss(reduction="sum")(inter_sampled, inter) / batch_size

    def predict(self, dataset):
        self.eval()
        for _, input_vecs in torch.utils.data.DataLoader(
            dataset, batch_size=len(dataset)
        ):
            # input_vecs: (batch, vec_length)のテンソルを想定
            with torch.no_grad():
                input_vecs = input_vecs.to(self.device)
                output_vecs, _, _ = self.forward(input_vecs)

            return input_vecs, output_vecs

    def fit(
        self,
        dataset,
        save_path,
        batch_size=32,
        lr=1e-3,
        clip_norm=50.0,
        beta=0.0,
        step_beta=0.002,
        max_beta=1.0,
        warmup=40000,
        epochs=20,
        anneal_rate=0.9,
        anneal_iter=40000,
        kl_anneal_iter=2000,
    ):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, anneal_rate)
        total_step = 0

        for epoch in tqdm(range(epochs)):
            train_loss = 0
            inter_recon_loss = 0
            inter_kl_loss = 0

            self.train()
            for batch_idx, (_, input_vecs) in enumerate(
                torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True,
                )
            ):
                total_step += 1
                input_vecs = input_vecs.to(self.device)
                optimizer.zero_grad()

                # ロス計算
                output_vecs, z, tmp_inter_kl_loss = self.forward(input_vecs)
                tmp_inter_recon_loss = self.inter_reconstruction_loss(
                    output_vecs, input_vecs
                )

                loss = beta * tmp_inter_kl_loss + tmp_inter_recon_loss

                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), clip_norm)
                optimizer.step()

                if total_step % anneal_iter == 0 and total_step >= warmup:
                    scheduler.step()
                    print("learning rate: %.6f" % scheduler.get_last_lr()[0])

                if total_step % kl_anneal_iter == 0 and total_step >= warmup:
                    beta = min(max_beta, beta + step_beta)

                train_loss += loss
                inter_kl_loss += tmp_inter_kl_loss
                inter_recon_loss += tmp_inter_recon_loss

            print(
                epoch,
                "epoch:",
                "total_step:",
                total_step,
                " train loss = ",
                train_loss.item() / (batch_idx + 1),
                ", inter_kl_loss = ",
                inter_kl_loss.item() / (batch_idx + 1),
                ", inter_recon_loss = ",
                inter_recon_loss.item() / (batch_idx + 1),
                ", beta = ",
                beta,
                ", lr = ",
                scheduler.get_last_lr()[0],
            )

        torch.save(self.state_dict(), save_path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # データの読み込み
    vec_length = 100
    dataset = torch.randn(1000, vec_length)

    # モデルの定義
    model = InteractionVAE(device, vec_length)

    # 学習
    model.fit(dataset, "model.pth")



if __name__ == "__main__":
    main()
        