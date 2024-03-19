import sys

sys.path.append("../../JTVAE/JTVAE/FastJTNNpy3/fast_molvae")
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

import math, random, sys
import numpy as np
from collections import deque
import pickle as pickle

sys.path.append("../../JTVAE/JTVAE/FastJTNNpy3/")
from fast_jtnn import *
import rdkit
from tqdm import tqdm
import os

##############################ヘッダ#######################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)
print("num: ", torch.cuda.device_count())


def torch_fix_seed(seed=1):
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

############################################################################

def fit(
    device,
    model,
    moses_dir,
    vocab,
    save_path,
    batch_size,
    lr=1e-3,
    clip_norm=50.0,
    beta=0.0,
    step_beta=0.002,
    max_beta=1.0,
    warmup=40000,
    epoch=20,
    anneal_rate=0.9,
    anneal_iter=40000,
    kl_anneal_iter=2000,
    print_iter=50,
):
    model = model.to(device)
    print(model)

    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

    print(
        "Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,)
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, anneal_rate)
    scheduler.step()

    param_norm = lambda m: math.sqrt(
        sum([p.norm().item() ** 2 for p in m.parameters()])
    )
    grad_norm = lambda m: math.sqrt(
        sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None])
    )

    total_step = 0
    beta = beta
    meters = np.zeros(4)

    for epoch in tqdm(range(epoch)):
        loader = MolTreeFolder(moses_dir, vocab, batch_size)  # , num_workers=4)
        for batch in loader:
            total_step += 1
            try:
                model.zero_grad()
                loss, kl_div, wacc, tacc, sacc = model(batch, beta)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
            except Exception as e:
                print(e)
                continue

            meters = meters + np.array([kl_div, wacc * 100, tacc * 100, sacc * 100])

            if total_step % print_iter == 0:
                meters /= print_iter
                print(
                    "[%d] Beta: %.3f, KL: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f"
                    % (
                        total_step,
                        beta,
                        meters[0],
                        meters[1],
                        meters[2],
                        meters[3],
                        param_norm(model),
                        grad_norm(model),
                    )
                )
                sys.stdout.flush()
                meters *= 0

            if total_step % anneal_iter == 0:
                scheduler.step()
                print("learning rate: %.6f" % scheduler.get_lr()[0])

            if total_step % kl_anneal_iter == 0 and total_step >= warmup:
                beta = min(max_beta, beta + step_beta)
            # torch.save(model.state_dict(), save_dir + "/model.epoch-" + str(epoch))
    torch.save(model.state_dict(), save_path)
    print("Model saved to: ", save_path)
    return model



if __name__ == "__main__":


    vocab_path = "../data/vocab_drd2_train_no_dot.txt"
    trained_jtvae_path = "jtvae_drd2_no_dot.pt"
    moses_preprocessed_dir = "../../JTVAE/JTVAE/FastJTNNpy3/fast_molvae/drd2_train_smiles_no_dot_moses-processed"

    vocab = [x.strip("\r\n ") for x in open(vocab_path)]
    vocab = Vocab(vocab)

    depthT = 20
    depthG = 3
    latent_size = 56
    hidden_size = 450

    torch_fix_seed()
    jtvae = JTNNVAE(vocab, hidden_size, latent_size, depthT, depthG)
    if os.path.exists(trained_jtvae_path):
        print(f"{trained_jtvae_path}はすでに存在しています。")

    else:
        print("JT-VAEの学習を開始します。")
        jtvae = fit(
            device=device,
            model=jtvae,
            moses_dir=moses_preprocessed_dir,
            vocab=vocab,
            save_path=trained_jtvae_path,
            batch_size=2,
            beta=0,
            lr=1e-3, 
            warmup=20000, 
            anneal_iter=20000,
            kl_anneal_iter=1000,
        )
        print(f"JT-VAEとして学習したものを{trained_jtvae_path}に保存します。")
        torch.save(jtvae.state_dict(), trained_jtvae_path)
