import torch

class Smiles_Vector_Dataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list, vector_list):
        super().__init__()
        self.smiles = smiles_list
        self.vectors = vector_list

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx], self.vectors[idx]


'''
smiles_list = ["CCO", "CCN", "CCF"]
IEV_list = []
をSmiles_Vector_Dataset(smiles_list, IEV_list)としてインスタンス化してtorch.saveでダンプしたやつがdrd2_train_dataset_no_dot.ptとか

'''