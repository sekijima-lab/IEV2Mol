import os

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

# make_csv.pyで作られたcsvファイルから，メトリクスを計算する
test_len = 10
model_dict = {}

# テストごとにみていく
for i in range(test_len):
    print("test", i)
    dir = f"results/test{i}/raw_csv"

    ifprnn_tmp = {"smiles_valid": 0, "iev_valid": 0, "cos_07": 0, "uniqueness": 0, "diversity": 0, "cos_08": 0, "docking_6": 0, "docking_7": 0}
    num_ifprnn_pose = 0
    # 各テスト内のモデルごとにみていく
    for file in sorted(os.listdir(dir)):
        if file.endswith(".csv"):
            model_name = file.split(".")[0]

            if "ifp-rnn" in model_name:
                model_name = "ifp-rnn"

            if model_name not in model_dict.keys():
                print("model_dictに", model_name, "を追加しました")
                # 全体の平均をとる
                model_dict[model_name] ={}
                model_dict[model_name]['MEAN'] = {"smiles_valid": 0, "iev_valid": 0, "cos_07": 0, "uniqueness": 0, "diversity": 0, "cos_08": 0, "docking_6": 0, "docking_7": 0}
            
            # テストデータごとの情報を残す
            model_dict[model_name][i] = {"smiles_valid": 0, "iev_valid": 0, "cos_07": 0, "uniqueness": 0, "diversity": 0, "cos_08": 0, "docking_6": 0, "docking_7": 0}

            df = pd.read_csv(f"{dir}/{file}")

            valid_smiles = [smiles for smiles in df["smiles"] if smiles is not np.nan and Chem.MolFromSmiles(smiles)is not None]
            print(f" {model_name} の有効なSMILES数: {len(valid_smiles)}")

            valid_iev_num = len(df.dropna(subset=["ievcos"]))
            print(f" {model_name} のIEVが有効なSMILES数: {valid_iev_num}")
            
            uniqueness = len(set(valid_smiles))/len(valid_smiles)
            print(f" {model_name} のUniqueness: {uniqueness}")

            mols = [Chem.MolFromSmiles(smile) for smile in valid_smiles]
            fps = [AllChem.GetMorganFingerprintAsBitVect(m,2) for m in mols]
            sum = 0
            for j in range(len(fps)):
                tanimotos = DataStructs.BulkTanimotoSimilarity(fps[j],fps)
                sum += np.sum(tanimotos)

            diversity = 1 - sum/(len(fps)**2)
            print(f" {model_name} のDiversity: {diversity}")
            
            docking_score_list = df["dscore"].tolist()
            cos_sim_list = df["ievcos"].tolist()

            if model_name == "ifp-rnn":
                num_ifprnn_pose += 1
                ifprnn_tmp["smiles_valid"] += len(valid_smiles)
                ifprnn_tmp["iev_valid"] += valid_iev_num
                ifprnn_tmp["cos_07"] += len([cos_sim for cos_sim in cos_sim_list if cos_sim >= 0.7])
                ifprnn_tmp["uniqueness"] += uniqueness
                ifprnn_tmp["diversity"] += diversity
                ifprnn_tmp["cos_08"] += len([cos_sim for cos_sim in cos_sim_list if cos_sim >= 0.8])
                ifprnn_tmp["docking_6"] += len([docking_score for docking_score in docking_score_list if docking_score <= -6])
                ifprnn_tmp["docking_7"] += len([docking_score for docking_score in docking_score_list if docking_score <= -7])
            else:
                model_dict[model_name][i] = {}
                model_dict[model_name][i]["smiles_valid"] = len(valid_smiles)
                model_dict[model_name][i]["iev_valid"] = valid_iev_num
                model_dict[model_name][i]["cos_07"] = len([cos_sim for cos_sim in cos_sim_list if cos_sim >= 0.7])
                model_dict[model_name][i]["uniqueness"] = uniqueness
                model_dict[model_name][i]["diversity"] = diversity
                model_dict[model_name][i]["cos_08"] = len([cos_sim for cos_sim in cos_sim_list if cos_sim >= 0.8])
                model_dict[model_name][i]["docking_6"] = len([docking_score for docking_score in docking_score_list if docking_score <= -6])
                model_dict[model_name][i]["docking_7"] = len([docking_score for docking_score in docking_score_list if docking_score <= -7])
            
                model_dict[model_name]['MEAN']["smiles_valid"] += model_dict[model_name][i]["smiles_valid"]/test_len
                model_dict[model_name]['MEAN']["iev_valid"] += model_dict[model_name][i]["iev_valid"]/test_len
                model_dict[model_name]['MEAN']["cos_07"] += model_dict[model_name][i]["cos_07"]/test_len
                model_dict[model_name]['MEAN']["uniqueness"] += model_dict[model_name][i]["uniqueness"]/test_len
                model_dict[model_name]['MEAN']["diversity"] += model_dict[model_name][i]["diversity"]/test_len
                model_dict[model_name]['MEAN']["cos_08"] += model_dict[model_name][i]["cos_08"]/test_len
                model_dict[model_name]['MEAN']["docking_6"] += model_dict[model_name][i]["docking_6"]/test_len
                model_dict[model_name]['MEAN']["docking_7"] += model_dict[model_name][i]["docking_7"]/test_len

    model_dict["ifp-rnn"][i] = {}
    model_dict["ifp-rnn"][i]["smiles_valid"] = ifprnn_tmp["smiles_valid"]/num_ifprnn_pose
    model_dict["ifp-rnn"][i]["iev_valid"] = ifprnn_tmp["iev_valid"]/num_ifprnn_pose
    model_dict["ifp-rnn"][i]["cos_07"] = ifprnn_tmp["cos_07"]/num_ifprnn_pose
    model_dict["ifp-rnn"][i]["uniqueness"] = ifprnn_tmp["uniqueness"]/num_ifprnn_pose
    model_dict["ifp-rnn"][i]["diversity"] = ifprnn_tmp["diversity"]/num_ifprnn_pose
    model_dict["ifp-rnn"][i]["cos_08"] = ifprnn_tmp["cos_08"]/num_ifprnn_pose
    model_dict["ifp-rnn"][i]["docking_6"] = ifprnn_tmp["docking_6"]/num_ifprnn_pose
    model_dict["ifp-rnn"][i]["docking_7"] = ifprnn_tmp["docking_7"]/num_ifprnn_pose

    model_dict["ifp-rnn"]['MEAN']["smiles_valid"] += model_dict["ifp-rnn"][i]["smiles_valid"]/test_len
    model_dict["ifp-rnn"]['MEAN']["iev_valid"] += model_dict["ifp-rnn"][i]["iev_valid"]/test_len
    model_dict["ifp-rnn"]['MEAN']["cos_07"] += model_dict["ifp-rnn"][i]["cos_07"]/test_len
    model_dict["ifp-rnn"]['MEAN']["uniqueness"] += model_dict["ifp-rnn"][i]["uniqueness"]/test_len
    model_dict["ifp-rnn"]['MEAN']["diversity"] += model_dict["ifp-rnn"][i]["diversity"]/test_len
    model_dict["ifp-rnn"]['MEAN']["cos_08"] += model_dict["ifp-rnn"][i]["cos_08"]/test_len
    model_dict["ifp-rnn"]['MEAN']["docking_6"] += model_dict["ifp-rnn"][i]["docking_6"]/test_len
    model_dict["ifp-rnn"]['MEAN']["docking_7"] += model_dict["ifp-rnn"][i]["docking_7"]/test_len


models = ["iev2mol", "jt-vae", "ifp-rnn", "random_chembl33"]
japanese_models = ["IEV2Mol", "JT-VAE", "IFP-RNN", "Random_ChEMBL"]

print(", Validity, Uniqueness, Diversity, ドッキング可能数, IEVCos類似度≥0.7\n")
for test_index in range(10):
    print(f"test{test_index}")
    for model, japanese_model in zip(models, japanese_models):
        print(f"{japanese_model}, {model_dict[model][test_index]['smiles_valid']}, {model_dict[model][test_index]['uniqueness']}, {model_dict[model][test_index]['diversity']}, {model_dict[model][test_index]['iev_valid']}, {model_dict[model][test_index]['cos_07']}, ")
    print()
print()

print('MEAN')
for model, japanese_model in zip(models, japanese_models):
    print(f"{japanese_model}, {model_dict[model]['MEAN']['smiles_valid']}, {model_dict[model]['MEAN']['uniqueness']}, {model_dict[model]['MEAN']['diversity']}, {model_dict[model]['MEAN']['iev_valid']}, {model_dict[model]['MEAN']['cos_07']}")