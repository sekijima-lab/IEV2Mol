# 最終的なinteraction出力から，各タイトルについて最良のdocking scoreを持つもののみを残す

import pandas as pd
import sys

path = smiles = sys.argv[1] # interactionファイルを指定
data = pd.read_csv(path)

# titleが重複するものがある場合は，最後のカラムの値（docking score）が最も良い（＝より負の値の）ものを残す
print("before: data length = ", len(data))
data = data.sort_values("docking_score")
data = data.drop_duplicates("# title", keep="first")

identity = path.split("/")[-1].split(".")[0] # 元となったファイル名の，拡張子より前の部分を取得
new_path = identity + "_max.interaction"
print("after: data length = ", len(data))
print(f"IEV is saved at {new_path} ")
data.to_csv(new_path, index=False, sep=",")
