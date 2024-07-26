import pandas as pd

df = pd.read_csv("IFP_ResIFP.csv", sep=",", header=0)

test_name = ["CHEMBL4467359", "CHEMBL932", "CHEMBL3220212", "CHEMBL3323114", "CHEMBL4451683", "CHEMBL1221498", "CHEMBL5172201", "CHEMBL201878", "CHEMBL3104520", "CHEMBL107973"]

# dfと同じ行数で、全ての要素がFalseのSeriesを作成
match = pd.Series([False] * len(df))

for name in test_name:
    match = match | df["Molecule"].str.startswith(name + "_")

test_df = df[match]
train_df = df[~match]

test_df.to_csv("IFP_ResIFP_test.csv", index=False)
train_df.to_csv("IFP_ResIFP_train.csv", index=False)

print(f"df: {len(df)}")
print(f"test_df: {len(test_df)}")
print(f"test_df_mol: {test_df['Molecule']}")
print(f"train_df: {len(train_df)}")
