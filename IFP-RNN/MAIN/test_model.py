
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import os
import sys
sys.path.append('../')
from ddc.ddc_pub import ddc_v3 as ddc
from rdkit import Chem

def cal_valid(smiList):
    total = len(smiList)
    valid = 0
    valSmis = []
    for smi in smiList:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid += 1
            valSmis.append(smi)
    valid_rate = valid/float(total+0.01)*100
    return valid_rate, valSmis


def prepare_input(ifp_df):
    ifp_df['index'] = ifp_df['Molecule']
    ifp_df = ifp_df.set_index('index')
    seedList = list(ifp_df['Molecule'])

    print(f'Number of seeds: {len(seedList)}')
    print(f'seed molecules: {seedList}')
    dfNew = ifp_df.loc[seedList]
    inputList = []
    colDrop = ['index', 'smi', 'Molecule', 'logP', 'QED', 'SA',
               'Wt', 'NP', 'score_0', 'TPSA', 'MW', 'HBA', 'HBD', 'QED']
    for i in range(1024):
        colDrop.append(f'ecfp{i}')
    for idx, row in dfNew.iterrows():
        smi = row['smi']
        molID = row['Molecule']
        IFP = row.copy()
        row = row.drop(['smi', 'Molecule'])
        '''Get a clean IFP without other informations!'''
        for colName in colDrop:
            try:
                IFP = IFP.drop([colName])
            except Exception as e:
                print(e)
                continue
        row = np.array(row)
        IFP = np.array(IFP)
        # row=add_random(row)
        inputDic = {'smi': smi, 'molID': molID, 'row': row, 'IFP': IFP}
        inputList.append(inputDic)
        print(f'smi:{smi} molID:{molID} row:{row}')
    return inputList


def generate_smis(args):
    IFP_Df = pd.read_csv(args.IFP)
    inputList = prepare_input(IFP_Df)
    model_name = args.model
    print(model_name)
    model = ddc.DDC(model_name=model_name)
    temp = 0.5

    os.system(f'mkdir sampled_100smiles')
    opFileName = f'sampled_100smiles/{args.save}_temp_{temp}'
    opFile = open(opFileName + '.smi', 'w')
    opFile.writelines('SMILES\tName\n')
    for inputDic in inputList:
        seedSmi = inputDic['smi']
        molID = inputDic['molID']
        IFP = inputDic['IFP']
        opFile.writelines(f'{seedSmi}\tSeed\n')   # file head
        

        model.batch_input_length = 100 #生成する数
        IFP = np.array([IFP]*model.batch_input_length)
        # print(IFP)
        print(f'Sampling for molecule: {molID}')

        smiles, _ = model.predict_batch(latent=IFP, temp=temp)
        smiles = list(smiles)

        validity, valSmis = cal_valid(smiles)

        for idx, smi in enumerate(valSmis):
                    opFile.writelines(
                        f'{smi}\t{molID}_sampled{idx}\n')
                    
        print(f'validity: {validity}')
    opFile.close()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--IFP", help="IFP file to obtained seed randomly",
                        default='')
    parser.add_argument("--seed", help='csv file of seeds',
                        default='')
    parser.add_argument("--model", help="trained model",
                        default='')
    parser.add_argument("--save", help="file name for saving sampled SMILES (pkl)",
                        default='')
    parser.add_argument("--label", help="label of the result file",
                        type=str, default='sample')
    parser.add_argument("--switch_pt", help="the percentage of bits will be switched.",
                        type=float, default='0')
    args = parser.parse_args()
    return args


def main(args):
    savePath = Path(args.save)
    savePath.parent.mkdir(parents=True, exist_ok=True)

    generate_smis(args)

    df = pd.read_csv("./sampled_100smiles/generated100smi_temp_0.5.smi", sep="\t")
    start = 0
    for i in range(16):
        end = start+1
        while df.iloc[end].iloc[1]!="Seed" and end<len(df)-1:
            end = end+1
        df[start:end].to_csv(f"./sampled_100smiles/{df.iloc[start+1].iloc[1][:-9]}.smi", header=0, index=None, sep=" ")
        start = end



if __name__ == "__main__":
    args = get_parser()
    main(args)
