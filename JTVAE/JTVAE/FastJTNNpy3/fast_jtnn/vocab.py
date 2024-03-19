import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import copy
import numpy as np


def get_slots(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return [
        (atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalNumHs())
        for atom in mol.GetAtoms()
    ]


class Vocab(object):
    benzynes = [
        "C1=CC=CC=C1",
        "C1=CC=NC=C1",
        "C1=CC=NN=C1",
        "C1=CN=CC=N1",
        "C1=CN=CN=C1",
        "C1=CN=NC=N1",
        "C1=CN=NN=C1",
        "C1=NC=NC=N1",
        "C1=NN=CN=N1",
    ]
    penzynes = [
        "C1=C[NH]C=C1",
        "C1=C[NH]C=N1",
        "C1=C[NH]N=C1",
        "C1=C[NH]N=N1",
        "C1=COC=C1",
        "C1=COC=N1",
        "C1=CON=C1",
        "C1=CSC=C1",
        "C1=CSC=N1",
        "C1=CSN=C1",
        "C1=CSN=N1",
        "C1=NN=C[NH]1",
        "C1=NN=CO1",
        "C1=NN=CS1",
        "C1=N[NH]C=N1",
        "C1=N[NH]N=C1",
        "C1=N[NH]N=N1",
        "C1=NN=N[NH]1",
        "C1=NN=NS1",
        "C1=NOC=N1",
        "C1=NON=C1",
        "C1=NSC=N1",
        "C1=NSN=C1",
    ]

    def __init__(self, smiles_list):
        self.vocab = smiles_list
        self.ecfp2s = []
        self.num_atoms = []
        for vo in self.vocab:
            mol = Chem.MolFromSmiles(vo)
            self.ecfp2s.append(
                AllChem.GetMorganFingerprintAsBitVect(mol, 1, nBits=1024)
            )
            self.num_atoms.append(mol.GetNumAtoms())
        self.vmap = {x: i for i, x in enumerate(self.vocab)}
        self.slots = [get_slots(smiles) for smiles in self.vocab]
        Vocab.benzynes = [
            s
            for s in smiles_list
            if s.count("=") >= 2 and Chem.MolFromSmiles(s).GetNumAtoms() == 6
        ] + ["C1=CCNCC1"]
        Vocab.penzynes = [
            s
            for s in smiles_list
            if s.count("=") >= 2 and Chem.MolFromSmiles(s).GetNumAtoms() == 5
        ] + ["C1=NCCN1", "C1=NNCC1"]

    def get_index(self, smiles):
        # ifの部分は勝手に足した．原子数差が小さい方からTanimoto類似度の最大値を見ていき，0.6以上のやつを返す
        if smiles not in self.vmap:
            mol = Chem.MolFromSmiles(smiles)
            num_atom = mol.GetNumAtoms()
            unknown_ecfp2 = AllChem.GetMorganFingerprintAsBitVect(mol, 1, nBits=1024)
            tanimotos = DataStructs.BulkTanimotoSimilarity(unknown_ecfp2, self.ecfp2s)
            diff_num_atoms = np.abs(np.array(self.num_atoms) - num_atom)

            sorted_by_diff_num_atoms = sorted(
                zip(diff_num_atoms, tanimotos, self.vocab)
            )
            sorted_diff_num_atoms, sorted_tanimotos, sorted_vocab = zip(
                *sorted_by_diff_num_atoms
            )

            diff_list = sorted(list(set(sorted_diff_num_atoms)))
            start_idx = 0
            max_tanimoto = 0
            for diff in diff_list:
                end_idx = start_idx + 1
                while end_idx < len(sorted_diff_num_atoms):
                    if sorted_diff_num_atoms[end_idx] != diff:
                        break
                    end_idx += 1
                # print("diff: ", diff, "start_idx: ", start_idx, "end_idx: ", end_idx)
                tmp_max_tanimoto = max(sorted_tanimotos[start_idx:end_idx])
                # print("tmp_max_tanimoto: ", tmp_max_tanimoto)
                if max_tanimoto < tmp_max_tanimoto:
                    # print("max_tanimoto < tmp_max_tanimoto")
                    max_tanimoto = tmp_max_tanimoto
                    # print("max_tanimoto: ", max_tanimoto)
                    if max_tanimoto >= 0.6:
                        # print("max_tanimoto >= 0.6")
                        break
                start_idx = end_idx

            max_tanimoto_smi_idx = sorted_tanimotos.index(max_tanimoto)
            max_tanimoto_smi = sorted_vocab[max_tanimoto_smi_idx]
            print("Change: ", smiles, " -> ", max_tanimoto_smi)
            print("Tanimoto: ", sorted_tanimotos[max_tanimoto_smi_idx])
            print("Diff num atoms: ", sorted_diff_num_atoms[max_tanimoto_smi_idx])
            return self.vmap[max_tanimoto_smi]

        return self.vmap[smiles]

    def get_smiles(self, idx):
        return self.vocab[idx]

    def get_slots(self, idx):
        return copy.deepcopy(self.slots[idx])

    def size(self):
        return len(self.vocab)
