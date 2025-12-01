import json

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from subword_nmt.apply_bpe import BPE
import codecs

def process_smi(smi):

    if ';' in smi:
        smi = smi.split(';')[0].split(';')[0].split(';')[0]
    m = Chem.MolFromSmiles(smi)
    remover = SaltRemover()

    try:
        moll, deleted = remover.StripMolWithDeleted(m)
        Chem.AddHs(moll)
    except Exception as e:
        print('except:',e)

    canonical_smi = Chem.MolToSmiles(moll)
    return canonical_smi

def drug2emb_encoder(smile):
        vocab_path = './data/raw_data/drug_substructure/drug_codes.txt'
        sub2idx = np.load('./data/raw_data/drug_substructure/sub2idx.npy', allow_pickle=True).item()
        bpe_codes_drug = codecs.open(vocab_path)
        dbpe = BPE(bpe_codes_drug, merges=-1, separator='')

        max_d = 50
        try:
            t1 = dbpe.process_line(smile).split()
        except:
            print('this molecular is error:')
            print(smile)

        try:
            i1 = np.asarray([sub2idx[i] for i in t1])
        except:
            i1 = np.array([0])

        l = len(i1)
        if l < max_d:
            i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
            input_mask = ([1] * l) + ([0] * (max_d - l))
        else:
            i = i1[:max_d]
            input_mask = [1] * max_d

        return i, np.asarray(input_mask)

# Obtain SMILES of drugs and convert them to canonical SMILES using the Rdkit package
input_file = "/home/dell/disks/lsq/MILSyn-main/case_study_unique_smiles.json" #drugcomb
with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
drug_smiles = [process_smi(smi) for _,smi in data.items()]
drug_smiles = np.unique(drug_smiles)
# print(drug_smiles)
drugSmile_drugSubEmbed = { smi[1]:drug2emb_encoder(k) for k,smi in zip(drug_smiles,data.items())}
print(drugSmile_drugSubEmbed)
np.save('./data/1_drug_data/case_drugSubEmbed.npy', drugSmile_drugSubEmbed)