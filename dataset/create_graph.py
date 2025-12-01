import json
from itertools import islice
import numpy
import numpy as np
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data
from rdkit.Chem.rdchem import BondType as BT
BOND_LIST = [
    BT.SINGLE,
    BT.DOUBLE,
    BT.TRIPLE,
    BT.AROMATIC
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]
import torch

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])
def bond_features(bond):
    return np.array(
        one_of_k_encoding(bond.GetBondType(), BOND_LIST) + one_of_k_encoding(bond.GetBondDir(), BONDDIR_LIST)
    )
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))
def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile2graph_pyg(smile):
    mol = Chem.MolFromSmiles(smile)

    features = []
    atom_num = 0
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        # print(feature / sum(feature))
        features.append(feature / sum(feature))

    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        feature = bond_features(bond)
        edge_feat.append(feature)
        edge_feat.append(feature)
    edge_index = torch.tensor(numpy.array([row, col]), dtype=torch.long)
    center_node = find_central_node_pyg(edge_index)
    center_node = torch.tensor(np.array(center_node,dtype=np.int32), dtype=torch.int32)
    edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
    x = torch.tensor(numpy.array(features,dtype = np.float32), dtype=torch.float)
    # x_feature = torch.tensor(x_feature, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, center_node=center_node)
    return data
from unimol_tools import UniMolRepr
def extract_graph_from_csv():
    drug_graph = {}
    input_file = "/data/lsq/MILSyn-main/case_study_unique_smiles.json" #drugcomb
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    clf = UniMolRepr(data_type='molecule',
                     remove_hs=True,
                     model_name='unimolv2',  # avaliable: unimolv1, unimolv2
                     model_size='570m',  # work when model_name is unimolv2. avaliable: 84m, 164m, 310m, 570m, 1.1B.
                     use_gpu='cuda:1',
                     batch_size=8
                     )

    smiles_list = data
    # print(data)

    smiles_list = [
        Chem.MolToSmiles(Chem.MolFromSmiles(k), isomericSmiles=True)
        for _,k in data.items()
    ]
    unimol_repr = clf.get_repr(smiles_list, return_atomic_reprs=True)
    unimol_repr = [torch.from_numpy(repr) for repr in unimol_repr['cls_repr']]
    print(len(data))
    print(len(unimol_repr))
    drug_cls = {}
    for ori,smi,features in zip(data.items(),smiles_list,unimol_repr):
        # print(type(ori))
        ori = ori[1]
        print(are_smiles_identical(ori,smi))
        graph = smile2graph_pyg(smi)
        drug_graph[ori] = graph
        drug_cls[ori] = features
        # print(graph)
        # print(features.shape)
    return drug_graph,drug_cls

import torch
from torch_geometric.data import Data
import networkx as nx
from rdkit import Chem


def are_smiles_identical(smiles1, smiles2):
    # 将SMILES转换为分子对象
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    # 检查转换是否成功（无效SMILES会返回None）
    if mol1 is None or mol2 is None:
        return False

    # 标准化分子（去除冗余信息，统一表示方式）
    mol1.Compute2DCoords()
    mol2.Compute2DCoords()

    # 比较分子指纹或结构
    return Chem.MolToSmiles(mol1) == Chem.MolToSmiles(mol2)

def find_central_node_pyg(edge_index):
    # 转换为NetworkX图
    edge_index = edge_index.cpu().numpy()
    G = nx.Graph()
    G.add_edges_from(edge_index.T.tolist())  # 边索引需转置为(edges, 2)

    # 计算介数中心性
    betweenness = nx.betweenness_centrality(G)

    # 找到中心节点
    if not betweenness:
        return 0  # 空图处理
    center_node = max(betweenness, key=betweenness.get)
    return center_node

def load_smiles_dict(file_path):
    smiles_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                smiles = parts[0]
                drug_name = parts[1]
                smiles_dict[drug_name] = smiles
    return smiles_dict
if __name__ == "__main__":
    drug_graph,drug_cls = extract_graph_from_csv()
    # 保存指纹张量到 pt 文件
    torch.save(drug_graph, '../dataset/pt_data/case_graph.pt')
    torch.save(drug_cls, '../dataset/pt_data/case_drug_cls_unimol2_570m.pt')
