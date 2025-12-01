import torch.nn.functional as F
import os
import os.path as osp
import numpy as np
import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from tqdm import tqdm

from rdkit import Chem

from dataset.base_InMemory_dataset import BaseInMemoryDataset


class MyInMemoryDataset(BaseInMemoryDataset):
    def __init__(self,
                 data_root,
                 data_items,
                 celllines_data,
                 drugs_data,
                 dgi_data=None,
                 transform=None,
                 pre_transform=None,
                 args=None,
                 max_node_num=155):

        super(MyInMemoryDataset, self).__init__(root=data_root, transform=transform, pre_transform=pre_transform)

        if args.celldataset == 1:
            self.name = osp.basename(data_items).split('items')[0] + '18498g'
        elif args.celldataset == 2:
            self.name = osp.basename(data_items).split('items')[0] + '4079g'
        elif args.celldataset == 3:
            self.name = osp.basename(data_items).split('items')[0] + '963g'

        self.name = self.name + '_TransDrug_norm'

        if args.mode == 'infer':
            self.name = osp.basename(data_items).split('items')[0]

        self.args = args
        self.data_items = np.load(data_items, allow_pickle=True)
        self.celllines = np.load(celllines_data, allow_pickle=True).item()
        self.drugs = np.load(drugs_data, allow_pickle=True).item()
        if dgi_data:
            self.dgi = np.load(dgi_data, allow_pickle=True).item()
        else:
            self.dgi = {}
        self.max_node_num = max_node_num

        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process()
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [self.name + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self):
        data_list = []
        data_len = len(self.data_items)
        llm_cell_pt = torch.load("/home/dell/disks/lsq/MILSyn-main/json/cell_v3_features_dpsk.pt")
        llm_drug_pt = torch.load("/home/dell/disks/lsq/MILSyn-main/json/drug_v3_features_dpsk.pt")
        graph = torch.load("/home/dell/disks/lsq/MILSyn-main/dataset/pt_data/all_graph.pt")
        unimol_cls = torch.load("/home/dell/disks/lsq/MILSyn-main/dataset/pt_data/all_drug_cls_unimol2_570m.pt")
        omic0 = torch.load("/home/dell/disks/lsq/MILSyn-main/data/0_cell_data/4079g/pt_data/omic_0.pt")
        omic1 = torch.load("/home/dell/disks//lsq/MILSyn-main/data/0_cell_data/4079g/pt_data/omic_1.pt")
        omic2 = torch.load("/home/dell/disks/lsq/MILSyn-main/data/0_cell_data/4079g/pt_data/omic_2.pt")
        omic3 = torch.load("/home/dell/disks/lsq/MILSyn-main/data/0_cell_data/4079g/pt_data/omic_3.pt")
        omic4 = torch.load("/home/dell/disks/lsq/MILSyn-main/data/0_cell_data/4079g/pt_data/omic_4.pt")
        omic5 = torch.load("/home/dell/disks/lsq/MILSyn-main/data/0_cell_data/4079g/pt_data/omic_5.pt")
        for i in tqdm(range(data_len)):
            drugA, drugB, c1, label = self.data_items[i]
            cell_features = self.celllines[c1]
            cell_omic0 = omic0[c1]
            cell_omic1 = omic1[c1]
            cell_omic2 = omic2[c1]
            cell_omic3 = omic3[c1]
            cell_omic4 = omic4[c1]
            cell_omic5 = omic5[c1]
            drugA_features = self.drugs[drugA]
            drugB_features = self.drugs[drugB]
            cell_drug_data = Data()

            smiA, mask_smiA = smiles2onehot(drugA)
            smiB, mask_smiB = smiles2onehot(drugB)

            cell_drug_data.smiA = torch.Tensor(smiA).to(dtype=torch.long)
            cell_drug_data.smiB = torch.Tensor(smiB).to(dtype=torch.long)  # torch.Size([179])
            cell_drug_data.smiA_mask = get_smi_mask(torch.Tensor(mask_smiA)).to(dtype=torch.float32)
            cell_drug_data.smiB_mask = get_smi_mask(torch.Tensor(mask_smiB)).to(
                dtype=torch.float32)  # torch.Size([1, 1, 179])

            da = torch.Tensor(np.array([drugA_features]))  # torch.Size([1, 2, 50])
            db = torch.Tensor(np.array([drugB_features]))
            cell_drug_data.drugA = da[:, 0, :].to(dtype=torch.long)  # torch.Size([1,50])
            cell_drug_data.drugB = db[:, 0, :].to(dtype=torch.long)
            cell_drug_data.drugA_mask = get_smi_mask(da[:, 1, :]).to(dtype=torch.float32)
            cell_drug_data.drugB_mask = get_smi_mask(db[:, 1, :]).to(dtype=torch.float32)  # torch.Size([1, 1, 1, 50])
            cell_fea = torch.as_tensor(cell_features).to(dtype=torch.float32).transpose(0, 1)  # 6,4079
            cell_drug_data.x_cell = cell_fea

            x_cell2 = torch.stack((cell_omic0, cell_omic1, cell_omic2, cell_omic3, cell_omic4, cell_omic5), dim=0).to(
                dtype=torch.float32)

            cell_drug_data.y = torch.Tensor([float(label)]).to(dtype=torch.float32)
            cell_drug_data.graph = self.merge_graphs_optimized(graph[drugA], graph[drugB], x_cell2)

            cell_drug_data.clsA = torch.Tensor(unimol_cls[drugA]).to(dtype=torch.float32)
            cell_drug_data.clsB = torch.Tensor(unimol_cls[drugB]).to(dtype=torch.float32)
            cell_drug_data.llm_cell_feature = llm_cell_pt[c1]
            cell_drug_data.llmA = llm_drug_pt[drugA].to(dtype=torch.float32)
            cell_drug_data.llmB = llm_drug_pt[drugB].to(dtype=torch.float32)

            data_list.append(cell_drug_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        print('Dataset construction done.')

    def merge_graphs(self, graph1: Data, graph2: Data) -> Data:
        # 1. 合并节点特征
        n1, n2 = graph1.num_nodes, graph2.num_nodes
        x = torch.cat([graph1.x, graph2.x], dim=0)
        # 2. 合并原始边（向量化操作）
        edge_index1 = graph1.edge_index
        edge_index2 = graph2.edge_index + n1
        edge_index = torch.cat([edge_index1, edge_index2], dim=1)
        c1 = graph1.center_node
        c2 = graph2.center_node + n1  # 调整graph2中心节点索引
        # 生成中心节点和虚拟节点索引
        existing1 = torch.tensor([c1], dtype=torch.long)
        existing2 = torch.tensor([c2], dtype=torch.long)
        # 生成虚拟节点索引
        # 构建连接边（双向）
        src = torch.cat([
            existing1.repeat(1),  # 现有节点→虚拟节点
            existing2.repeat(1)  # 虚拟节点→现有节点
        ])
        dst = torch.cat([
            existing2.repeat(1),
            existing1.repeat(1)
        ])

        edge_index = torch.cat([edge_index, torch.stack([src, dst])], dim=1)

        pd = torch.zeros(2, 7, dtype=torch.long)
        edge_attr = torch.cat([
            graph1.edge_attr,
            graph2.edge_attr,
            pd
        ])


        assert edge_index.size(1) == edge_attr.size(0)
        # print(edge_index)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def merge_graphs_optimized(self,graph1: Data, graph2: Data, virtual_feat: Tensor) -> Data:
        # 1. 合并节点特征
        n1, n2 = graph1.num_nodes, graph2.num_nodes
        x = torch.cat([graph1.x, graph2.x, virtual_feat], dim=0)
        # 2. 合并原始边（向量化操作）
        edge_index1 = graph1.edge_index
        edge_index2 = graph2.edge_index + n1
        edge_index = torch.cat([edge_index1, edge_index2], dim=1)
        # 3. 生成虚拟节点连接边（无循环批量生成）
        num_virtual = virtual_feat.size(0)
        total_nodes = n1 + n2
        # 生成现有节点索引
        c1 = graph1.center_node
        c2 = graph2.center_node + n1  # 调整graph2中心节点索引
        # 生成中心节点和虚拟节点索引
        existing1 = torch.tensor([c1], dtype=torch.long)
        existing2 = torch.tensor([c2], dtype=torch.long)
        # 生成虚拟节点索引
        virtual_ids = torch.arange(total_nodes, total_nodes + num_virtual)
        # 构建连接边（双向）
        src = torch.cat([
            existing1.repeat(num_virtual),  # 现有节点→虚拟节点
            virtual_ids.repeat_interleave(1)  # 虚拟节点→现有节点
        ])
        dst = torch.cat([
            virtual_ids.repeat_interleave(1),
            existing1.repeat(num_virtual)
        ])
        src2 = torch.cat([
            existing2.repeat(num_virtual),  # 现有节点→虚拟节点
            virtual_ids.repeat_interleave(1)  # 虚拟节点→现有节点
        ])
        dst2 = torch.cat([
            virtual_ids.repeat_interleave(1),
            existing2.repeat(num_virtual)
        ])
        # 组学相似边
        sim_edge = SimGraphConstruction(2)(virtual_feat)  # [2,  num_virtual * 2]
        sim_edge = sim_edge + (n1 + n2)
        # 合并所有边
        edge_index = torch.cat([edge_index, torch.stack([src, dst]), torch.stack([src2, dst2]), sim_edge], dim=1)

        pd = torch.zeros(num_virtual * 2 * 2 + num_virtual * 2, 7, dtype=torch.long)
        at1 = torch.ones(num_virtual * 2, 1, dtype=torch.long)
        at2 = torch.full((num_virtual * 2, 1), 2, dtype=torch.long)
        at3 = torch.full((num_virtual * 2, 1), 3, dtype=torch.long)
        at = torch.cat((at1, at2, at3), dim=0)
        pdat = torch.cat([pd, at], dim=1)

        g1 = torch.full((graph1.edge_index.size(1), 1), 4, dtype=torch.long)
        g2 = torch.zeros(graph2.edge_index.size(1), 1, dtype=torch.long)
        edge_attr = torch.cat([
            torch.cat([graph1.edge_attr, g1], dim=1),
            torch.cat([graph2.edge_attr, g2], dim=1),
            pdat
        ])

        at1 = torch.ones(num_virtual * 2, dtype=torch.long)
        at2 = torch.full((num_virtual * 2,), 2, dtype=torch.long)
        at3 = torch.full((num_virtual * 2,), 3, dtype=torch.long)
        at4 = torch.full((graph1.edge_index.size(1),), 4, dtype=torch.long)
        at5 = torch.zeros(graph2.edge_index.size(1), dtype=torch.long)

        edge_type = torch.cat([at4, at5, at1, at2, at3], dim=0)
        assert edge_index.size(1) == edge_attr.size(0)
        # print(edge_index)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_type=edge_type)

def convert_to_onehot(original_tensor, onehot_length=2586):
    onehot_tensor = torch.zeros(onehot_length, dtype=torch.int32)

    # 遍历原始tensor中的每个值
    for value in original_tensor:
        # 确保值是整数且在有效范围内
        idx = int(value.item())
        if 0 <= idx < onehot_length:
            onehot_tensor[idx] = 1

    return onehot_tensor

def get_smi_mask(v):# seq
    subs_mask = v.long()
    expanded_subs_mask = subs_mask.unsqueeze(0).unsqueeze(1)  # torch.Size([1, 1, seq])
    expanded_subs_mask = (1.0 - expanded_subs_mask) * -10000.0
    return expanded_subs_mask.float()

class SimGraphConstruction(nn.Module):
    def  __init__(self, k):
        super(SimGraphConstruction, self).__init__()
        self.k = k
    def forward(self, feature):
        sim = feature / (torch.norm(feature, dim=-1, keepdim=True) + 1e-10)
        sim = torch.mm(sim, sim.T)
        diag = torch.diag(sim)
        diag = torch.diag_embed(diag)
        sim = sim - diag
        tmp = torch.topk(sim, k=self.k, dim=1)
        row_indices = torch.arange(sim.shape[0]).unsqueeze(1).repeat(1, self.k).flatten()
        col_indices = tmp.indices.flatten()
        edge_index = torch.stack([row_indices, col_indices], dim=0).long()
        return edge_index

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

def smiles2onehot(smiles, MAX_SMI_LEN=179):
    x = np.zeros(MAX_SMI_LEN)
    lenx = len(smiles)
    if lenx > MAX_SMI_LEN:
        lenx = MAX_SMI_LEN
        # print("err")
    mask_x = ([1] * lenx) + ([0] * (MAX_SMI_LEN - lenx))
    for i, ch in enumerate(smiles[:MAX_SMI_LEN]):
        x[i] = CHARISOSMISET[ch]

    return torch.tensor(x),torch.tensor(mask_x)
