import argparse
import random

import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch

def same_seeds(seed):
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(args):
    same_seeds(32)
    device = args.device
    omics_data = np.load(
        "/data/lsq/MILSyn-main/data/0_cell_data/4079g/985_cellGraphs_exp_mut_cn_eff_dep_met_4079_genes_norm.npy",allow_pickle=True).item()
    print(type(omics_data))
    array_list = [arr for arr in omics_data.values()]
    stacked_data = np.stack(array_list, axis=0)
    out = torch.tensor(stacked_data, dtype=torch.float32)
    omics_data = out[:, :, args.omics_idx:args.omics_idx + 1].squeeze()
    print(omics_data.shape)  # torch.Size([985, 4079])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--omics_idx', type=int, default=4)
    parser.add_argument('--device', default='cuda:0',
                        help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)

