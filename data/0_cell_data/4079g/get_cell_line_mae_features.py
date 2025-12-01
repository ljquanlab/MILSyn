from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from utils import read_split_data, train_one_epoch, evaluate, write_pickle
from utils import MyDataset
import pickle as pkl
import random
from torchvision import transforms
from tensorboardX import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import argparse
import math
import os
import sys
from MAE_modal import MaskedAutoencoderViT

if __name__ == '__main__':
    device = "cuda:0"
    omics_data = np.load(
        "/data/lsq/MILSyn-main/data/0_cell_data/4079g/985_cellGraphs_exp_mut_cn_eff_dep_met_4079_genes_norm.npy",
        allow_pickle=True).item()
    labels = [arr for arr in omics_data.keys()]
    print(labels)
    # array_list = [arr for arr in omics_data.values()]
    # stacked_data = np.stack(array_list, axis=0)
    # out = torch.as_tensor(stacked_data).to(dtype=torch.float16)
    for omics_idx in range(0,6):
        print(omics_idx)
        model = MaskedAutoencoderViT(sequence_length=52,
                                     embed_dim=78, depth=4, num_heads=13,
                                     decoder_embed_dim=78 * 2, decoder_depth=2, decoder_num_heads=13,
                                     mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False).to(device)
        check_point_path = f"/data/lsq/MILSyn-main/data/0_cell_data/4079g/output{omics_idx}/weights/model_checkpoint_best.pth"
        check_point = torch.load(check_point_path, map_location="cpu")
        model.load_state_dict(check_point, strict=True)
        model = model.to(device)
        model.eval()
        valid_output_embedding = {}
        with torch.no_grad():
            for cell in labels:
                print(cell)
                in_feature = omics_data[cell]
                in_feature = in_feature[:,omics_idx:omics_idx + 1].squeeze()
                in_feature = torch.tensor(in_feature,dtype=torch.float32)
                # in_feature = np.array(in_feature)
                # # print(type(in_feature))
                # in_feature = getData(in_feature)
                # # print(in_feature.shape)
                in_feature = in_feature[ :4056]
                in_feature = in_feature.view(-1, 78).to(device)
                # print(type(in_feature))
                latent, mask, ids_restore = model.forward_encoder(in_feature, mask_ratio=0.0)
                # print(latent)
                # print(latent.shape)
                valid_output_embedding[cell] = latent[:, 0, :].squeeze().cpu().detach()
                # print(latent.shape)
                # print(valid_output_embedding[cell].shape)
                # print(valid_output_embedding[cell])
        torch.save(valid_output_embedding, f'./pt_data/omic_{omics_idx}.pt')