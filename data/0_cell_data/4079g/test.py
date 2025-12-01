
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
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
os.chdir(sys.path[0])
sys.path.append(
    "/home/huyongfei/PycharmProjects/PublicCode/self_reproduce/NLP/CIForm_MAE")
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"

from MAE_modal import MaskedAutoencoderViT  # NOQA: E402
# from torch.utils.tensorboard import SummaryWriter


def find_all_ele_in_list(lst, ele):
    name = lst
    first_pos = 0
    res = []
    if ele not in lst:
        return []
    for i in range(name.count(ele)):
        new_list = name[first_pos:]
        next_pos = new_list.index(ele) + 1
        res.append(first_pos + new_list.index(ele))
        first_pos += next_pos
    return res


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


def getData(feature_M: np.array, gap: int = 512):
    single_cell_list = []
    epsilon = 1e-8  # 防止除零的极小值
    for single_cell in feature_M:
        feature = []
        length = len(single_cell)
        for k in range(0, length, gap):
            if (k + gap > length):
                a = np.pad(single_cell[k:], (0, gap - (length - k)), mode='constant')  # 用0填充
            else:
                a = single_cell[k:k + gap]

            # 1. 中心化（减去均值）
            mean_a = np.mean(a, axis=0)
            centered_a = a - mean_a

            # 2. 计算标准差，处理接近0的情况
            std_a = np.std(centered_a, axis=0, ddof=1)  # ddof=1 计算无偏标准差
            std_a = np.maximum(std_a, epsilon)  # 避免标准差为0

            # 3. 标准化（仅在标准差有效时进行，或始终标准化）
            scaled_a = centered_a / std_a  # 用处理后的标准差

            feature.append(scaled_a)

        feature = np.asarray(feature)
        single_cell_list.append(feature)

    single_cell_list = np.asarray(single_cell_list)  # (n_cells, gap_num, gap)
    return single_cell_list


def main(args):
    same_seeds(32)
    device = args.device
    omics_data = np.load(
        "/data/lsq/MILSyn-main/data/0_cell_data/4079g/985_cellGraphs_exp_mut_cn_eff_dep_met_4079_genes_norm.npy",allow_pickle=True).item()
    print(type(omics_data))
    array_list = [arr for arr in omics_data.values()]
    stacked_data = np.stack(array_list, axis=0)
    out = torch.tensor(stacked_data,dtype=torch.float16)
    omics_data = out[:, :, args.omics_idx:args.omics_idx + 1].squeeze()
    print(omics_data.shape)
    # omics_data  = np.asarray(omics_data)
    output_dir = f"./output{args.omics_idx}"
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "weights"), exist_ok=True)

    tb_writer = SummaryWriter(logdir=os.path.join(output_dir, "logs"))
    pd  = torch.zeros(985,512*8 - 4079,dtype=torch.float16)
    train_input_M = torch.cat((omics_data,pd), dim=1).view(985,-1,512)

    train_dataset = MyDataset(train_input_M)
    test_dataset = train_dataset
    batch_size = args.batch_size
    # number of workers
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw,
                                               )

    val_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=nw,
                                             )

    model = MaskedAutoencoderViT(sequence_length=8,
                                 embed_dim=512, depth=4, num_heads=16,
                                 decoder_embed_dim=256, decoder_depth=2, decoder_num_heads=16,
                                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False).to(device)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)

    def lf(x): return ((1 + math.cos(x * math.pi / args.epochs)) / 2) * \
        (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    statistics = []
    train_loss_list = []
    val_loss_list = []
    best_val_loss = 10000000

    for epoch in range(args.epochs):
        # train
        train_loss = train_one_epoch(model=model,
                                     optimizer=optimizer,
                                     data_loader=train_loader,
                                     device=device,
                                     epoch=epoch)

        scheduler.step()

        # validate
        val_loss = evaluate(model=model,
                            data_loader=val_loader,
                            device=device,
                            epoch=epoch)

        tags = ["train_loss", "val_loss", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], val_loss, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        if best_val_loss > val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(output_dir, "weights", "model_checkpoint_best.pth"))

        statistics.append({"epoch": epoch,
                           "train_loss": train_loss,
                           "val_loss": val_loss,
                           "learning_rate": optimizer.param_groups[0]["lr"]})
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        if (epoch+1) % 50 == 0:
            torch.save(model.state_dict(), os.path.join(
                output_dir, "weights", "model_checkpoint_epoch{0}.pth".format(epoch)))
    write_pickle(statistics, os.path.join(output_dir, "train_statistics.pkl"))

    N = np.arange(0, args.epochs)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, np.array(train_loss_list), label="train_loss")
    plt.plot(N, np.array(val_loss_list), label="val_loss")
    plt.title("training Loss")
    plt.xlabel("Epochs #")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "train_loss_plot.pdf"))
    print("\n")
    print("Best Model Val loss: {0:.3f}".format(best_val_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--omics_idx', type=int, default=0)

    parser.add_argument('--embedding_dim', type=int, default=512)

    parser.add_argument('--test_dataset_name', type=str, default="TCGA-TGCT")
    parser.add_argument('--data_path', type=str,
                        default="/home/huyongfei/PycharmProjects/PublicCode/self_reproduce/NLP/CIForm_MAE/datasets/TCGA")

    parser.add_argument('--device', default='cuda:0',
                        help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
