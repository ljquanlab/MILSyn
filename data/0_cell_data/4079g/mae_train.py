
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

from MAE_modal import MaskedAutoencoderViT  # NOQA: E402
# from torch.utils.tensorboard import SummaryWriter

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
    out = torch.tensor(stacked_data,dtype=torch.float32)
    omics_data = out[:, :, args.omics_idx:args.omics_idx + 1].squeeze()
    print(omics_data.shape)#torch.Size([985, 4079])
    # omics_data  = np.asarray(omics_data)
    output_dir = f"./output{args.omics_idx}"
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "weights"), exist_ok=True)

    tb_writer = SummaryWriter(logdir=os.path.join(output_dir, "logs"))
    omics_data = omics_data[:,:4056]
    train_input_M = omics_data.view(985,-1,78)
    print(train_input_M.shape)
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

    model = MaskedAutoencoderViT(sequence_length=52,
                                 embed_dim=78, depth=4, num_heads=13,
                                 decoder_embed_dim=78 * 2, decoder_depth=2, decoder_num_heads=13,
                                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False).to(device)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr, weight_decay=0.)

    def lf(x): return ((1 + math.cos(x * math.pi / args.epochs)) / 2) * \
        (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    statistics = []
    train_loss_list = []
    val_loss_list = []
    best_val_loss = 10000000
    down_count = 0
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
        down_count = down_count + 1
        if down_count > args.patience:
            print("early stopping.....")
            break
        if best_val_loss > val_loss:
            down_count = 0
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(output_dir, "weights", "model_checkpoint_best.pth"))

        statistics.append({"epoch": epoch,
                           "train_loss": train_loss,
                           "val_loss": val_loss,
                           "learning_rate": optimizer.param_groups[0]["lr"]})
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        # if (epoch+1) % 50 == 0:
        #     torch.save(model.state_dict(), os.path.join(
        #         output_dir, "weights", "model_checkpoint_epoch{0}.pth".format(epoch)))
    write_pickle(statistics, os.path.join(output_dir, "train_statistics.pkl"))

    N = np.arange(0, len(train_loss_list))
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N , np.array(train_loss_list), label="train_loss")
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

    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--omics_idx', type=int, default=5)
    parser.add_argument('--device', default='cuda:0',
                        help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--patience', type=int, default=20)
    opt = parser.parse_args()

    main(opt)
