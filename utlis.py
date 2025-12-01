import os
import os.path as osp
import random
import numpy as np
import torch
from mmcv.utils import collect_env as collect_base_env
from torch.utils.data import DataLoader
from dataset.My_inMemory_dataset import MyInMemoryDataset
from metrics import get_metrics
from tqdm import tqdm
from torch_geometric.data.batch import Batch


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def set_random_seed(seed, deterministic=True):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EarlyStopping():
    def __init__(self, mode='higher', patience=50, filename=None, metric=None, n_fold=None, folder=None):
        """
        Initialize EarlyStopping object.

        Args:
            mode (str): 'higher' if a higher score is better, 'lower' if a lower score is better.
            patience (int): Number of epochs to wait for improvement before early stopping.
            filename (str): Name of the checkpoint file to save the model state.
            metric (str): Metric to monitor for early stopping. Can be 'r2', 'mae', 'rmse', 'roc_auc_score', 'pr_auc_score', or 'mse'.
            n_fold (int): Fold number used for naming checkpoint file.
            folder (str): Folder path to save checkpoint file.
        """

        if filename is None:
            filename = os.path.join(folder, '{}_fold_early_stop.pth'.format(n_fold))

        if metric is not None:
            assert metric in ['r2', 'mae', 'rmse', 'roc_auc_score', 'pr_auc_score', 'mse'], \
                "Expect metric to be 'r2' or 'mae' or " \
                "'rmse' or 'roc_auc_score' or 'mse', got {}".format(metric)
            if metric in ['r2', 'roc_auc_score', 'pr_auc_score']:
                print('For metric {}, the higher the better'.format(metric))
                mode = 'higher'
            if metric in ['mae', 'rmse', 'mse']:
                print('For metric {}, the lower the better'.format(metric))
                mode = 'lower'

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        """
        Check if the new score is higher than the previous best score.
        """
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        """
        Check if the new score is lower than the previous best score.
        """
        return score < prev_best_score

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):

        model.load_state_dict(torch.load(self.filename))


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    return env_info


def collate_pyg(batch):
    smiA, smiB, smiA_mask, smiB_mask, drugAA, drugBB, drugA_mask, drugB_mask, x_cell, y, graph, clsA, clsB, llm_cell_feature,llmA,llmB = zip(
        *batch)

    graph_data_list = [g[1] for g in graph]
    graph_batch = Batch.from_data_list(graph_data_list)
    drugA_list = [g[1] for g in drugAA]
    drugA = torch.cat(drugA_list, dim=0)
    drugB_list = [g[1] for g in drugBB]
    drugB = torch.cat(drugB_list, dim=0)
    cell_list = [g[1] for g in x_cell]
    cell = torch.stack(cell_list, dim=0)
    y_list = [item[1] for item in y]
    y = torch.cat(y_list, dim=0)
    smiA_list = [item[1] for item in smiA]
    smiA = torch.stack(smiA_list, dim=0)
    smiB_list = [item[1] for item in smiB]
    smiB = torch.stack(smiB_list, dim=0)

    smiA_mask_list = [item[1] for item in smiA_mask]
    smiA_mask = torch.stack(smiA_mask_list, dim=0)
    smiB_mask_list = [item[1] for item in smiB_mask]
    smiB_mask = torch.stack(smiB_mask_list, dim=0)

    drugA_mask_list = [item[1] for item in drugA_mask]
    drugA_mask = torch.cat(drugA_mask_list, dim=0)
    drugB_mask_list = [item[1] for item in drugB_mask]
    drugB_mask = torch.cat(drugB_mask_list, dim=0)

    clsA_list = [item[1] for item in clsA]
    clsA = torch.stack(clsA_list, dim=0)
    clsB_list = [item[1] for item in clsB]
    clsB = torch.stack(clsB_list, dim=0)

    llm_cell_list = [item[1] for item in llm_cell_feature]
    llm_cell = torch.stack(llm_cell_list, dim=0)

    llm_a_list = [item[1] for item in llmA]
    llm_A = torch.stack(llm_a_list, dim=0)
    llm_b_list = [item[1] for item in llmB]
    llm_B = torch.stack(llm_b_list, dim=0)
    return drugA, drugB, cell, graph_batch, y, smiA, smiB, smiA_mask, smiB_mask, drugA_mask, drugB_mask, clsA, clsB, llm_cell,llm_A ,llm_B


def load_dataloader(n_fold, args):
    work_dir = args.workdir
    data_root = osp.join(work_dir, 'data')

    if args.celldataset == 1:
        celllines_data = osp.join(data_root, '0_cell_data/18498g/985_cellGraphs_exp_mut_cn_18498_genes_norm.npy')
    elif args.celldataset == 2:
        celllines_data = osp.join(data_root,
                                  '0_cell_data/4079g/985_cellGraphs_exp_mut_cn_eff_dep_met_4079_genes_norm.npy')
    elif args.celldataset == 3:
        celllines_data = osp.join(data_root, '0_cell_data/963g/985_cellGraphs_exp_mut_cn_963_genes_norm.npy')

    drugs_data = osp.join(data_root, '1_drug_data/drugSmile_drugSubEmbed_2644.npy')

    tr_data_items = osp.join(data_root, f'split/{n_fold}_fold_tr_items.npy')
    val_data_items = osp.join(data_root, f'split/{n_fold}_fold_val_items.npy')
    test_data_items = osp.join(data_root, f'split/{n_fold}_fold_test_items.npy')

    tr_dataset = MyInMemoryDataset(data_root, tr_data_items, celllines_data, drugs_data, args=args)
    val_dataset = MyInMemoryDataset(data_root, val_data_items, celllines_data, drugs_data, args=args)
    test_dataset = MyInMemoryDataset(data_root, test_data_items, celllines_data, drugs_data, args=args)

    tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True,
                               collate_fn=collate_pyg)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True,
                                collate_fn=collate_pyg)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True,
                                 collate_fn=collate_pyg)

    print(f'train data:{len(tr_dataloader) * args.batch_size}')
    print(f'Valid data:{len(val_dataloader) * args.batch_size}')
    print(f'Test data:{len(test_dataloader) * args.batch_size}')

    return tr_dataloader, val_dataloader, test_dataloader


def load_infer_dataloader(args):
    data_root = osp.join(args.workdir, 'data')
    if args.celldataset == 1:
        celllines_data = osp.join(data_root, '0_cell_data/18498g/985_cellGraphs_exp_mut_cn_18498_genes_norm.npy')
    elif args.celldataset == 2:
        celllines_data = osp.join(data_root, '0_cell_data/4079g/selected_celllines.npy')
    elif args.celldataset == 3:
        celllines_data = osp.join(data_root, '0_cell_data/963g/985_cellGraphs_exp_mut_cn_963_genes_norm.npy')
    drugs_data = osp.join(data_root, '1_drug_data/drugSmile_drugSubEmbed_2644.npy')

    data_items = args.infer_path
    infer_dataset = MyInMemoryDataset(data_root, data_items, celllines_data, drugs_data, args=args)
    infer_dataloader = DataLoader(infer_dataset, batch_size=1, shuffle=False, num_workers=4)

    infer_data_arr = np.load(data_items, allow_pickle=True)

    return infer_dataloader, infer_data_arr


from tqdm import tqdm
import torch


def train(model, criterion, opt, dataloader, device, args=None):
    model.train()
    # train_loss_sum = 0
    total_loss = []
    # 使用 tqdm 包装 dataloader，显示进度条
    for data in tqdm(dataloader, desc="Training Batch: "):
        opt.zero_grad()
        drugA, drugB, cell, graph, y, smiA, smiB, smiA_mask, smiB_mask, drugA_mask, drugB_mask, clsA, clsB, llm_cell,llm_A ,llm_B = data
        y = y.unsqueeze(1).to(device)
        drugA = drugA.to(device)
        drugB = drugB.to(device)
        cell = cell.to(device)
        graph = graph.to(device)
        smiA = smiA.to(device)
        smiB = smiB.to(device)
        smiA_mask = smiA_mask.to(device)
        smiB_mask = smiB_mask.to(device)
        drugA_mask = drugA_mask.to(device)
        drugB_mask = drugB_mask.to(device)
        clsA = clsA.to(device)
        clsB = clsB.to(device)
        llm_A = llm_A.to(device)
        llm_B = llm_B.to(device)
        llm_cell = llm_cell.to(device)
        output = model(drugA, drugB, cell, graph, smiA, smiB, smiA_mask, smiB_mask, drugA_mask,
                                     drugB_mask, clsA, clsB, llm_cell, llm_A, llm_B)

        train_loss_mse = criterion(output, y)
        #todo cls loss
        total_loss.append(train_loss_mse.item())
        train_loss_mse.backward()
        opt.step()
    return (sum(total_loss) / len(total_loss))


from math import sqrt


def validate(model, criterion, dataloader, device, args=None):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        # 使用 tqdm 包装 dataloader，显示进度条
        for data in tqdm(dataloader, desc="Validating", leave=False):
            # i += 1
            drugA, drugB, cell, graph, y, smiA, smiB, smiA_mask, smiB_mask, drugA_mask, drugB_mask, clsA, clsB, llm_cell,llm_A ,llm_B = data
            y = y.unsqueeze(1).to(device)
            drugA = drugA.to(device)
            drugB = drugB.to(device)
            cell = cell.to(device)
            graph = graph.to(device)
            smiA = smiA.to(device)
            smiB = smiB.to(device)
            smiA_mask = smiA_mask.to(device)
            smiB_mask = smiB_mask.to(device)
            drugA_mask = drugA_mask.to(device)
            drugB_mask = drugB_mask.to(device)
            clsA = clsA.to(device)
            clsB = clsB.to(device)
            llm_A = llm_A.to(device)
            llm_B = llm_B.to(device)
            llm_cell = llm_cell.to(device)
            y_true.append(y.view(-1, 1))
            # output, _, _ = model(x)
            output = model(drugA, drugB, cell, graph, smiA, smiB, smiA_mask, smiB_mask, drugA_mask,
                                         drugB_mask, clsA, clsB, llm_cell, llm_A ,llm_B)

            y_pred.append(output)

    y_true = torch.cat(y_true, dim=0).cpu().detach().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().detach().numpy()
    mse, rmse, mae, r2, pearson, spearman = get_metrics(y_true, y_pred)

    return mse, rmse, mae, r2, pearson, spearman