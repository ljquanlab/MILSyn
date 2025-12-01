import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os


def kshot_finetuning_data_split(npy_file_path, csv_file_path, target_tissue, kshot_sequence,include_base_data=False, random_state=42):
    """
    K-shot微调数据划分：
    1. 从目标tissue中抽取最大样本数作为微调池
    2. 将剩余样本均分为测试集和验证集
    3. 其他tissue的数据作为基础训练集
    4. 按照kshot序列逐步向训练集加入目标tissue样本

    参数:
    npy_file_path: npy文件路径
    csv_file_path: 包含细胞系信息的csv文件路径
    target_tissue: 目标组织类型
    kshot_sequence: kshot序列，如[0, 2, 4, 8, 16, 32, 64, 128, 256]
    random_state: 随机种子

    返回:
    包含不同阶段数据集的字典
    """

    # 设置随机种子
    random.seed(random_state)
    np.random.seed(random_state)

    # 1. 加载数据
    print("正在加载数据...")
    data = np.load(npy_file_path, allow_pickle=True)
    cell_line_info = pd.read_csv(csv_file_path)

    # 2. 创建细胞系ID到组织类型的映射
    cell_line_to_tissue = dict(zip(cell_line_info['depmap_id'],
                                   cell_line_info['sample_collection_site']))

    # 3. 为每个样本添加组织类型
    tissues = []
    for i in range(len(data)):
        cell_id = data[i, 2]
        tissue = cell_line_to_tissue.get(cell_id, 'Unknown')
        tissues.append(tissue)

    # 4. 分离目标组织类型和其他组织类型的样本
    target_indices = [i for i, tissue in enumerate(tissues) if tissue == target_tissue]
    other_indices = [i for i, tissue in enumerate(tissues) if tissue != target_tissue]

    print(f"目标组织类型 '{target_tissue}' 的样本数: {len(target_indices)}")
    print(f"其他组织类型的样本数: {len(other_indices)}")
    print(f"总样本数: {len(data)}")

    # 5. 获取最大kshot值
    max_kshot = max(kshot_sequence)

    if max_kshot > len(target_indices):
        print(f"警告: 最大kshot值 {max_kshot} 大于目标组织类型的样本数 {len(target_indices)}")
        print(f"将使用所有可用样本: {len(target_indices)}")
        max_kshot = len(target_indices)
        # 调整kshot序列，过滤掉超过最大样本数的值
        kshot_sequence = [k for k in kshot_sequence if k <= max_kshot]

    # 6. 从目标tissue中抽取微调池（最大kshot数量的样本）
    target_data = data[target_indices]
    finetune_pool_indices = np.random.choice(
        len(target_data),
        size=max_kshot,
        replace=False
    )
    finetune_pool = target_data[finetune_pool_indices]

    # 剩余的目标tissue样本
    remaining_indices = np.setdiff1d(np.arange(len(target_data)), finetune_pool_indices)
    remaining_target_data = target_data[remaining_indices]

    print(f"\n微调池大小: {len(finetune_pool)}")
    print(f"剩余目标样本数: {len(remaining_target_data)}")

    # 7. 将剩余目标样本均分为测试集和验证集
    test_data, val_data = train_test_split(
        remaining_target_data,
        test_size=0.5,
        random_state=random_state
    )

    # 8. 基础训练集（不包含目标tissue）
    base_train_data = data[other_indices]

    print(f"\n初始划分:")
    print(f"基础训练集大小: {len(base_train_data)} (不包含 {target_tissue})")
    print(f"验证集大小: {len(val_data)} (全部为 {target_tissue})")
    print(f"测试集大小: {len(test_data)} (全部为 {target_tissue})")
    print(f"微调池大小: {len(finetune_pool)} (用于逐步加入训练集)")

    # 9. 按照kshot序列逐步构建训练集
    results = {}
    start_fold = 120

    for i, kshot in enumerate(kshot_sequence):
        fold_number = start_fold + i

        if kshot == 0:
            # 0-shot必须使用基础数据（因为没有目标数据）
            current_train = base_train_data
            description = f'Fold {fold_number}: 0-shot (仅基础训练集)'
        else:
            kshot_samples = finetune_pool[:kshot]

            if include_base_data:
                # 策略A: 基础数据 + k-shot目标数据
                current_train = np.vstack([base_train_data, kshot_samples])
                description = f'Fold {fold_number}: {kshot}-shot (基础数据 + {kshot}目标样本)'
            else:
                # 策略B: 仅k-shot目标数据
                current_train = kshot_samples
                description = f'Fold {fold_number}: {kshot}-shot (仅{kshot}目标样本)'

        # 保存当前阶段的数据
        results[f'{fold_number}'] = {
            'train': current_train.copy(),
            'val': val_data.copy(),
            'test': test_data.copy(),
            'description': description,
            'kshot': kshot,
            'fold_number': fold_number,
            'include_base_data': include_base_data if kshot > 0 else True  # 0-shot总是包含基础数据
        }

    return results


def save_datasets(results, output_dir):
    """
    保存每个阶段的数据集到文件，按照指定的命名格式

    参数:
    results: 包含各阶段数据集的字典
    output_dir: 输出目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for fold, data_dict in results.items():
        fold_number = data_dict['fold_number']
        # 保存训练集、验证集和测试集，使用指定的命名格式
        np.save(os.path.join(output_dir, f'{fold_number}_fold_tr_items.npy'), data_dict['train'])
        np.save(os.path.join(output_dir, f'{fold_number}_fold_val_items.npy'), data_dict['val'])
        np.save(os.path.join(output_dir, f'{fold_number}_fold_test_items.npy'), data_dict['test'])

    print(f"\n所有fold的数据已保存到: {output_dir}")


def create_experiment_summary(results, target_tissue, output_dir):
    """
    创建实验摘要文件

    参数:
    results: 包含各阶段数据集的字典
    target_tissue: 目标组织类型
    output_dir: 输出目录
    """
    summary_lines = []
    summary_lines.append("K-shot微调实验摘要")
    summary_lines.append("=" * 50)
    summary_lines.append(f"目标组织类型: {target_tissue}")
    summary_lines.append(f"总fold数: {len(results)}")
    summary_lines.append("")

    # 按fold编号排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]['fold_number'])

    for fold, data_dict in sorted_results:
        fold_number = data_dict['fold_number']
        kshot = data_dict['kshot']
        summary_lines.append(f"Fold {fold_number}: {data_dict['description']}")
        summary_lines.append(f"  训练集: {len(data_dict['train'])} 个样本")
        summary_lines.append(f"  验证集: {len(data_dict['val'])} 个样本")
        summary_lines.append(f"  测试集: {len(data_dict['test'])} 个样本")
        summary_lines.append("")

    summary_text = "\n".join(summary_lines)

    with open(os.path.join(output_dir, 'experiment_summary.txt'), 'w') as f:
        f.write(summary_text)

    print(summary_text)


# 使用示例
if __name__ == "__main__":
    # 设置参数
    npy_file = "/home/dell/disks/lsq/MILSyn-main/data/split/all_items.npy"
    csv_file = "/home/dell/disks/lsq/MILSyn-main/data/raw_data/cell_info.csv"
    target_tissue = "liver"
    kshot_sequence = [0, 2, 4, 8, 16]
    output_dir = "liver"

    try:
        # 执行K-shot微调数据划分
        results = kshot_finetuning_data_split(
            npy_file,
            csv_file,
            target_tissue,
            kshot_sequence,
            random_state=42
        )

        # 保存数据集
        save_datasets(results, output_dir)

        # 创建实验摘要
        create_experiment_summary(results, target_tissue, output_dir)

        print(f"\nK-shot微调实验完成! 结果保存在: {output_dir}")

    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
    except Exception as e:
        print(f"发生错误: {e}")