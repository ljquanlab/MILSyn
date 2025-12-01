import numpy as np
import os
from sklearn.model_selection import KFold


def split_cv_data(data_path, output_dir, n_folds=5):
    """
    划分药物协同预测数据为五折交叉验证集

    参数:
    data_path: 原始数据npy文件路径
    output_dir: 输出文件夹路径
    n_folds: 交叉验证的折数，默认为5
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    data = np.load(data_path, allow_pickle=True)

    # 假设数据结构是结构化数组，包含'smilesA', 'smilesB', 'cell', 'S_score'字段
    # 提取所有唯一的细胞系
    unique_cells = np.unique(data[:, 2])
    print(f"总共有 {len(unique_cells)} 种独特的细胞系")

    # 初始化K折交叉验证器
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # 对细胞系进行K折划分
    for fold, (train_val_index, test_index) in enumerate(kf.split(unique_cells)):
        print(f"正在处理第 {fold} 折...")

        # 划分训练+验证细胞系和测试细胞系
        train_val_cells = unique_cells[train_val_index]
        test_cells = unique_cells[test_index]

        # 从训练+验证细胞系中再划分训练集和验证集(按6:2比例，即3:1)
        val_size = int(len(train_val_cells) * 0.25)  # 25% of train_val is 20% of total
        val_cells = train_val_cells[:val_size]
        train_cells = train_val_cells[val_size:]

        # 根据细胞系划分数据
        train_mask = np.isin(data[:, 2], train_cells)
        val_mask = np.isin(data[:, 2], val_cells)
        test_mask = np.isin(data[:, 2], test_cells)

        train_data = data[train_mask]
        val_data = data[val_mask]
        test_data = data[test_mask]

        # 打印各数据集大小和比例
        total = len(train_data) + len(val_data) + len(test_data)
        print(f"第 {fold} 折数据分布:")
        print(f"训练集: {len(train_data)} 样本 ({len(train_data) / total:.2%})")
        print(f"验证集: {len(val_data)} 样本 ({len(val_data) / total:.2%})")
        print(f"测试集: {len(test_data)} 样本 ({len(test_data) / total:.2%})")

        # 检查细胞系是否有重叠
        train_unique = set(np.unique(train_data[:, 2]))
        val_unique = set(np.unique(val_data[:, 2]))
        test_unique = set(np.unique(test_data[:, 2]))

        assert train_unique.isdisjoint(val_unique), "训练集和验证集存在重叠的细胞系"
        assert train_unique.isdisjoint(test_unique), "训练集和测试集存在重叠的细胞系"
        assert val_unique.isdisjoint(test_unique), "验证集和测试集存在重叠的细胞系"

        # 保存数据
        # np.save(os.path.join(output_dir, f"4{fold}_fold_tr_items.npy"), train_data)
        # np.save(os.path.join(output_dir, f"4{fold}_fold_val_items.npy"), val_data)
        # np.save(os.path.join(output_dir, f"4{fold}_fold_test_items.npy"), test_data)

    print("五折交叉验证数据划分完成!")


# 使用示例
if __name__ == "__main__":
    # 请替换为实际的数据路径和输出目录
    data_path = "all_items.npy"  # 输入数据路径
    output_dir = "./"  # 输出目录

    # 执行划分
    split_cv_data(data_path, output_dir)
