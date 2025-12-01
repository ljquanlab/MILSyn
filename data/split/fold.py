import numpy as np
from sklearn.model_selection import KFold, train_test_split

def kfold_split(data, k, test_size=0.2, val_size=0.2, random_state=None):
    splits = []
    # 设置 KFold 的分割次数为 1 / test_size 的倒数，确保测试集比例正确
    kf = KFold(n_splits=int(1 / test_size), shuffle=True, random_state=random_state)

    for fold, (train_val_indices, test_indices) in enumerate(kf.split(data)):
        # 划分测试集
        test_data = data[test_indices]

        # 从训练+验证集中划分验证集和训练集
        train_val_data = data[train_val_indices]
        # 验证集比例为 val_size，直接使用 train_test_split 的 test_size 参数
        train_data, val_data = train_test_split(train_val_data, test_size=0.25, random_state=random_state)

        # 保存当前划分
        splits.append({
            "train": train_data,
            "val": val_data,
            "test": test_data
        })

        print(f"Fold {fold + 1}/{kf.n_splits}:")
        print(f"  Train size: {len(train_data)}")
        print(f"  Val size: {len(val_data)}")
        print(f"  Test size: {len(test_data)}")

    return splits

# 主函数
def main(input_file, k, test_size=0.1, val_size=0.1, random_state=None):
    # 加载数据
    data = np.load(input_file, allow_pickle=True)
    print(f"加载数据完成，总样本数: {len(data)}")

    # 进行 k-fold 划分
    splits = kfold_split(data, k=k, test_size=test_size, val_size=val_size, random_state=random_state)

    # 保存每次划分的结果
    for i, split in enumerate(splits):
        np.save(f"1{i}_fold_tr_items.npy", split["train"], allow_pickle=True)
        np.save(f"1{i}_fold_val_items.npy", split["val"], allow_pickle=True)
        np.save(f"1{i}_fold_test_items.npy", split["test"], allow_pickle=True)
        print(f"Fold {i} 的划分结果已保存到文件中。")

# 示例用法
input_file = "/home/dell/disks/lsq/MILSyn-main/data/independent dataset/indep2-OncologyScreen/all_items.npy"  # 输入文件路径
k = 5
main(input_file, k=k, test_size=0.2, val_size=0.2, random_state=42)