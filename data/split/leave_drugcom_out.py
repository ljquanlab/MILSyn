import numpy as np
from sklearn.model_selection import KFold


def create_drug_pair_folds(data, n_splits=5, random_state=42):
    """
    为药物对协同预测任务创建5折交叉验证划分，保持原始数据格式不变

    参数:
    data: numpy数组，格式为 [(smilesA, smilesB, cell, score), ...]
    n_splits: 折数 (默认5)
    random_state: 随机种子 (默认42)

    返回:
    folds: 包含5个字典的列表，每个字典包含:
        {'train': 训练集数据, 'val': 验证集数据, 'test': 测试集数据}
    """
    # 1. 创建标准化的药物对标识（排序SMILES，避免(A,B)和(B,A)被视为不同对）
    drug_pairs = [tuple(sorted([smilesA, smilesB])) for smilesA, smilesB, _, _ in data]

    # 2. 创建唯一药物对列表
    unique_pairs = list(set(drug_pairs))

    # 3. 创建药物对到样本索引的映射
    pair_to_samples = {}
    for idx, pair in enumerate(drug_pairs):
        if pair not in pair_to_samples:
            pair_to_samples[pair] = []
        pair_to_samples[pair].append(idx)

    # 4. 准备KFold划分
    unique_pair_indices = np.arange(len(unique_pairs))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    folds = []

    # 5. 进行双重交叉验证划分
    for fold, (train_val_idx, test_idx) in enumerate(kf.split(unique_pair_indices)):
        # 5.1 划分测试集药物对
        test_pairs = [unique_pairs[i] for i in test_idx]

        # 5.2 划分非测试集药物对（用于训练和验证）
        non_test_pairs = [unique_pairs[i] for i in train_val_idx]
        non_test_indices = np.arange(len(non_test_pairs))

        # 5.3 在非测试集上划分验证集
        kf_val = KFold(n_splits=n_splits, shuffle=True, random_state=random_state + fold)
        # 取第一个验证折
        train_inner_idx, val_inner_idx = next(kf_val.split(non_test_indices))

        val_pairs = [non_test_pairs[i] for i in val_inner_idx]
        train_pairs = [non_test_pairs[i] for i in train_inner_idx]

        # 5.4 收集样本数据（保持原始格式）
        train_data = []
        for pair in train_pairs:
            for idx in pair_to_samples[pair]:
                # 保持原始数据格式 (smilesA, smilesB, cell, score)
                train_data.append(tuple(data[idx]))

        val_data = []
        for pair in val_pairs:
            for idx in pair_to_samples[pair]:
                val_data.append(tuple(data[idx]))

        test_data = []
        for pair in test_pairs:
            for idx in pair_to_samples[pair]:
                test_data.append(tuple(data[idx]))

        # 5.5 转换为numpy数组保持与原始数据一致格式
        train_data = np.array(train_data, dtype=data.dtype)
        val_data = np.array(val_data, dtype=data.dtype)
        test_data = np.array(test_data, dtype=data.dtype)

        # 5.6 创建折叠字典（包含实际数据）
        fold_dict = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        folds.append(fold_dict)

    return folds


# 使用示例
if __name__ == "__main__":
    # 1. 加载数据 (假设数据是npy格式)
    data = np.load('all_items.npy', allow_pickle=True)
    # 2. 创建5折划分
    folds = create_drug_pair_folds(data, n_splits=5)

    # 3. 验证划分结果
    for fold_idx, fold_data in enumerate(folds):
        print(f"\nFold {fold_idx + 1}:")
        print(f"  Train samples: {len(fold_data['train'])}")
        print(f"  Val samples: {len(fold_data['val'])}")
        print(f"  Test samples: {len(fold_data['test'])}")
        np.save(f'3{fold_idx}_fold_tr_items.npy', fold_data['train'])  # leave drug comb out
        np.save(f'3{fold_idx}_fold_val_items.npy', fold_data['test'])
        np.save(f'3{fold_idx}_fold_test_items.npy', fold_data['val'])
        # 获取各集合中的药物对
        train_pairs = set(tuple(sorted((d1, d2)))
                          for d1,d2,c,s in fold_data['train'])
        val_pairs = set(tuple(sorted((d1, d2)))
                        for d1,d2,c,s in fold_data['val'])
        test_pairs = set(tuple(sorted((d1, d2)))
                         for d1,d2,c,s in fold_data['test'])

        # 验证无重叠
        assert len(val_pairs & train_pairs) == 0, f"Fold {fold_idx + 1}: Val与Train有重叠药物对!"
        assert len(test_pairs & train_pairs) == 0, f"Fold {fold_idx + 1}: Test与Train有重叠药物对!"
        assert len(val_pairs & test_pairs) == 0, f"Fold {fold_idx + 1}: Val与Test有重叠药物对!"

        print(f"  Unique train pairs: {len(train_pairs)}")
        print(f"  Unique val pairs: {len(val_pairs)}")
        print(f"  Unique test pairs: {len(test_pairs)}")

        # 验证数据格式保持原样
        sample = fold_data['train'][0]
        assert len(sample) == 4, "数据格式错误!"
        assert isinstance(sample[0], str), "SMILES格式错误!"
        assert isinstance(sample[2], str), "细胞格式错误!"
        assert isinstance(sample[3], float), "分数格式错误!"

        print("  Validation passed: No overlapping drug pairs and format preserved!")

    # 4. 保存划分结果 (可选)
    # np.save('drug_pair_folds.npy', folds)
    print("\nAll folds validated successfully!")

# import random
# import numpy as np
# from collections import defaultdict
#
#
# def split_dataset_by_drug_combination(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
#     """
#     划分数据集，确保验证集和测试集中的药物组合在训练集中从未出现过
#     但单个药物可能在训练集中出现
#
#     参数:
#         data: 原始数据集，格式为 [(smilesA, smilesB, cell, score), ...]
#         train_ratio: 训练集比例
#         val_ratio: 验证集比例
#         test_ratio: 测试集比例
#         seed: 随机种子
#
#     返回:
#         train_set: 训练集
#         val_set: 验证集
#         test_set: 测试集
#     """
#     # 设置随机种子
#     random.seed(seed)
#     np.random.seed(seed)
#
#     # 1. 提取所有唯一的药物组合（无序对）
#     drug_pairs = defaultdict(list)
#     pair_to_samples = defaultdict(list)
#
#     for d in data:
#         smilesA, smilesB, cell, score = d
#         # 创建标准化的药物对标识（无序）
#         pair_key = tuple(sorted([smilesA, smilesB]))
#         drug_pairs[pair_key].append(d)
#         pair_to_samples[pair_key].append(d)
#
#     # 获取所有唯一的药物组合
#     unique_pairs = list(drug_pairs.keys())
#     random.shuffle(unique_pairs)
#
#     # 2. 划分药物组合
#     total_pairs = len(unique_pairs)
#     num_train = int(total_pairs * train_ratio)
#     num_val = int(total_pairs * val_ratio)
#     num_test = total_pairs - num_train - num_val
#
#     # 3. 分配药物组合到不同集合
#     train_pairs = set(unique_pairs[:num_train])
#     val_pairs = set(unique_pairs[num_train:num_train + num_val])
#     test_pairs = set(unique_pairs[num_train + num_val:])
#
#     # 4. 构建数据集
#     train_set = []
#     val_set = []
#     test_set = []
#
#     # 用于检查单个药物是否在训练集中出现
#     train_drugs = set()
#
#     # 添加训练集样本并收集训练集药物
#     for pair in train_pairs:
#         train_set.extend(pair_to_samples[pair])
#         # 收集训练集中的药物
#         train_drugs.add(pair[0])
#         train_drugs.add(pair[1])
#
#     # 添加验证集样本
#     for pair in val_pairs:
#         val_set.extend(pair_to_samples[pair])
#
#     # 添加测试集样本
#     for pair in test_pairs:
#         test_set.extend(pair_to_samples[pair])
#
#     # 5. 打乱各集合中的样本顺序
#     random.shuffle(train_set)
#     random.shuffle(val_set)
#     random.shuffle(test_set)
#
#     # 6. 验证划分结果
#     # 检查验证集和测试集中的药物组合是否在训练集中出现
#     for pair in val_pairs:
#         if pair in train_pairs:
#             raise ValueError(f"验证集药物组合 {pair} 出现在训练集中")
#
#     for pair in test_pairs:
#         if pair in train_pairs:
#             raise ValueError(f"测试集药物组合 {pair} 出现在训练集中")
#
#     # 检查单个药物是否可能在训练集中出现
#     val_test_drugs = set()
#     for pair in val_pairs | test_pairs:
#         val_test_drugs.add(pair[0])
#         val_test_drugs.add(pair[1])
#
#     # 计算在训练集中出现的单个药物比例
#     overlap_drugs = val_test_drugs & train_drugs
#     overlap_ratio = len(overlap_drugs) / len(val_test_drugs) if val_test_drugs else 0
#
#     print(f"验证集和测试集中有 {len(overlap_drugs)}/{len(val_test_drugs)} 个药物在训练集中出现 ({overlap_ratio:.2%})")
#
#     return train_set, val_set, test_set
#
#
# if __name__ == "__main__":
#     # 假设 data 是原始数据集，格式: [(smilesA, smilesB, cell, score), ...]
#     data = np.load('all_items.npy', allow_pickle=True)
#     train, val, test = split_dataset_by_drug_combination(data)
#
#     print(f"总样本数: {len(data)}")
#     print(f"训练集大小: {len(train)} ({len(train) / len(data):.2%})")
#     print(f"验证集大小: {len(val)} ({len(val) / len(data):.2%})")
#     print(f"测试集大小: {len(test)} ({len(test) / len(data):.2%})")
#
#     # 检查测试集中的药物组合是否在训练集中出现
#     train_pairs = set()
#     for d in train:
#         smilesA, smilesB, _, _ = d
#         pair_key = tuple(sorted([smilesA, smilesB]))
#         train_pairs.add(pair_key)
#
#     test_pairs = set()
#     for d in test:
#         smilesA, smilesB, _, _ = d
#         pair_key = tuple(sorted([smilesA, smilesB]))
#         test_pairs.add(pair_key)
#
#     overlap = train_pairs & test_pairs
#     if overlap:
#         print(f"警告: {len(overlap)} 个药物组合同时出现在训练集和测试集中")
#     else:
#         print("验证成功: 测试集中的所有药物组合在训练集中均未出现")
#
#     np.save('1_fold_tr_items.npy', train)
#     np.save('1_fold_val_items.npy', val)
#     np.save('1_fold_test_items.npy', test)