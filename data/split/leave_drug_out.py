import numpy as np
from collections import defaultdict
import random
import os


def load_data(file_path):
    """加载npy文件"""
    return np.load(file_path, allow_pickle=True)


def get_unique_drugs(data):
    """获取所有独特的药物"""
    drugs_a = set(data[:, 0])
    drugs_b = set(data[:, 1])
    return drugs_a.union(drugs_b)


def split_data_with_novel_drug_condition(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    """
    划分数据，确保验证集和测试集中的每条数据有且仅有一个新药
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "比例之和必须为1"

    # 获取所有独特药物
    all_drugs = get_unique_drugs(data)
    n_drugs = len(all_drugs)

    # 随机选择一部分药物作为训练集中出现的药物
    random.seed(seed)
    train_drugs = set(random.sample(list(all_drugs), int(n_drugs * train_ratio)))
    remaining_drugs = all_drugs - train_drugs

    # 将剩余药物分为验证集和测试集使用的药物
    val_test_drugs = list(remaining_drugs)
    random.shuffle(val_test_drugs)

    split_idx = int(len(val_test_drugs) * (val_ratio / (val_ratio + test_ratio)))
    val_drugs = set(val_test_drugs[:split_idx])
    test_drugs = set(val_test_drugs[split_idx:])

    # 根据药物分配情况划分数据
    train_data = []
    val_data = []
    test_data = []

    for item in data:
        drug_a, drug_b, cell, score = item
        in_train_a = drug_a in train_drugs
        in_train_b = drug_b in train_drugs

        # 两个药物都在训练集中 -> 训练集
        if in_train_a and in_train_b:
            train_data.append(item)
        # 只有一个药物在训练集中
        elif (in_train_a and not in_train_b) or (not in_train_a and in_train_b):
            # 确定另一个药物属于验证集还是测试集
            novel_drug = drug_b if in_train_a else drug_a

            if novel_drug in val_drugs:
                val_data.append(item)
            elif novel_drug in test_drugs:
                test_data.append(item)
        # 两个药物都不在训练集中 -> 丢弃或重新分配
        else:
            # 随机分配给验证集或测试集，但确保只有一个新药的条件
            # 这里我们选择丢弃这些数据，因为它们不符合我们的条件
            pass

    return np.array(train_data), np.array(val_data), np.array(test_data), train_drugs, val_drugs, test_drugs


def five_fold_cross_validation(data, output_dir, seed=42):
    """执行五折交叉验证"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有独特药物
    all_drugs = get_unique_drugs(data)
    n_drugs = len(all_drugs)
    drug_list = list(all_drugs)

    # 设置随机种子以确保可重复性
    random.seed(seed)
    random.shuffle(drug_list)

    # 将药物分为5折
    fold_size = n_drugs // 5
    drug_folds = [drug_list[i * fold_size:(i + 1) * fold_size] for i in range(5)]

    # 处理最后一折可能多出的药物
    if n_drugs % 5 != 0:
        drug_folds[-1].extend(drug_list[5 * fold_size:])

    # 进行五折交叉验证
    for fold in range(5):
        print(f"处理第 {fold} 折...")

        # 测试集药物是当前折的药物
        test_drugs = set(drug_folds[fold])

        # 剩余药物用于训练和验证
        remaining_drugs = set(drug_list) - test_drugs
        remaining_drugs_list = list(remaining_drugs)
        random.shuffle(remaining_drugs_list)

        # 将剩余药物分为训练集和验证集药物
        n_remaining = len(remaining_drugs_list)
        split_idx = int(n_remaining * 0.75)  # 训练集占75%，验证集占25%的剩余药物

        train_drugs = set(remaining_drugs_list[:split_idx])
        val_drugs = set(remaining_drugs_list[split_idx:])

        # 根据药物分配情况划分数据
        train_data = []
        val_data = []
        test_data = []

        for item in data:
            drug_a, drug_b, cell, score = item
            in_train_a = drug_a in train_drugs
            in_train_b = drug_b in train_drugs
            in_val_a = drug_a in val_drugs
            in_val_b = drug_b in val_drugs
            in_test_a = drug_a in test_drugs
            in_test_b = drug_b in test_drugs

            # 检查是否符合条件：有且仅有一个新药
            def check_condition(drug1_in_set, drug2_in_set):
                return (drug1_in_set and not drug2_in_set) or (not drug1_in_set and drug2_in_set)

            # 分配数据
            if in_train_a and in_train_b:
                train_data.append(item)
            elif check_condition(in_val_a, in_val_b) and (in_val_a or in_val_b):
                val_data.append(item)
            elif check_condition(in_test_a, in_test_b) and (in_test_a or in_test_b):
                test_data.append(item)

        # 转换为numpy数组
        train_data = np.array(train_data)
        val_data = np.array(val_data)
        test_data = np.array(test_data)

        # 保存结果
        np.save(os.path.join(output_dir, f"5{fold}_fold_tr_items.npy"), train_data)
        np.save(os.path.join(output_dir, f"5{fold}_fold_val_items.npy"), val_data)
        np.save(os.path.join(output_dir, f"5{fold}_fold_test_items.npy"), test_data)

        print(f"第 {fold} 折完成: 训练集 {len(train_data)} 条, 验证集 {len(val_data)} 条, 测试集 {len(test_data)} 条")


# 使用示例
if __name__ == "__main__":
    # 加载数据
    data = load_data("all_items.npy")

    # 执行五折交叉验证
    five_fold_cross_validation(data, "./")