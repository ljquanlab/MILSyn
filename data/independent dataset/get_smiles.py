import numpy as np
import json
import os


def extract_unique_smiles(npy_files, output_json):
    """
    从多个npy文件中提取独特的药物SMILES并保存到JSON文件

    参数:
        npy_files: npy文件路径列表
        output_json: 输出JSON文件路径
    """
    # 存储独特的药物SMILES
    unique_drugs = set()

    # 遍历所有npy文件
    for file_path in npy_files:
        if not os.path.exists(file_path):
            print(f"警告: 文件 {file_path} 不存在，已跳过")
            continue

        try:
            # 加载npy文件
            data = np.load(file_path, allow_pickle=True)

            # 假设数据结构为DrugA, DrugB, cell, score
            # 提取DrugA和DrugB列的SMILES
            for entry in data:
                # 假设前两列是药物SMILES
                drug_a = entry[0]
                drug_b = entry[1]

                # 添加到集合中自动去重
                unique_drugs.add(drug_a)
                unique_drugs.add(drug_b)

            print(f"已处理文件: {file_path}, 目前独特药物数量: {len(unique_drugs)}")

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")

    # 转换为字典格式以便保存为JSON
    drug_dict = {i: smiles for i, smiles in enumerate(unique_drugs)}

    # 保存到JSON文件
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(drug_dict, f, ensure_ascii=False, indent=4)

    print(f"处理完成，共提取 {len(unique_drugs)} 种独特药物")
    print(f"结果已保存到: {output_json}")


if __name__ == "__main__":
    # 在这里设置你的五个npy文件路径
    npy_file_paths = [
        "../split/all_items.npy",
        "indep0-oneil/all_items.npy",
        "indep1-almanac/all_items.npy",
        "indep2-OncologyScreen/all_items.npy",
        "indep3-DrugCombDB/all_items.npy"
    ]

    # 输出JSON文件路径
    output_json_path = "unique_drugs_smiles.json"

    # 执行提取操作
    extract_unique_smiles(npy_file_paths, output_json_path)
