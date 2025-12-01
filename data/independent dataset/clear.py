import pandas as pd
import numpy as np

# 1. 读取CSV文件（替换为你的CSV文件路径）
csv_file_path = "./indep0-oneil/oneil_drugcombs_filter.csv"  # 输入CSV文件路径
df = pd.read_csv(csv_file_path)

# 2. 检查目标列是否存在（避免列名错误导致的报错）
target_columns = ["drugA_canonical_smi", "drugB_canonical_smi", "DepMap_ID", "synergy_clean"]
missing_columns = [col for col in target_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"CSV文件中缺少以下列：{missing_columns}")

# 3. 提取目标列（按指定顺序）
selected_data = df[target_columns]

# 4. 转换为numpy数组（注意：字符串和数值混合时，dtype会自动转为object）
data_array = selected_data.to_numpy()

# 5. 保存为npy文件（替换为你的输出路径）
npy_file_path = "./indep0-oneil/all_items.npy"  # 输出npy文件路径
np.save(npy_file_path, data_array)

print(f"成功提取列并保存为npy文件：{npy_file_path}")