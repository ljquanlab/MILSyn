import re
import requests
from openai import OpenAI
import json
import pandas as pd
if __name__ == '__main__':
    # DeepSeek API地址和密钥
    base_url = ""
    api_key = ""
    csv_file = "../data/cell_line_info.csv"  # 替换为你的CSV文件路径
    data = pd.read_csv(csv_file)
    cell_line_features = {}
    for index, row in data.iterrows():
        # 构造提示词
        print(index)
        prompt = (
            f"Generate a concise, structured description of the cell line [DepMap ID: {row['DepMap_ID']},cell_line_name: {row['cell_line_name']},"
            f"CCLE_Name: {row['CCLE_Name']} ] focusing on features relevant to synergy prediction. Include:"
            f"1. Cell type and origin (e.g., cancer type, tissue source)"
            f"2. Molecular signatures (e.g., genetic mutations, protein expression)"
            f"3. Signaling pathways (e.g., activated/inhibited pathways)"
            f"4. Known drug response profiles (e.g., sensitivity/resistance to known compounds)"
            f"5. Cellular microenvironment (e.g., extracellular matrix, immune cell presence)"
            f"6. Key biomarkers (e.g., prognostic markers, therapeutic targets)."
            f"Generate strictly according to the above format, Use concise and professional language to describe, without any unnecessary messages and note messages.")
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model="deepseek-v3",
                messages = [
                {"role": "system", "content": "You are a cell_line feature generation assistant."},
                {"role": "user", "content": prompt}],
                temperature=0.5,
                stream = False,
                max_tokens=1500,
            )
            answer = response.choices[0].message.content
            cell_line_features[row['DepMap_ID']] = answer
        except Exception as e:
            print(f"请求失败: {e}")
            break
    # 将结果保存到JSON文件
    with open("../data/cell_text.json", "w") as f:
        json.dump(cell_line_features, f, indent=4)
