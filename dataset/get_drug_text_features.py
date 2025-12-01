import re
import requests
from openai import OpenAI
import json
import pandas as pd
if __name__ == '__main__':
    # DeepSeek API地址和密钥
    base_url = ""
    api_key = ""
    csv_file = "../data/drug_info.csv"  # 替换为你的CSV文件路径
    data = pd.read_csv(csv_file)
    drug_features = {}
    for index, row in data.iterrows():
        # 构造提示词
        print(index)
        prompt = (f"Generate a comprehensive biomedical description for the drug [smiles:{row['canonicalsmiles']},pubchem_id:{row['pubchem_id']},drug_name:{row['drug_name']}]. Include:"
                f"1. Chemical structure characteristics (e.g., functional groups, molecular weight)"
                f"2. Known molecular targets and pathways"
                f"3. Mechanism of action in human biology"
                f"4. Pharmacokinetic properties (absorption, distribution, metabolism, excretion)"
                f"5. Common therapeutic applications"
                f"6. Reported side effects and toxicity profiles"
                f"7. Any known drug-drug interaction patterns"
                f"Generate strictly according to the above format, Use concise and professional language to describe, without any unnecessary messages and note messages.")
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model="deepseek-v3",
                messages = [
                {"role": "system", "content": "You are a drug feature generation assistant."},
                {"role": "user", "content": prompt}],
                temperature=0.5,
                stream = False,
                max_tokens=1500,
            )
            answer = response.choices[0].message.content
            drug_features[row['canonicalsmiles']] = answer
        except Exception as e:
            print(f"请求失败: {e}")
            break
    # 将结果保存到JSON文件
    with open("../data/drug_text.json", "w") as f:
        json.dump(drug_features, f, indent=4)
