import pandas as pd

# 读取CSV文件
df = pd.read_csv('accepted_2007_to_2018Q4.csv')

# 删除第十六万行之后的行
df = df.iloc[:160000]  # 保留前160000行

# 保存修改后的CSV文件
df.to_csv('customers.csv', index=False)