import pandas as pd

# -----------------------------
# 1. 读取数据
# -----------------------------
df = pd.read_csv('Credit Scoring.csv', low_memory=False)
features = df.iloc[:, :8]  # 前8列特征
label = df.iloc[:, 8]      # 第9列标签

Q1 = features['LastReputationScore'].quantile(0.25)
Q3 = features['LastReputationScore'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
outliers = features[features['LastReputationScore'] < lower]
print(outliers)
#
# median_score = df[df['LastReputationScore'] != 0]['LastReputationScore'].median()
# df['LastReputationScore'] = df['LastReputationScore'].replace(0, median_score)
# df.to_csv('Credit Scoring.csv', index=False)
