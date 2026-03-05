import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# 1. 读取数据
# -----------------------------
df = pd.read_csv('real_data.csv', low_memory=False)
features = df.iloc[:, :8]  # 前8列特征
label = df.iloc[:, 8]      # 第9列标签

# 创建文件夹保存图表
os.makedirs("feature_plots", exist_ok=True)

# -----------------------------
# 2. 每个特征的统计描述和零值比例
# -----------------------------
desc_stats = features.describe().T
desc_stats['missing_rate'] = features.isna().mean()
desc_stats['zero_ratio'] = (features==0).sum() / len(features)

print("=== 特征统计描述 ===")
print(desc_stats)

# 保存统计表
desc_stats.to_csv("feature_stats.csv")

# -----------------------------
# 3. 所有特征的箱型图 (一张图)
# -----------------------------
sns.set(style="whitegrid")
plt.figure(figsize=(16, 10))
for i, col in enumerate(features.columns, 1):
    plt.subplot(2, 4, i)   # 2行4列排布
    sns.boxplot(x=features[col])
    plt.title(f"Boxplot of {col}", fontsize=10)
    plt.xlabel(col, fontsize=8)
plt.tight_layout()
plt.savefig("feature_plots/all_boxplots.png", dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------
# 4. 所有特征的直方图 (一张图)
# -----------------------------
plt.figure(figsize=(16, 10))
for i, col in enumerate(features.columns, 1):
    plt.subplot(2, 4, i)
    sns.histplot(features[col], bins=30, kde=True)
    plt.title(f"Histogram of {col}", fontsize=10)
    plt.xlabel(col, fontsize=8)
    plt.ylabel("Frequency", fontsize=8)
plt.tight_layout()
plt.savefig("feature_plots/all_histograms.png", dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------
# 6. 相关性矩阵可视化 (单张图)
# -----------------------------
corr = features.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.savefig("feature_plots/feature_correlation_matrix.png", dpi=300, bbox_inches='tight')
plt.close()
