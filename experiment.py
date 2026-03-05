import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 读取数据
Credits = pd.read_csv('Credit Scoring.csv', low_memory=False)

# 删除异常值（假设异常值为空值）
Credits.dropna(inplace=True)

# 检查数据是否为空
if Credits.empty:
    raise ValueError("数据为空，请检查数据清洗步骤。")

# 定义目标列
target = 'ServiceCompletionStatus'

# 删除'Current'的行
Credits = Credits[Credits[target] != 'Current']


# 检查数据是否为空
if Credits.empty:
    raise ValueError("数据为空，请检查数据清洗步骤。")

# 将标签转换为数值（'Charged Off'为1，'Fully Paid'为0）
Credits[target] = Credits[target].apply(lambda x: 1 if x == 'Charged Off' else 0)

# 保存处理后的表
# Credits.to_csv('real_data.csv', index=False)


# 特征和标签分离
X = Credits.drop(columns=[target])
y = Credits[target]

# 检查特征是否为空
if X.empty:
    raise ValueError("特征数据为空，请检查特征选择步骤。")

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用KMeans进行聚类
kmeans = KMeans(n_clusters=6, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# 将聚类结果作为新特征
X_scaled = np.hstack((X_scaled, clusters.reshape(-1, 1)))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 使用XGBoost代替ANN
from xgboost import XGBClassifier

# 构建XGBoost模型
model = XGBClassifier(scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]), random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 获取特征重要性
feature_importances = model.feature_importances_

# 打印特征重要性
for feature, importance in zip(X.columns.tolist() + ['cluster'], feature_importances):
    print(f"特征: {feature}, 重要性: {importance:.4f}")

# 根据特征重要性选择前N个重要特征
important_features = np.argsort(feature_importances)[-5:]  # 选择最重要的5个特征
X_train_selected = X_train[:, important_features]
X_test_selected = X_test[:, important_features]

# 使用选择的特征重新训练模型
model_selected = XGBClassifier(scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]), random_state=42)
model_selected.fit(X_train_selected, y_train)

# 评估模型
from sklearn.metrics import precision_score, recall_score, f1_score

# 预测结果
y_train_pred = model_selected.predict(X_train_selected)
y_test_pred = model_selected.predict(X_test_selected)

# 计算各项指标
train_accuracy = model_selected.score(X_train_selected, y_train)
test_accuracy = model_selected.score(X_test_selected, y_test)

train_precision = precision_score(y_train, y_train_pred)
test_precision = precision_score(y_test, y_test_pred)

train_recall = recall_score(y_train, y_train_pred)
test_recall = recall_score(y_test, y_test_pred)

train_f1 = f1_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)

# 输出结果
print(f"训练集准确率: {train_accuracy:.4f}")
print(f"测试集准确率: {test_accuracy:.4f}")
print(f"训练集精确率: {train_precision:.4f}")
print(f"测试集精确率: {test_precision:.4f}")
print(f"训练集召回率: {train_recall:.4f}")
print(f"测试集召回率: {test_recall:.4f}")
print(f"训练集F1分数: {train_f1:.4f}")
print(f"测试集F1分数: {test_f1:.4f}")