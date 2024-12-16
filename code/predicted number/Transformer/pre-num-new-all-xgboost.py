import pandas as pd
from sentence_transformers import SentenceTransformer
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import numpy as np
import joblib
import torch

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 all-mpnet-base-v2 模型
model_name = 'sentence-transformers/all-mpnet-base-v2'
sentence_model = SentenceTransformer(model_name).to(device)

# 读取CSV文件
df = pd.read_csv('cleaned_data2.csv', encoding='utf-8-sig')

# 拼接 'title', 'abstract' 和 'toc' 列
texts = [str(row['title']) + " " + str(row['abstract'])  for _, row in df.iterrows()]

# 生成文本嵌入
embeddings = sentence_model.encode(texts, convert_to_tensor=True).cpu().numpy()

# 计算每个文档的 LCSH 标签数量
df['label_count'] = df['lcsh_subject_headings'].apply(
    lambda x: len(str(x).split(';')) if pd.notna(x) else 0
)

# 提取特征和标签
X = embeddings
y = df['label_count'].values

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练 XGBoost 回归模型
reg = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
reg.fit(X_train, y_train)

# 进行预测
y_pred = reg.predict(X_test)

# 确保预测标签个数为整数
y_pred = np.round(y_pred).astype(int)

# 计算各项指标
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
correlation, _ = pearsonr(y_test, y_pred)
average_difference = np.mean(np.abs(y_test - y_pred))
precision = np.sum(np.abs(y_test - y_pred) < 1) / len(y_test)

# 打印结果
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Correlation: {correlation}")
print(f"Average Difference: {average_difference}")
print(f"Precision: {precision}")

# 保存模型性能指标
with open('model_performance_xgboost.txt', 'w') as f:
    f.write(f"Mean Squared Error (MSE): {mse}\n")
    f.write(f"Root Mean Squared Error (RMSE): {rmse}\n")
    f.write(f"Mean Absolute Error (MAE): {mae}\n")
    f.write(f"Correlation: {correlation}\n")
    f.write(f"Average Difference: {average_difference}\n")
    f.write(f"Precision: {precision}\n")

# 将预测结果保存到 CSV 文件
results_df = pd.DataFrame({'Actual_Label_Count': y_test, 'Predicted_Label_Count': y_pred})
results_df.to_csv('label_count_predictions_xgboost-all-at-new-all.csv', index=False, encoding='utf-8-sig')

# 保存训练好的 XGBoost 模型
joblib.dump(reg, 'xgboost_model.pkl')
