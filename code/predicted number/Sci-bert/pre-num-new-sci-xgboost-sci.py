import pandas as pd
from sentence_transformers import SentenceTransformer
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import numpy as np
import joblib
import torch

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load SentenceTransformer model
model_name = 'allenai/scibert_scivocab_uncased'
sentence_model = SentenceTransformer(model_name).to(device)

# Read CSV file
df = pd.read_csv('cleaned_data2.csv', encoding='utf-8-sig')

# Concatenate 'title' and 'abstract' columns
texts = [str(row['title']) + " " + str(row['abstract']) for _, row in df.iterrows()]

# Generate text embeddings
embeddings = sentence_model.encode(texts, convert_to_tensor=True).cpu().numpy()

# Calculate LCSH label count for each document
df['label_count'] = df['lcsh_subject_headings'].apply(
    lambda x: len(str(x).split(';')) if pd.notna(x) else 0
)

# Extract features and labels
X = embeddings
y = df['label_count'].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost regressor
reg = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
reg.fit(X_train, y_train)

# Make predictions
y_pred = reg.predict(X_test)

# Ensure predicted label counts are integers
y_pred = np.round(y_pred).astype(int)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
correlation, _ = pearsonr(y_test, y_pred)
average_difference = np.mean(np.abs(y_test - y_pred))
precision = np.sum(np.abs(y_test - y_pred) < 1) / len(y_test)

# Print results
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Correlation: {correlation}")
print(f"Average Difference: {average_difference}")
print(f"Precision: {precision}")

# Save model performance metrics
with open('model_performance_scibert_xgboost.txt', 'w') as f:
    f.write(f"Mean Squared Error (MSE): {mse}\n")
    f.write(f"Root Mean Squared Error (RMSE): {rmse}\n")
    f.write(f"Mean Absolute Error (MAE): {mae}\n")
    f.write(f"Correlation: {correlation}\n")
    f.write(f"Average Difference: {average_difference}\n")
    f.write(f"Precision: {precision}\n")

# Save prediction results to CSV
results_df = pd.DataFrame({'Actual_Label_Count': y_test, 'Predicted_Label_Count': y_pred})
results_df.to_csv('label_count_predictions_scibert_xgboost.csv', index=False, encoding='utf-8-sig')

# Save trained XGBoost model
joblib.dump(reg, 'scibert_xgboost_model.pkl')
