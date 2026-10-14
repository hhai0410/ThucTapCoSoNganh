import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

df = pd.read_csv('./data/raw/healthcare-dataset-stroke-data.csv')

# 2. Dọn dẹp cột không cần thiết
df = df.drop(columns=['id'])


# Xuất ra file
os.makedirs('./data/processed', exist_ok=True)
df.to_csv('./data/processed/stroke_clean.csv', index=False)
print("\nĐã xuất file stroke_clean.csv thành công! Tổng số cột:", len(df.columns))