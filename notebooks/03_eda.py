import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 1. Điền các cột bị thiếu bằng trung vị
df = pd.read_csv('./data/raw/healthcare-dataset-stroke-data.csv')
median_bmi = df['bmi'].median()
df['bmi'] = df['bmi'].fillna(median_bmi)

# 2. Dọn dẹp cột không cần thiết
df = df.drop(columns=['id'])

# 3. Mã hóa dữ liệu (Tuân thủ tuyệt đối Data Spec V1)
# Ánh xạ thủ công các cột nhạy cảm để Backend không bị lỗi khi dùng get_dummies(drop_first)
df['gender_Male'] = df['gender'].map({'Male': 1, 'Female': 0, 'Other': 0})
df['gender_Other'] = df['gender'].map({'Other': 1, 'Male': 0, 'Female': 0})
df['ever_married_Yes'] = df['ever_married'].map({'Yes': 1, 'No': 0})

# Chạy One-Hot cho các cột còn lại
df = pd.get_dummies(df, columns=['work_type', 'Residence_type', 'smoking_status'], drop_first=True, dtype=int)

# Bỏ đi các cột gốc dạng chữ sau khi đã ánh xạ
df = df.drop(columns=['gender', 'ever_married'])

# 4. Chuẩn hóa/Scale các biến numerical (Theo đúng yêu cầu Tuần 3 của DE)
numeric_features = ['age', 'avg_glucose_level', 'bmi']
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# 5. Xuất file bàn giao cho ML Engineer
# Chuyển cột 'stroke' xuống cuối cùng cho đẹp ma trận
cols = [col for col in df.columns if col != 'stroke'] + ['stroke']
df_clean = df[cols]

# Xuất ra file
df_clean.to_csv('./data/clean/stroke_clean.csv', index=False)
print("\nĐã xuất file stroke_clean.csv thành công! Tổng số cột:", len(df_clean.columns))