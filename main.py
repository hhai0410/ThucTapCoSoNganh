import pandas as pd

df = pd.read_csv('./data/healthcare-dataset-stroke-data.csv')
print('-----THÔNG TIN CẤU TRÚC BỘ DỮ LIỆU-----')
print(df.info())

print('\n-----PHÂN BỐ NHÓM BỆNH NHÂN (CỘT STROKE)-----')
print(df['stroke'].value_counts())


print('-----THỰC HIỆN ĐIỀN CÁC DỮ LIỆU CÒN THIẾU VÀ LOẠI BỎ CỘT ID-----')
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

df_clean = pd.get_dummies(df, drop_first=True, dtype=int)
df_clean = df_clean.drop(columns=['id'])

print('-----Bảng dữ liệu đã sạch sẽ-----')
#Tách ra bảng X và bảng Y
X = df_clean.drop(columns=['stroke'])
Y = df_clean['stroke']

#In ra kích thước để kiểm tra
print('-----Kích thước ma trận dữ liệu-----')
print(f'Tập dữ liệu X: {X.shape} (Dòng, Cột)')
print(f'Tập dữ liệu Y: {Y.shape} (Dòng, Cột)')

#Lưu toàn bộ bảng ra 1 file csv 
df_clean.to_csv('./data/stroke_clean.csv', index=False)
print('\n[Thành công] Đã lưu file stroke_clean.csv vào thư mục data')


from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
#Đọc dữ liệu sạch
df = pd.read_csv('./data/stroke_clean.csv')
X = df.drop(columns=['stroke'])
Y = df['stroke']
#Chia tập train 80% và test 20%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"Trước SMOTE, số ca đột quỵ trong tập Train: {sum(Y_train == 1)}")
#Cân bằng dữ liệu với SMOTE
smote = SMOTE(random_state=42)
X_train_smote, Y_train_smote = smote.fit_resample(X_train, Y_train)
print(f"Sau SMOTE, số ca đột quỵ trong tập Train: {sum(Y_train_smote == 1)}\n")

# 4. Dạy mô hình Random Forest
print("Đang huấn luyện mô hình...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train_smote, Y_train_smote)

# 5. Làm bài thi và chấm điểm
Y_pred = model.predict(X_test)
print("--- MA TRẬN NHẦM LẪN (CONFUSION MATRIX) ---")
print(confusion_matrix(Y_test, Y_pred))
print("\n--- BÁO CÁO CHI TIẾT ---")
print(classification_report(Y_test, Y_pred))
