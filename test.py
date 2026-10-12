import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
import os



#Test preprocessing.py



# Import hàm build_preprocessor từ file preprocessing cậu vừa viết
# (Giả sử file preprocessing.py nằm trong thư mục src)
from src.preprocessing import build_preprocessor

# =====================================================================
# PHẦN 1: GIẢ LẬP KHỐI AI (Nhiệm vụ của Nhật Giang - Tuần 3, 4, 6)
# =====================================================================
print("--- ĐANG HUẤN LUYỆN MÔ HÌNH ---")

# 1. Đọc dữ liệu đầu vào (File 10 cột thô mà cậu đã xuất ở 03_clean_data.py)
# Lưu ý: Cậu cần đảm bảo file này đã tồn tại ở './data/processed/stroke_model_input.csv'
df_train = pd.read_csv('./data/processed/stroke_clean.csv')

# Tách biến mục tiêu (y) và đặc trưng (X)
X_train = df_train.drop(columns=['stroke'])
y_train = df_train['stroke']

# 2. Tạo Pipeline TỔNG: Gồm Tiền xử lý (của cậu) + Thuật toán AI (của Giang)
preprocessor = build_preprocessor()

# Dùng RandomForest cơ bản làm ví dụ
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

# 3. Huấn luyện (Fit) Pipeline trên toàn bộ dữ liệu Train
model_pipeline.fit(X_train, y_train)

# 4. Xuất file Pipeline ra thư mục gốc
os.makedirs('./models', exist_ok=True)
joblib.dump(model_pipeline, './models/model_pipeline.pkl')
print("Đã đóng gói thành công file ./models/model_pipeline.pkl!\n")


# =====================================================================
# PHẦN 2: GIẢ LẬP KHỐI BACKEND (Nhiệm vụ của Xuân Hai - Tuần 4, 6)
# =====================================================================
print("--- BÁC SĨ NHẬP LIỆU (BACKEND DỰ ĐOÁN) ---")

# 1. Đây là JSON thô bay từ Frontend (React/Vue) về Backend
json_payload = {
    "age": 67.0,
    "gender": "Male",
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked"
}

# 2. Backend load file Pipeline (Chỉ load 1 lần khi khởi động Server)
loaded_pipeline = joblib.load('./models/model_pipeline.pkl')

# 3. Backend chuyển JSON thành DataFrame 1 dòng
df_input = pd.DataFrame([json_payload])

# 4. Backend đẩy vào mô hình để dự đoán và lấy xác suất
prediction = loaded_pipeline.predict(df_input) # Kết quả 0 hoặc 1
probabilities = loaded_pipeline.predict_proba(df_input) # Xác suất [% Không bị, % Bị]

# Lấy xác suất của lớp 1 (Có nguy cơ đột quỵ)
stroke_risk_percent = probabilities[0, 1] * 100

# =====================================================================
# PHẦN 3: KẾT QUẢ TRẢ VỀ CHO FRONTEND
# =====================================================================
print(f"JSON Input: {json_payload}")
print("-" * 30)
if prediction[0] == 1:
    print(f"🚨 CẢNH BÁO: Bệnh nhân có nguy cơ ĐỘT QUỴ!")
else:
    print(f"✅ AN TOÀN: Bệnh nhân thuộc nhóm rủi ro thấp.")
    
print(f"Điểm nguy cơ của mô hình: {stroke_risk_percent:.2f}%")