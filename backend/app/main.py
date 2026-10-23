from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from jose import jwt, JWTError
import json
from fastapi import Header
from app.database import engine, Base, get_db
from app import models, schemas, security
import pandas as pd
import joblib
import os

# Khởi tạo bảng tự động (nếu chưa dùng Alembic)
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Prediction Service API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép mọi domain gọi tới (chỉ dùng khi code local)
    allow_credentials=True,
    allow_methods=["*"],  # Quan trọng nhất: Cho phép mọi phương thức (để không bị chặn OPTIONS)
    allow_headers=["*"],  # Cho phép gửi mọi header (để không bị chặn Authorization token)
)


# 1. Hàm trung gian giải mã Token để tìm ra User hiện tại
def get_current_user(authorization: str = Header(None), db: Session = Depends(get_db)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Thiếu token xác thực")
    token = authorization.split(" ")[1]
    try:
        # Dùng chung SECRET_KEY và ALGORITHM từ file security.py
        payload = jwt.decode(token, security.SECRET_KEY, algorithms=[security.ALGORITHM])
        email = payload.get("sub")
        user = db.query(models.User).filter(models.User.email == email).first()
        if not user:
            raise HTTPException(status_code=401, detail="User không tồn tại")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Token không hợp lệ hoặc đã hết hạn")


#2. API Đăng ký & Đăng nhập
@app.post("/auth/register", response_model=schemas.UserResponse, status_code=status.HTTP_201_CREATED)
def register(user_in: schemas.UserRegister, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user_in.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="email đã tồn tại trên hệ thống")
    
    hashed_pwd = security.hash_password(user_in.password)
    new_user = models.User(
        full_name=user_in.full_name, # Thêm dòng này
        email=user_in.email,
        hashed_password=hashed_pwd,
        role=user_in.role,
        region=user_in.region
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

# API Đăng nhập
@app.post("/auth/login", response_model=schemas.Token)
def login(credentials: schemas.UserLogin, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == credentials.email).first()
    if not user or not security.verify_password(credentials.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Tài khoản hoặc mật khẩu không chính xác")
    
    access_token = security.create_access_token(data={"sub": user.email, "role": user.role})
    return {"access_token": access_token, "token_type": "bearer"}

# 3. Load Model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "outputs/models", "decision_tree_pipeline.pkl")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Lỗi load model: {e}")
    model = None
    
import shap
import json
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, Header, status
from sqlalchemy.orm import Session
from app.database import get_db
from app import models, schemas
# (Giữ nguyên các import và cấu hình khác của bạn)

# 4. API Predict (Tích hợp thư viện SHAP thực tế)
@app.post("/predict", response_model=schemas.PredictResponse)
def predict(request: schemas.PredictRequest, authorization: str = Header(None), db: Session = Depends(get_db)):
    user = get_current_user(authorization, db)
    
    if model is None:
        return schemas.PredictResponse(
            prediction_result=0, risk_probability=0.0,
            message="Lỗi load model", shap=[]
        )

    # 1. Chuyển JSON thành DataFrame
    df_input = pd.DataFrame([request.model_dump()])
    
    # 2. Lấy xác suất dự đoán từ Pipeline tổng
    risk_prob = model.predict_proba(df_input)[0][1]
    is_high_risk = 1 if risk_prob > 0.5 else 0

    # 3. GIẢI THÍCH MÔ HÌNH VỚI SHAP
    shap_dynamic = []
    try:
        # Tách preprocessor và classifier từ Pipeline đã load
        preprocessor = model.named_steps['preprocessor']
        classifier = model.named_steps['classifier']
        
        # Biến đổi dữ liệu đầu vào qua preprocessor để lấy ma trận số (21 cột)
        X_processed = preprocessor.transform(df_input)
        feature_names = preprocessor.get_feature_names_out()
        
        # Dùng TreeExplainer cho Decision Tree / Random Forest
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_processed)
        
        # Xử lý format đầu ra của SHAP (DecisionTree trả về list chứa 2 mảng cho 2 class)
        if isinstance(shap_values, list):
            vals = shap_values[1][0] # Lấy mảng SHAP của class 1 (Đột quỵ), dòng đầu tiên
        elif len(shap_values.shape) == 3:
            vals = shap_values[0, :, 1]
        else:
            vals = shap_values[0]
            
        # Dictionary map tên tiếng Anh/kỹ thuật sang Tiếng Việt cho báo cáo đẹp hơn
        name_mapping = {
            "num__age": "Tuổi",
            "num__avg_glucose_level": "Đường huyết",
            "num__bmi": "Chỉ số BMI",
            "pass__hypertension": "Huyết áp cao",
            "pass__heart_disease": "Bệnh tim mạch",
            "cat__gender_Male": "Giới tính (Nam)",
            "cat__gender_Female": "Giới tính (Nữ)",
            "cat__gender_Other": "Giới tính (Khác)",
            "cat__ever_married_Yes": "Đã kết hôn",
            "cat__ever_married_No": "Chưa kết hôn",
            "cat__work_type_Private": "Công việc (Tư nhân)",
            "cat__work_type_Self-employed": "Công việc (Tự do)",
            "cat__work_type_Govt_job": "Công việc (Nhà nước)",
            "cat__work_type_children": "Công việc (Trẻ em)",
            "cat__work_type_Never_worked": "Công việc (Chưa từng làm)",
            "cat__Residence_type_Urban": "Nơi ở (Thành thị)",
            "cat__Residence_type_Rural": "Nơi ở (Nông thôn)",
            "cat__smoking_status_smokes": "Đang hút thuốc",
            "cat__smoking_status_formerly smoked": "Từng hút thuốc",
            "cat__smoking_status_never smoked": "Chưa từng hút thuốc",
            "cat__smoking_status_Unknown": "Tình trạng hút thuốc (Không rõ)"
        }

        grouped_shap = {}
        for name, val in zip(feature_names, vals):
            # Xác định tên nhóm gốc dựa vào tiền tố
            if "smoking_status" in name:
                group_name = "Tình trạng hút thuốc"
            elif "work_type" in name:
                group_name = "Đặc thù công việc"
            elif "Residence_type" in name:
                group_name = "Môi trường sống"
            elif "ever_married" in name:
                group_name = "Tình trạng hôn nhân"
            elif "gender" in name:
                group_name = "Giới tính"
            else:
                # Các biến số (tuổi, BMI...) hoặc biến giữ nguyên thì dùng name_mapping
                clean_name = name_mapping.get(name, name.replace("num__", "").replace("pass__", ""))
                group_name = clean_name
            
            # Cộng dồn điểm SHAP
            if group_name not in grouped_shap:
                grouped_shap[group_name] = 0.0
            grouped_shap[group_name] += val

        # Lọc các nhóm có ảnh hưởng đáng kể và đưa vào kết quả trả về
        for feature_name, total_val in grouped_shap.items():
            if abs(total_val) > 0.001: 
                shap_dynamic.append({
                    "feature": feature_name,
                    "value": float(round(total_val, 4))
                })
                
                
        # Sắp xếp các yếu tố theo mức độ ảnh hưởng (giá trị tuyệt đối giảm dần)
        shap_dynamic.sort(key=lambda x: abs(x["value"]), reverse=True)

    except Exception as e:
        print(f"Lỗi khi chạy SHAP: {e}")
        shap_dynamic = [] # Trả về mảng rỗng nếu có lỗi để API không bị crash

    # 4. Lưu lịch sử
    action_data = {
        "glucose": request.avg_glucose_level,
        "bmi": request.bmi,
        "bp": "Có" if request.hypertension == 1 else "Không",
        "risk": float(risk_prob * 100) if risk_prob <= 1 else float(risk_prob)
    }
    
    new_history = models.History(
        user_id=user.id,
        action=json.dumps(action_data) 
    )
    db.add(new_history)
    db.commit()

    return schemas.PredictResponse(
        prediction_result=is_high_risk,
        risk_probability=risk_prob,
        message="Chẩn đoán AI hoàn tất",
        shap=shap_dynamic
    )
# 5. API History
@app.get("/history")
def get_history(authorization: str = Header(None), db: Session = Depends(get_db)):
    user = get_current_user(authorization, db)
    histories = db.query(models.History).filter(models.History.user_id == user.id).order_by(models.History.timestamp).all()
    
    results = []
    for h in histories:
        try:
            data = json.loads(h.action) 
            results.append({
                "date": h.timestamp.strftime("%d/%m %H:%M"), 
                "glucose": data.get("glucose"),
                "bmi": data.get("bmi"),
                "bp": data.get("bp"),
                "risk": data.get("risk")
            })
        except:
            continue
            
    return results