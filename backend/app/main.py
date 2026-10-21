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
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_pipeline.pkl")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Lỗi load model: {e}")
    model = None
    
# 4. API Predict (Đã kèm logic lưu Database)
@app.post("/predict", response_model=schemas.PredictResponse)
def predict(request: schemas.PredictRequest, authorization: str = Header(None), db: Session = Depends(get_db)):
    user = get_current_user(authorization, db)
    
    if model is None:
        return schemas.PredictResponse(
            prediction_result=0, risk_probability=0.0,
            message="Lỗi load model", shap=[]
        )

    df_input = pd.DataFrame([request.model_dump()])
    risk_prob = model.predict_proba(df_input)[0][1]
    is_high_risk = 1 if risk_prob > 0.5 else 0

    shap_mock = [
        {"feature": "Đường huyết", "value": 0.2 if request.avg_glucose_level > 140 else -0.1},
        {"feature": "Tuổi", "value": 0.15 if request.age > 60 else -0.05},
        {"feature": "BMI", "value": 0.1 if request.bmi > 25 else -0.02}
    ]

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
        shap=shap_mock
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