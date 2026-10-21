from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import engine, Base, get_db
from app import models, schemas, security

# Khởi tạo bảng tự động (nếu chưa dùng Alembic)
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Prediction Service API")

# API Đăng ký
@app.post("/auth/register", response_model=schemas.UserResponse, status_code=status.HTTP_201_CREATED)
def register(user_in: schemas.UserRegister, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.username == user_in.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username đã tồn tại trên hệ thống")
    
    hashed_pwd = security.hash_password(user_in.password)
    new_user = models.User(
        username=user_in.username,
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
    user = db.query(models.User).filter(models.User.username == credentials.username).first()
    if not user or not security.verify_password(credentials.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Tài khoản hoặc mật khẩu không chính xác")
    
    access_token = security.create_access_token(data={"sub": user.username, "role": user.role})
    return {"access_token": access_token, "token_type": "bearer"}

# Draft API Predict (Cho Frontend ghép thử UI)
@app.post("/predict", response_model=schemas.PredictResponse)
def predict(request: schemas.PredictRequest):
    # Dummy logic đơn giản: Nếu chỉ số BMI > 30 hoặc Glucose > 140 thì cảnh báo nguy cơ cao (1)
    is_high_risk = 1 if (request.bmi > 30.0 or request.avg_glucose_level > 140.0 or request.hypertension == 1) else 0
    confidence = 0.88 if is_high_risk == 1 else 0.95
    return schemas.PredictResponse(
        prediction_result=is_high_risk,
        confidence=confidence,
        message="Dự đoán nguy cơ đột quỵ thành công"
    )