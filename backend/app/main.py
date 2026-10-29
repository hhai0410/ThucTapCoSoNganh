from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from jose import jwt, JWTError
import json
from fastapi import Header, Query, UploadFile, File
from .database import engine, Base, get_db
from . import models, schemas, security
import pandas as pd
import joblib
import os

# Khởi tạo bảng tự động (nếu chưa dùng Alembic)
Base.metadata.create_all(bind=engine)
# Backward compatible migration for the existing local database.
from sqlalchemy import inspect, text
if "is_active" not in {c["name"] for c in inspect(engine).get_columns("users")}:
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE users ADD COLUMN is_active BOOLEAN NOT NULL DEFAULT 1"))

DEMO_MODE = os.getenv("DEMO_MODE", "1") == "1"
from .privacy import encrypt_legacy_rows
encrypt_legacy_rows(engine)
app = FastAPI(title="StrokeCare Screening API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],  # Cho phép mọi domain gọi tới (chỉ dùng khi code local)
    allow_credentials=True,
    allow_methods=["*"],  # Quan trọng nhất: Cho phép mọi phương thức (để không bị chặn OPTIONS)
    allow_headers=["*"],  # Cho phép gửi mọi header (để không bị chặn Authorization token)
)


# 1. Hàm trung gian giải mã Token để tìm ra User hiện tại
def get_current_user(authorization: str = Header(None), db: Session = Depends(get_db)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Thiếu token xác thực")
    token = authorization[7:].strip()
    try:
        # Dùng chung SECRET_KEY và ALGORITHM từ file security.py
        payload = jwt.decode(token, security.SECRET_KEY, algorithms=[security.ALGORITHM])
        email = payload.get("sub")
        user = db.query(models.User).filter(models.User.email == email).first()
        if not user:
            raise HTTPException(status_code=401, detail="User không tồn tại")
        if not user.is_active:
            raise HTTPException(status_code=403, detail="Tài khoản đã bị khóa")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Token không hợp lệ hoặc đã hết hạn")


#2. API Đăng ký & Đăng nhập
@app.post("/auth/register", response_model=schemas.UserResponse, status_code=status.HTTP_201_CREATED)
def register(user_in: schemas.UserRegister, db: Session = Depends(get_db)):
    if user_in.role == "doctor" and not DEMO_MODE:
        raise HTTPException(status_code=403, detail="Tài khoản bác sĩ cần được Admin cấp quyền")
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
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Email đã tồn tại")
    db.refresh(new_user)
    return new_user

# API Đăng nhập
@app.post("/auth/login", response_model=schemas.Token)
def login(credentials: schemas.UserLogin, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == credentials.email.strip().lower()).first()
    if not user or not security.verify_password(credentials.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Tài khoản hoặc mật khẩu không chính xác")
    
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Tài khoản đã bị khóa")
    access_token = security.create_access_token(data={"sub": user.email, "role": user.role})
    return {"access_token": access_token, "token_type": "bearer", "role": user.role}

# 3. Load Model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "outputs/models", "model_pipeline.pkl")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "outputs/models", "best_model_week5.pkl")

import logging
import numpy as np
import shap
from threading import RLock
model_lock = RLock()
model = None
explainer = None
threshold = 0.18

def load_model():
    global model, explainer, threshold
    candidate = joblib.load(MODEL_PATH)
    cutoff = float(getattr(candidate, "decision_threshold_", 0.18))
    if not 0 < cutoff < 1:
        raise ValueError("Invalid model threshold")
    explanation = shap.TreeExplainer(candidate.named_steps["classifier"])
    with model_lock:
        model, explainer, threshold = candidate, explanation, cutoff

try:
    load_model()
except Exception:
    logging.exception("Không nạp được mô hình")

# 4. API Predict (Tích hợp thư viện SHAP thực tế)
@app.post("/predict", response_model=schemas.PredictResponse)
def predict(request: schemas.PredictRequest, authorization: str = Header(None), db: Session = Depends(get_db),
            patient_id: int | None = None, cutoff: float | None = Query(None, gt=0, lt=1)):
    user = get_current_user(authorization, db)
    target = user
    if patient_id is not None or cutoff is not None:
        if user.role != "doctor":
            raise HTTPException(status_code=403, detail="Chỉ bác sĩ được chọn bệnh nhân hoặc ngưỡng")
        if patient_id is not None:
            target = db.get(models.User, patient_id)
            if not target or target.role != "user":
                raise HTTPException(status_code=404, detail="Không tìm thấy bệnh nhân")
    with model_lock:
        active_model, active_explainer, active_threshold = model, explainer, threshold
    if active_model is None:
        raise HTTPException(status_code=503, detail="Mô hình chưa sẵn sàng")
    if cutoff is not None:
        active_threshold = cutoff
    df_input = pd.DataFrame([request.model_dump()])
    risk_prob = float(active_model.predict_proba(df_input)[0][1])
    is_high_risk = int(risk_prob >= active_threshold)

    # 3. GIẢI THÍCH MÔ HÌNH VỚI SHAP
    shap_dynamic = []
    try:
        # Tách preprocessor và classifier từ Pipeline đã load
        preprocessor = active_model.named_steps['preprocessor']
        classifier = active_model.named_steps['classifier']
        
        # Biến đổi dữ liệu đầu vào qua preprocessor để lấy ma trận số (21 cột)
        X_processed = preprocessor.transform(df_input)
        feature_names = preprocessor.get_feature_names_out()
        
        # Dùng TreeExplainer cho Decision Tree / Random Forest
        shap_values = active_explainer.shap_values(X_processed)
        
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
        raise HTTPException(status_code=503, detail="Không tạo được giải thích mô hình")

    # 4. Lưu lịch sử
    action_data = {
        "performed_by": user.id,
        "input": request.model_dump(),
        "prediction_result": is_high_risk,
        "threshold": active_threshold,
        "model_name": getattr(active_model,"model_name_",type(active_model.named_steps["classifier"]).__name__),
        "shap": shap_dynamic,
        "glucose": request.avg_glucose_level,
        "bmi": request.bmi,
        "bp": "Có" if request.hypertension == 1 else "Không",
        "risk": float(risk_prob * 100) if risk_prob <= 1 else float(risk_prob)
    }
    
    new_history = models.History(
        user_id=target.id,
        action=json.dumps(action_data) 
    )
    db.add(new_history)
    db.commit()

    return schemas.PredictResponse(
        prediction_result=is_high_risk,
        risk_probability=risk_prob,
        message="Sàng lọc AI hoàn tất",
        threshold=active_threshold,
        shap=shap_dynamic
    )
# 5. API History
@app.get("/history")
def get_history(authorization: str = Header(None), db: Session = Depends(get_db)):
    user = get_current_user(authorization, db)
    return history_for(user.id, db)


def history_for(user_id, db):
    from datetime import timezone, timedelta
    rows = db.query(models.History).filter(models.History.user_id == user_id).order_by(models.History.timestamp, models.History.id).all()
    results = []
    for row in rows:
        try:
            data = json.loads(row.action)
        except (ValueError, TypeError):
            continue
        local_time = row.timestamp.replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=7)))
        results.append({**data, "id": row.id, "date": local_time.strftime("%d/%m/%Y %H:%M:%S"),
                        "prediction_result": data.get("prediction_result", int(data.get("risk", 0) >= 18)),
                        "threshold": data.get("threshold", 0.18)})
    return results


def require_role(*roles):
    def check(user=Depends(get_current_user)):
        if user.role not in roles:
            raise HTTPException(status_code=403, detail="Không có quyền truy cập")
        return user
    return check


@app.get("/auth/me")
def me(user=Depends(get_current_user)):
    return {"id": user.id, "email": user.email, "full_name": user.full_name, "role": user.role}


@app.get("/doctor/patients")
def patients(q: str = "", user=Depends(require_role("doctor")), db: Session = Depends(get_db)):
    rows = db.query(models.User).filter(models.User.role == "user")
    records = rows.all()
    if q:
        records = [r for r in records if q.casefold() in ((r.full_name or "") + " " + r.email).casefold()]
    return [{"id": r.id, "full_name": r.full_name, "email": r.email, "region": r.region} for r in records]


@app.get("/doctor/patient/{patient_id}/history")
def patient_history(patient_id: int, user=Depends(require_role("doctor")), db: Session = Depends(get_db)):
    patient = db.get(models.User, patient_id)
    if not patient or patient.role != "user":
        raise HTTPException(status_code=404, detail="Không tìm thấy bệnh nhân")
    return history_for(patient_id, db)


@app.get("/admin/users")
def users(user=Depends(require_role("admin")), db: Session = Depends(get_db)):
    return [{"id": r.id, "email": r.email, "full_name": r.full_name, "role": r.role,
             "is_active": r.is_active} for r in db.query(models.User).order_by(models.User.id).all()]


@app.put("/admin/users/{id}")
def update_user(id: int, change: schemas.UserUpdate, user=Depends(require_role("admin")), db: Session = Depends(get_db)):
    target = db.get(models.User, id)
    if target is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài khoản")
    if id == user.id and (change.role not in (None, "admin") or change.is_active is False):
        raise HTTPException(status_code=400, detail="Không thể tự khóa hoặc hạ quyền Admin đang đăng nhập")
    for key, value in change.model_dump(exclude_none=True).items():
        setattr(target, key, value)
    db.commit()
    return {"message": "Đã cập nhật"}


@app.delete("/admin/users/{id}")
def delete_user(id: int, user=Depends(require_role("admin")), db: Session = Depends(get_db)):
    if id == user.id:
        raise HTTPException(status_code=400, detail="Không thể tự xóa Admin đang đăng nhập")
    target = db.get(models.User, id)
    if target is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài khoản")
    db.query(models.History).filter(models.History.user_id == id).delete()
    db.delete(target)
    db.commit()
    return {"message": "Đã xóa"}


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Mô hình chưa sẵn sàng")
    return {"status": "ok"}


@app.get("/admin/system")
def system(user=Depends(require_role("admin")), db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    return {"database": "ok", "history_count": db.query(models.History).count(),
            "model": type(model.named_steps["classifier"]).__name__ if model else None,
            "threshold": threshold, "model_file": os.path.basename(MODEL_PATH),
            "metrics": getattr(model,"holdout_metrics_",None), "metadata": getattr(model,"training_metadata_",None)}


@app.post("/admin/model/reload")
def reload(user=Depends(require_role("admin"))):
    try:
        load_model()
    except Exception:
        logging.exception("Model reload failed")
        raise HTTPException(status_code=503, detail="Không nạp được mô hình mới")
    return {"message": "Đã nạp lại mô hình"}


@app.delete("/admin/history/{id}")
def delete_history(id: int, user=Depends(require_role("admin")), db: Session = Depends(get_db)):
    row = db.get(models.History, id)
    if row is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy lịch sử")
    db.delete(row)
    db.commit()
    return {"message": "Đã xóa lịch sử"}


@app.get("/doctor/export")
def export(format: str = "xlsx", user=Depends(require_role("doctor")), db: Session = Depends(get_db)):
    import io
    import zipfile
    from xml.sax.saxutils import escape
    from fastapi.responses import Response
    rows = [["Patient ID", "Name", "Email", "Date", "Glucose", "BMI", "Risk (%)", "Result"]]
    for patient in db.query(models.User).filter(models.User.role == "user").all():
        for h in history_for(patient.id, db):
            rows.append([patient.id, patient.full_name, patient.email, h["date"], h.get("glucose"), h.get("bmi"), h.get("risk"), h["prediction_result"]])
    if format not in ("xlsx", "pdf"):
        raise HTTPException(status_code=422, detail="Định dạng phải là xlsx hoặc pdf")
    if format == "pdf":
        # Standard PDF with a core font. Transliterate Vietnamese for font portability.
        import unicodedata
        lines = ["STROKE SCREENING REPORT", "Reference screening only - not a medical diagnosis."]
        for row in rows:
            line = " | ".join(str(v if v is not None else "") for v in row)
            line = unicodedata.normalize("NFKD", line.replace("đ", "d").replace("Đ", "D")).encode("ascii", "ignore").decode()
            lines.extend(line[i:i+110] for i in range(0, len(line), 110))
        objects = [b"", b"", b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
        pages = []
        for offset in range(0, len(lines), 48):
            page_id, stream_id = len(objects)+1, len(objects)+2
            pages.append(page_id)
            content = "BT /F1 9 Tf 32 800 Td 15 TL "
            for line in lines[offset:offset+48]:
                safe = line.replace(chr(92), chr(92)*2).replace("(", chr(92)+"(").replace(")", chr(92)+")")
                content += "(" + safe + ") Tj T* "
            content = (content + "ET").encode("ascii")
            objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 842 842] /Resources << /Font << /F1 3 0 R >> >> /Contents {stream_id} 0 R >>".encode())
            objects.append(f"<< /Length {len(content)} >>\nstream\n".encode()+content+b"\nendstream")
        objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
        objects[1] = ("<< /Type /Pages /Count " + str(len(pages)) + " /Kids [" + " ".join(f"{n} 0 R" for n in pages) + "] >>").encode()
        pdf, positions = bytearray(b"%PDF-1.4\n"), [0]
        for n, obj in enumerate(objects, 1):
            positions.append(len(pdf))
            pdf.extend(f"{n} 0 obj\n".encode()+obj+b"\nendobj\n")
        start = len(pdf)
        pdf.extend(f"xref\n0 {len(positions)}\n0000000000 65535 f \n".encode())
        for pos in positions[1:]:
            pdf.extend(f"{pos:010d} 00000 n \n".encode())
        pdf.extend(f"trailer << /Size {len(positions)} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF".encode())
        return Response(bytes(pdf), media_type="application/pdf", headers={"Content-Disposition": 'attachment; filename="screening.pdf"'})
    xml_rows = []
    for row in rows:
        cells = []
        for value in row:
            if isinstance(value, (int, float)):
                cells.append(f'<c><v>{value}</v></c>')
            else:
                cells.append('<c t="inlineStr"><is><t>' + escape(str(value or "")) + '</t></is></c>')
        xml_rows.append('<row>' + ''.join(cells) + '</row>')
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>')
        z.writestr('_rels/.rels', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        z.writestr('xl/workbook.xml', '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Screening" sheetId="1" r:id="rId1"/></sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>')
        z.writestr('xl/worksheets/sheet1.xml', '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>' + ''.join(xml_rows) + '</sheetData></worksheet>')
    return Response(output.getvalue(), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": 'attachment; filename="screening.xlsx"'})


@app.get("/config")
def config():
    return {"demo_mode": DEMO_MODE}


@app.post("/auth/demo/{role}", response_model=schemas.Token)
def demo_login(role: str, db: Session = Depends(get_db)):
    if not DEMO_MODE or role not in ("user", "doctor", "admin"):
        raise HTTPException(status_code=404, detail="Không có chế độ demo")
    import secrets
    email = role + "@demo.strokecare.local"
    user = db.query(models.User).filter(models.User.email == email).first()
    if user is None:
        user = models.User(email=email, full_name={"user":"Người dùng mẫu", "doctor":"Bác sĩ Nguyễn Minh", "admin":"Quản trị viên mẫu"}[role], role=role,
                           region="Hà Nội", hashed_password=security.hash_password(secrets.token_urlsafe(24)))
        db.add(user); db.commit(); db.refresh(user)
    if not user.is_active or user.role != role:
        raise HTTPException(status_code=403, detail="Tài khoản demo đã bị khóa hoặc đổi vai trò")
    token=security.create_access_token({"sub":user.email})
    if role=="user" and model is not None and db.query(models.History).filter(models.History.user_id==user.id).count()==0:
        from datetime import datetime,timezone,timedelta
        for index,(age,glucose,bmi) in enumerate([(25,85,22.1),(45,110,24.2),(65,180,28.1)]):
            sample=schemas.PredictRequest(age=age,gender="Male",hypertension=int(index==2),heart_disease=0,ever_married="Yes",work_type="Private",Residence_type="Urban",avg_glucose_level=glucose,bmi=bmi,smoking_status="never smoked")
            predict(sample,"Bearer "+token,db,None,None)
            record=db.query(models.History).filter(models.History.user_id==user.id).order_by(models.History.id.desc()).first()
            record.timestamp=datetime.now(timezone.utc)-timedelta(days=2-index)
            db.commit()
    elif role in ("doctor","admin"):
        demo_login("user",db)
    return {"access_token":token,"token_type":"bearer","role":user.role}


@app.post("/doctor/patients", status_code=201)
def create_patient(data: schemas.PatientCreate, user=Depends(require_role("doctor")), db: Session = Depends(get_db)):
    import secrets
    email=data.email or ("patient-"+secrets.token_hex(8)+"@patient.local")
    from pydantic import ValidationError
    try:
        valid=schemas.UserRegister(full_name=data.full_name,email=email,password=secrets.token_urlsafe(24),region=data.region)
    except ValidationError as e:
        raise HTTPException(status_code=422,detail=str(e))
    if db.query(models.User).filter(models.User.email==valid.email).first():
        raise HTTPException(status_code=400,detail="Email đã tồn tại; hãy chọn hồ sơ có sẵn")
    patient=models.User(email=valid.email,full_name=valid.full_name,region=valid.region,role="user",hashed_password=security.hash_password(valid.password))
    db.add(patient);db.commit();db.refresh(patient)
    return {"id":patient.id,"full_name":patient.full_name,"email":patient.email,"region":patient.region}


@app.get("/admin/stats")
def stats(user=Depends(require_role("admin")),db: Session=Depends(get_db)):
    from collections import Counter
    all_users=db.query(models.User).all();histories=db.query(models.History).all()
    high=0; ages=Counter();regions=Counter();by_id={u.id:u for u in all_users}
    for row in histories:
        try:d=json.loads(row.action)
        except (ValueError,TypeError):continue
        if d.get("prediction_result",int(d.get("risk",0)>=18)):
            high+=1
            age=d.get("input",{}).get("age")
            ages["Không rõ" if age is None else "Dưới 40" if age<40 else "40–59" if age<60 else "Từ 60"]+=1
            patient=by_id.get(row.user_id);regions[(patient.region if patient else None) or "Chưa cập nhật"]+=1
    return {"users":len(all_users),"patients":sum(u.role=="user" for u in all_users),"doctors":sum(u.role=="doctor" for u in all_users),
            "predictions":len(histories),"high_risk":high,"high_risk_rate":high/max(1,len(histories)),"high_by_age":dict(ages),"high_by_region":dict(regions)}


@app.get("/admin/history")
def all_history(user=Depends(require_role("admin")),db: Session=Depends(get_db)):
    result=[]
    for account in db.query(models.User).all():
        for h in history_for(account.id,db):result.append({**h,"patient_name":account.full_name,"patient_id":account.id})
    return sorted(result,key=lambda h:h['id'],reverse=True)


@app.get("/doctor/template")
def template(user=Depends(require_role("doctor"))):
    from fastapi.responses import Response
    return Response("full_name,age,gender,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status\nNguyen Van An,65,Male,1,0,Yes,Private,Urban,200,28.1,formerly smoked\n",media_type="text/csv",headers={"Content-Disposition":'attachment; filename="screening_template.csv"'})


@app.post("/doctor/batch")
async def batch(file: UploadFile=File(...),cutoff: float | None=Query(None,gt=0,lt=1),authorization: str=Header(None),db: Session=Depends(get_db)):
    import io,zipfile
    from xml.etree import ElementTree as ET
    from pydantic import ValidationError
    user=get_current_user(authorization,db)
    if user.role!="doctor":raise HTTPException(status_code=403,detail="Chỉ bác sĩ được sàng lọc hàng loạt")
    content=await file.read(2*1024*1024+1)
    if len(content)>2*1024*1024:raise HTTPException(status_code=413,detail="Tệp tối đa 2 MB")
    try:
        if (file.filename or '').lower().endswith('.csv'):
            data=pd.read_csv(io.BytesIO(content),encoding='utf-8-sig')
        elif (file.filename or '').lower().endswith('.xlsx'):
            ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                if sum(i.file_size for i in z.infolist())>20*1024*1024:raise ValueError('Tệp Excel quá lớn khi giải nén')
                strings=[]
                if 'xl/sharedStrings.xml' in z.namelist():
                    strings=[''.join(n.itertext()) for n in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si',ns)]
                sheet=ET.fromstring(z.read('xl/worksheets/sheet1.xml'));rows=[]
                for row in sheet.findall('m:sheetData/m:row',ns):
                    cells={}
                    for cell in row.findall('m:c',ns):
                        ref=''.join(c for c in cell.get('r','A1') if c.isalpha());index=0
                        for c in ref:index=index*26+ord(c)-64
                        v=cell.find('m:v',ns);value=v.text if v is not None else ''
                        if cell.get('t')=='s':value=strings[int(value)]
                        elif cell.get('t')=='inlineStr':value=''.join(cell.find('m:is',ns).itertext())
                        cells[index-1]=value
                    if cells:rows.append([cells.get(i,'') for i in range(max(cells)+1)])
                if not rows:raise ValueError('Tệp Excel rỗng')
                data=pd.DataFrame(rows[1:],columns=rows[0])
        else:raise ValueError('Chỉ hỗ trợ CSV UTF-8 hoặc XLSX')
        if len(data)==0 or len(data)>100:raise ValueError('Mỗi tệp cần từ 1 đến 100 bệnh nhân')
        required=list(schemas.PredictRequest.model_fields)
        if not set(required).issubset(data.columns):raise ValueError('Thiếu cột: '+', '.join(set(required)-set(data.columns)))
    except Exception as e:raise HTTPException(status_code=422,detail=str(e))
    results=[]
    for index,row in data.iterrows():
        try:
            request=schemas.PredictRequest.model_validate({k:row[k] for k in required})
            raw_name=row.get('full_name','');name=str(raw_name).strip() if pd.notna(raw_name) else ''
            raw_id=row.get('patient_id');patient_id=None
            if pd.notna(raw_id) and str(raw_id).strip():patient_id=int(raw_id)
            if patient_id is None:
                created=create_patient(schemas.PatientCreate(full_name=name or f'Bệnh nhân dòng {index+2}'),user,db)
                patient_id=created['id']
            response=predict(request,authorization,db,patient_id,cutoff)
            results.append({'row':int(index)+2,'patient_id':patient_id,'full_name':name,**response.model_dump(),'status':'ok'})
        except (ValidationError,HTTPException,ValueError) as e:
            results.append({'row':int(index)+2,'status':'error','message':str(e.detail) if isinstance(e,HTTPException) else str(e)})
    return {'results':results,'success':sum(r['status']=='ok' for r in results),'total':len(results)}


# Same-origin local entry point also works when the separate frontend port is occupied.
from fastapi.staticfiles import StaticFiles
app.mount("/web", StaticFiles(directory=os.path.join(BASE_DIR,"frontend"),html=True),name="web")
