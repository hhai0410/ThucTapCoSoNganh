from enum import Enum
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional

# ==========================================
# 1. KHAI BÁO CÁC ENUM DỮ LIỆU ĐẦU VÀO
# ==========================================


class GenderEnum(str, Enum):
    MALE = "Male"
    FEMALE = "Female"
    OTHER = "Other"

class EverMarriedEnum(str, Enum):
    YES = "Yes"
    NO = "No"

class WorkTypeEnum(str, Enum):
    PRIVATE = "Private"
    SELF_EMPLOYED = "Self-employed"
    GOVT_JOB = "Govt_job"
    CHILDREN = "children"
    NEVER_WORKED = "Never_worked"

class ResidenceTypeEnum(str, Enum):
    URBAN = "Urban"
    RURAL = "Rural"

class SmokingStatusEnum(str, Enum):
    FORMERLY_SMOKED = "formerly smoked"
    NEVER_SMOKED = "never smoked"
    SMOKES = "smokes"
    UNKNOWN = "Unknown"


# ==========================================
# 2. SCHEMAS CHO AUTHENTICATION & USER
# ==========================================

class UserRegister(BaseModel):
    email: str = Field(..., min_length=3, description="Email đăng nhập")
    password: str = Field(..., min_length=6, description="Mật khẩu")
    role: Optional[str] = "user"
    region: Optional[str] = None

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserResponse(BaseModel):
    id: int
    email: str
    role: str
    region: Optional[str]

    model_config = ConfigDict(from_attributes=True)


# ==========================================
# 3. SCHEMA DỰ ĐOÁN (PREDICT REQUEST & RESPONSE)
# ==========================================

class PredictRequest(BaseModel):
    age: float = Field(
        ..., 
        ge=0.0, 
        le=120.0, 
        description="Độ tuổi (từ 0 đến 120 tuổi)", 
        json_schema_extra={"example": 45.5}
    )
    gender: GenderEnum = Field(
        ..., 
        description="Giới tính: Male, Female, hoặc Other", 
        json_schema_extra={"example": "Male"}
    )
    hypertension: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Bệnh cao huyết áp: 0 (Không), 1 (Có)", 
        json_schema_extra={"example": 0}
    )
    heart_disease: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Bệnh tim mạch: 0 (Không), 1 (Có)", 
        json_schema_extra={"example": 1}
    )
    ever_married: EverMarriedEnum = Field(
        ..., 
        description="Tình trạng hôn nhân: Yes hoặc No", 
        json_schema_extra={"example": "Yes"}
    )
    work_type: WorkTypeEnum = Field(
        ..., 
        description="Loại công việc: Private, Self-employed, Govt_job, children, Never_worked", 
        json_schema_extra={"example": "Private"}
    )
    Residence_type: ResidenceTypeEnum = Field(
        ..., 
        description="Môi trường sống: Urban (Thành thị), Rural (Nông thôn)", 
        json_schema_extra={"example": "Urban"}
    )
    avg_glucose_level: float = Field(
        ..., 
        gt=0.0, 
        description="Lượng đường trong máu (mg/dL)", 
        json_schema_extra={"example": 105.67}
    )
    bmi: float = Field(
        ..., 
        gt=0.0, 
        description="Chỉ số khối cơ thể (BMI)", 
        json_schema_extra={"example": 24.3}
    )
    smoking_status: SmokingStatusEnum = Field(
        ..., 
        description="Tình trạng hút thuốc: formerly smoked, never smoked, smokes, Unknown", 
        json_schema_extra={"example": "never smoked"}
    )

class ShapItem(BaseModel):
    feature: str
    value: float


class PredictResponse(BaseModel):
    prediction_result: int = Field(..., description="Kết quả dự đoán đột quỵ: 0 (Không bị), 1 (Có nguy cơ)")
    risk_probability: float = Field(..., description="Độ tin cậy của mô hình (ví dụ: 0.95)")
    message: str = Field(default="Dự đoán thành công", description="Thông báo phản hồi")
    shap: Optional[List[ShapItem]] = []