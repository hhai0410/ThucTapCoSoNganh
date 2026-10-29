from datetime import datetime,timezone
from sqlalchemy import Boolean, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from .database import Base
from .privacy import EncryptedText
# create table for database
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(EncryptedText, nullable=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="user")  # admin, doctor, user...
    is_active = Column(Boolean, default=True, nullable=False)
    region = Column(EncryptedText, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    history = relationship("History", back_populates="user")

class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    patient_code = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(EncryptedText, nullable=False)
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)
    
    predictions = relationship("Prediction", back_populates="patient")

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    input_data = Column(EncryptedText, nullable=False)  # Lưu JSON chuỗi input
    result = Column(String, nullable=False)    # Kết quả dự đoán
    confidence = Column(Float, nullable=True)  # Độ tin cậy (nếu có)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    patient = relationship("Patient", back_populates="predictions")

class History(Base):
    __tablename__ = "history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    action = Column(EncryptedText, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="history")