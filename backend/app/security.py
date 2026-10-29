from datetime import datetime, timedelta, timezone
from passlib.context import CryptContext
from jose import jwt

import os
import secrets
from pathlib import Path

_key_path = Path(__file__).resolve().parents[1] / ".jwt_secret"
if not os.getenv("JWT_SECRET_KEY") and not _key_path.exists():
    try:
        with _key_path.open("x") as f:
            f.write(secrets.token_urlsafe(48))
    except FileExistsError:
        pass
SECRET_KEY = os.getenv("JWT_SECRET_KEY") or _key_path.read_text().strip()  # Thay đổi key này trong production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 ngày

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    if len(plain_password.encode("utf-8")) > 72:
        return False
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)