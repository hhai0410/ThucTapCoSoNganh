from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Dùng SQLite cho nhanh hoặc đổi thành PostgreSQL/MySQL nếu cần
SQLALCHEMY_DATABASE_URL = "sqlite:///./app.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency lấy DB session cho FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()