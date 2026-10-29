"""Run from repository root: venv/Scripts/python backend/create_admin.py"""
import getpass
from app.database import SessionLocal
from app.main import app
from app import models, schemas, security

def main():
    email = input("Admin email: ").strip().lower()
    password = getpass.getpass("Password: ")
    data = schemas.UserRegister(full_name="Administrator", email=email, password=password)
    with SessionLocal() as db:
        if db.query(models.User).filter(models.User.email == email).first():
            raise SystemExit("Account already exists; use an admin to change its role.")
        db.add(models.User(full_name=data.full_name, email=data.email, hashed_password=security.hash_password(password), role="admin"))
        db.commit()
    print("Administrator created.")

if __name__ == "__main__":
    main()
