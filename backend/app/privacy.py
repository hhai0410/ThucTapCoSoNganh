"""Encrypt names/medical payloads; transparently read existing plaintext rows."""
import os
from pathlib import Path
from cryptography.fernet import Fernet
from sqlalchemy.types import TypeDecorator, Text
path=Path(__file__).resolve().parents[1]/'.data_key'
if not os.getenv('DATA_ENCRYPTION_KEY') and not path.exists():
    try:
        with path.open('xb') as f:f.write(Fernet.generate_key())
    except FileExistsError:pass
cipher=Fernet(os.getenv('DATA_ENCRYPTION_KEY').encode() if os.getenv('DATA_ENCRYPTION_KEY') else path.read_bytes().strip())
class EncryptedText(TypeDecorator):
    impl=Text
    cache_ok=True
    def process_bind_param(self,value,dialect):
        return 'enc:v1:'+cipher.encrypt(str(value).encode()).decode() if value is not None else None
    def process_result_value(self,value,dialect):
        return cipher.decrypt(value[7:].encode()).decode() if value and value.startswith('enc:v1:') else value


def encrypt_legacy_rows(engine):
    from sqlalchemy import text
    with engine.begin() as conn:
        for table, columns in {'users':['full_name','region'],'patients':['full_name'],'predictions':['input_data'],'history':['action']}.items():
            for column in columns:
                rows=conn.execute(text(f'SELECT id, {column} FROM {table} WHERE {column} IS NOT NULL')).all()
                for row_id,value in rows:
                    if not value.startswith('enc:v1:'):
                        encrypted='enc:v1:'+cipher.encrypt(value.encode()).decode()
                        conn.execute(text(f'UPDATE {table} SET {column}=:v WHERE id=:id'),{'v':encrypted,'id':row_id})
