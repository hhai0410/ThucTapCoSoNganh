"""
Tuần 3 – Feature Engineering & Xử lý mất cân bằng dữ liệu
=============================================================
Thứ tự xử lý (tránh Data Leakage):
    Split → Preprocess (fit trên train) → SMOTE (chỉ train)

Output (data/processed/):
    preprocessor.pkl      – ColumnTransformer đã fit
    train_resampled.npz   – X_train + y_train sau SMOTE
    train_original.npz    – X_train gốc (chưa SMOTE)
    test.npz              – X_test + y_test (phân phối gốc)
    feature_names.pkl     – tên 21 features sau OneHotEncoder
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Import build_preprocessor() từ preprocessing.py cùng thư mục
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import build_preprocessor

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW     = os.path.join(BASE_DIR, "data", "processed", "stroke_clean.csv")
OUTPUT_DIR   = os.path.join(BASE_DIR, "data", "processed")

print(DATA_RAW)
FEATURE_COLS = [
    "gender", "age", "hypertension", "heart_disease",
    "ever_married", "work_type", "Residence_type",
    "avg_glucose_level", "bmi", "smoking_status",
]
TARGET_COL = "stroke"


def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Đọc CSV và tách X (10 cột) / y (stroke)."""
    df = pd.read_csv(path)
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    return X, y


def split_data(X, y, test_size=0.2):
    """Chia train/test 80-20, stratify theo nhãn."""
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

 
def preprocess(X_train, X_test):
    """Fit preprocessor trên train, transform cả 2 tập."""
    #Hãy nhớ: preprocessor.pkl là bộ quy tắc biến đổi dữ liệu đã học từ train, chứ không phải mô hình dự đoán đột quỵ và cũng không phải dữ liệu đã biến đổi.
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)
    feature_names = list(preprocessor.get_feature_names_out())
    return X_train_proc, X_test_proc, preprocessor, feature_names


def apply_smote(X_train, y_train):
    """SMOTE chỉ trên tập train đã transform."""
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res


def save_artifacts(output_dir, preprocessor, feature_names,
                   X_train_res, y_train_res,
                   X_train_orig, y_train_orig,
                   X_test, y_test):
    """Lưu toàn bộ artifacts vào output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.pkl"))
    joblib.dump(feature_names, os.path.join(output_dir, "feature_names.pkl"))

    np.savez(os.path.join(output_dir, "train_resampled.npz"),
             X=X_train_res, y=y_train_res)
    np.savez(os.path.join(output_dir, "train_original.npz"),
             X=X_train_orig, y=y_train_orig)
    np.savez(os.path.join(output_dir, "test.npz"),
             X=X_test, y=y_test)


def main():
    # 1. Đọc dữ liệu
    X, y = load_data(DATA_RAW)

    # 2. Chia train / test
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 3. Tiền xử lý (fit trên train)
    X_train_proc, X_test_proc, preprocessor, feature_names = preprocess(X_train, X_test)

    # 4. SMOTE (chỉ trên train)
    X_train_res, y_train_res = apply_smote(X_train_proc, y_train)

    # 5. Lưu artifacts
    save_artifacts(
        OUTPUT_DIR, preprocessor, feature_names, #feature_names là mảng tên các cột sau khi tiền xử lí 
        X_train_res, y_train_res,       #Lưu lại tập dữ liệu sau khi đã được xử lí bằng SMOTE
        X_train_proc, y_train.values,    #Phải lưu lại để so sánh kết quả huấn luyện của mô hình trên tập test trước và sau khi áp dụng SMOTE
        X_test_proc, y_test.values,      #Lưu lại dữ liệu để kiểm tra mô hình sau huấn luyện
    )

    print("✅ Week 3 done – all artifacts saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()