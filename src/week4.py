

import os
import sys
import numpy as np
import joblib
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR    = os.path.join(BASE_DIR, "outputs", "models")


# ──────────────────────────────────────────────────────────────
# 1. Nạp dữ liệu đã tiền xử lý từ Tuần 3
# ──────────────────────────────────────────────────────────────
def load_train_data():
    """Nạp tập train đã SMOTE từ Tuần 3."""
    train = np.load(os.path.join(DATA_DIR, "train_resampled.npz"))
    X_train, y_train = train["X"], train["y"]
    print(f"📂 Đã nạp tập train (SMOTE): {X_train.shape[0]:,} mẫu × {X_train.shape[1]} features")
    print(f"   ↳ Không đột quỵ: {int((y_train == 0).sum()):,}")
    print(f"   ↳ Đột quỵ:       {int((y_train == 1).sum()):,}")
    return X_train, y_train


# ──────────────────────────────────────────────────────────────
# 2. Huấn luyện các mô hình baseline
# ──────────────────────────────────────────────────────────────
def train_logistic_regression(X_train, y_train):
    """Huấn luyện Logistic Regression."""
    lr = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    lr.fit(X_train, y_train)
    print("✅ Logistic Regression – huấn luyện xong.")
    return lr


def train_decision_tree(X_train, y_train):
    """Huấn luyện Decision Tree."""
    dt = DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
        max_depth=10,
    )
    dt.fit(X_train, y_train)
    print("✅ Decision Tree – huấn luyện xong.")
    return dt


# ──────────────────────────────────────────────────────────────
# 3. Lưu mô hình
# ──────────────────────────────────────────────────────────────
def save_models(models: dict):
    """Lưu các mô hình ra file .pkl."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, model in models.items():
        safe_name = name.lower().replace(" ", "_")
        path = os.path.join(MODEL_DIR, f"{safe_name}_baseline.pkl")
        joblib.dump(model, path)
        print(f"💾 Đã lưu {name} → {path}")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    # 1. Nạp dữ liệu
    X_train, y_train = load_train_data()

    # 2. Huấn luyện
    lr = train_logistic_regression(X_train, y_train)
    dt = train_decision_tree(X_train, y_train)

    models = {
        "Logistic Regression": lr,
        "Decision Tree": dt,
    }

    # 3. Lưu mô hình
    save_models(models)

    print("\n✅ Week 4 done – Baseline models trained & saved!")


if __name__ == "__main__":
    main()
