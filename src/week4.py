import os
import sys
import numpy as np
import joblib
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR    = os.path.join(BASE_DIR, "outputs", "models")


# ──────────────────────────────────────────────────────────────
# 1. Nạp dữ liệu và preprocessor từ Tuần 3
# ──────────────────────────────────────────────────────────────
def load_data_and_preprocessor():
    """Nạp tập train đã SMOTE và bộ tiền xử lý từ Tuần 3."""
    # Nạp dữ liệu train đã qua SMOTE
    train = np.load(os.path.join(DATA_DIR, "train_resampled.npz"))
    X_train, y_train = train["X"], train["y"]
    
    # Nạp preprocessor đã fit từ tuần 3
    preprocessor_path = os.path.join(DATA_DIR, "preprocessor.pkl")
    preprocessor = joblib.load(preprocessor_path)
    
    print(f"📂 Đã nạp tập train (SMOTE): {X_train.shape[0]:,} mẫu × {X_train.shape[1]} features")
    print(f"   ↳ Không đột quỵ: {int((y_train == 0).sum()):,}")
    print(f"   ↳ Đột quỵ:       {int((y_train == 1).sum()):,}")
    print(f"📂 Đã nạp Preprocessor.")
    
    return X_train, y_train, preprocessor


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
# 3. Tạo Pipeline và Lưu mô hình
# ──────────────────────────────────────────────────────────────
def save_pipeline_models(models: dict, preprocessor):
    """Gộp Preprocessor và Model thành Pipeline hoàn chỉnh rồi lưu."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    for name, model in models.items():
        # LƯU Ý QUAN TRỌNG: Pipeline dùng cho dự đoán (inference) chỉ cần Preprocessor và Model.
        # KHÔNG đưa SMOTE vào pipeline này vì SMOTE chỉ dùng để sinh thêm dữ liệu trong quá trình huấn luyện.
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        safe_name = name.lower().replace(" ", "_")
        path = os.path.join(MODEL_DIR, f"{safe_name}_pipeline.pkl")
        joblib.dump(pipeline, path)
        print(f"💾 Đã lưu Pipeline {name} → {path}")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    # 1. Nạp dữ liệu và preprocessor
    X_train, y_train, preprocessor = load_data_and_preprocessor()

    # 2. Huấn luyện mô hình (trên dữ liệu đã xử lý và SMOTE)
    lr = train_logistic_regression(X_train, y_train)
    dt = train_decision_tree(X_train, y_train)

    models = {
        "Logistic Regression": lr,
        "Decision Tree": dt,
    }

    # 3. Đóng gói thành Pipeline và lưu file pkl
    save_pipeline_models(models, preprocessor)

    print("\n✅ Week 4 done – Complete Pipelines trained & saved!")


if __name__ == "__main__":
    main()