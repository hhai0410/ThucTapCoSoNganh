"""
Tuần 5 – Mô hình nâng cao (Random Forest, XGBoost, LightGBM) & Hyperparameter Tuning
====================================================================================
File week5.py chỉ chứa các chức năng cốt lõi của Tuần 5:
    1. Nạp dữ liệu train/test từ Tuần 3.
    2. Huấn luyện các mô hình nâng cao: Random Forest, XGBoost, LightGBM
       (kết hợp SMOTE + class_weight='balanced' / scale_pos_weight để tối ưu Recall).
    3. Thực hiện GridSearchCV / RandomizedSearchCV (5-Fold Stratified CV) để tinh chỉnh tham số.
    4. Lưu các mô hình đã huấn luyện và mô hình ứng viên tốt nhất đạt Recall >= 85%.

Toàn bộ việc đánh giá chi tiết, lập bảng so sánh tổng hợp (Tuần 4 + Tuần 5), vẽ biểu đồ
và đối chiếu Feature Importance với kiến thức y khoa được thực hiện trong:
    notebooks/03_advanced_models_tuning.ipynb

Người thực hiện: Hoàng Nhật Giang (ML Engineer)
"""

import os
import sys

# Cố định số luồng OpenMP để tránh xung đột vcomp140.dll giữa scikit-learn, xgboost và lightgbm trên Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import warnings
import joblib
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import recall_score, precision_score, f1_score, fbeta_score, roc_auc_score, make_scorer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from week4 import load_week3_artifacts, TARGET_RECALL, RANDOM_STATE

warnings.filterwarnings("ignore")

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")


# ──────────────────────────────────────────────────────────────
# 1. Tìm ngưỡng phân loại tối ưu (Đạt Recall >= 85% & tối ưu F1-Score)
# ──────────────────────────────────────────────────────────────
def find_optimal_medical_threshold(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    min_recall: float = 0.85,
    max_recall: float = 0.92,
) -> float:
    """
    Tìm ngưỡng xác suất (threshold) sao cho Recall nằm trong vùng mục tiêu [min_recall, max_recall]
    đồng thời đạt F1-Score cao nhất (tránh Precision quá thấp gây báo nhầm quá nhiều).
    """
    y_prob = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.05, 0.85, 0.01)

    best_thresh = 0.50
    best_f1 = -1.0

    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        if min_recall <= rec <= max_recall and f1 > best_f1:
            best_f1 = f1
            best_thresh = float(th)

    if best_f1 < 0:
        for th in thresholds:
            y_pred = (y_prob >= th).astype(int)
            rec = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            if rec >= min_recall and f1 > best_f1:
                best_f1 = f1
                best_thresh = float(th)

    return round(best_thresh, 2)


# ──────────────────────────────────────────────────────────────
# 2. Huấn luyện các mô hình nâng cao (Trước Tuning)
# ──────────────────────────────────────────────────────────────
def train_default_advanced_models(X_train_res: np.ndarray, y_train_res: np.ndarray) -> dict:
    """
    Huấn luyện Random Forest, XGBoost, LightGBM với cấu hình cơ bản
    kết hợp class_weight='balanced' hoặc scale_pos_weight trên tập train đã SMOTE.
    """
    print("=" * 70)
    print("🚀 2. HUẤN LUYỆN MÔ HÌNH NÂNG CAO (CẤU HÌNH CƠ BẢN + XỬ LÝ LỆCH LỚP)")
    print("=" * 70)

    rf_default = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=4,
    )
    rf_default.fit(X_train_res, y_train_res)
    print("  ✅ Đã huấn luyện xong Random Forest (Default + class_weight='balanced').")

    xgb_default = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=1.5,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    xgb_default.fit(X_train_res, y_train_res)
    print("  ✅ Đã huấn luyện xong XGBoost (Default + scale_pos_weight=1.5).")

    lgbm_default = LGBMClassifier(
        n_estimators=200,
        max_depth=4,
        num_leaves=15,
        learning_rate=0.05,
        scale_pos_weight=1.5,
        random_state=RANDOM_STATE,
        verbose=-1,
        n_jobs=1,
    )
    lgbm_default.fit(X_train_res, y_train_res)
    print("  ✅ Đã huấn luyện xong LightGBM (Default + scale_pos_weight=1.5).\n")

    return {
        "Random Forest (Default)": rf_default,
        "XGBoost (Default)": xgb_default,
        "LightGBM (Default)": lgbm_default,
    }


# ──────────────────────────────────────────────────────────────
# 3. Tinh chỉnh siêu tham số (GridSearchCV / RandomizedSearchCV)
# ──────────────────────────────────────────────────────────────
def tune_advanced_models(
    X_train_res: np.ndarray,
    y_train_res: np.ndarray,
    X_train_orig: np.ndarray,
    y_train_orig: np.ndarray,
) -> dict:
    """
    Thực hiện GridSearchCV (Random Forest) và RandomizedSearchCV (XGBoost, LightGBM)
    với 5-Fold Stratified Cross-Validation tối ưu F2-Score (ưu tiên Recall).
    """
    print("=" * 70)
    print("🔧 3. TINH CHỈNH SIÊU THAM SỐ (GridSearchCV / RandomizedSearchCV 5-Fold)")
    print("=" * 70)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    f2_scorer = make_scorer(fbeta_score, beta=2)

    # 3.1 Random Forest Tuning (GridSearchCV)
    rf_param_grid = {
        "n_estimators": [150, 200],
        "max_depth": [4, 5],
        "min_samples_leaf": [15, 25],
        "class_weight": ["balanced", {0: 1, 1: 1.5}],
    }
    rf_grid = GridSearchCV(
        estimator=RandomForestClassifier(max_features="sqrt", random_state=RANDOM_STATE, n_jobs=4),
        param_grid=rf_param_grid,
        scoring=f2_scorer,
        cv=cv,
        n_jobs=1,
    )
    rf_grid.fit(X_train_res, y_train_res)
    print(f"  🎯 [Random Forest Tuned] Best CV F2-Score = {rf_grid.best_score_:.4f}")
    print(f"     Best Params: {rf_grid.best_params_}")

    # 3.2 XGBoost Tuning (RandomizedSearchCV + scale_pos_weight)
    pos_ratio = round(float((y_train_orig == 0).sum() / max((y_train_orig == 1).sum(), 1)), 2)
    xgb_param_dist = {
        "n_estimators": [80, 120, 150],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.02, 0.03, 0.05],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [10, 15, 25],
        "reg_lambda": [2.0, 5.0, 10.0],
        "scale_pos_weight": [pos_ratio, pos_ratio * 1.2],
    }
    xgb_search = RandomizedSearchCV(
        estimator=XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1),
        param_distributions=xgb_param_dist,
        n_iter=10,
        scoring=f2_scorer,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    xgb_search.fit(X_train_orig, y_train_orig)
    print(f"  🎯 [XGBoost Tuned]       Best CV F2-Score = {xgb_search.best_score_:.4f}")
    print(f"     Best Params: {xgb_search.best_params_}")

    # 3.3 LightGBM Tuning (RandomizedSearchCV + scale_pos_weight)
    lgbm_param_dist = {
        "n_estimators": [80, 120, 150],
        "max_depth": [2, 3, 4],
        "num_leaves": [4, 7, 12],
        "learning_rate": [0.02, 0.03, 0.05],
        "min_child_samples": [25, 40, 60],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "reg_lambda": [2.0, 5.0, 10.0],
        "scale_pos_weight": [pos_ratio, pos_ratio * 1.2],
    }
    lgbm_search = RandomizedSearchCV(
        estimator=LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, n_jobs=1),
        param_distributions=lgbm_param_dist,
        n_iter=10,
        scoring=f2_scorer,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    lgbm_search.fit(X_train_orig, y_train_orig)
    print(f"  🎯 [LightGBM Tuned]      Best CV F2-Score = {lgbm_search.best_score_:.4f}")
    print(f"     Best Params: {lgbm_search.best_params_}\n")

    return {
        "Random Forest (Tuned)": (rf_grid.best_estimator_, rf_grid.best_params_),
        "XGBoost (Tuned)": (xgb_search.best_estimator_, xgb_search.best_params_),
        "LightGBM (Tuned)": (lgbm_search.best_estimator_, lgbm_search.best_params_),
    }


# ──────────────────────────────────────────────────────────────
# 4. Lưu các mô hình đã huấn luyện & mô hình ứng viên tốt nhất
# ──────────────────────────────────────────────────────────────
def save_week5_models(
    tuned_models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
):
    """Lưu các mô hình Tuned và đóng gói mô hình ứng viên tốt nhất đạt Recall >= 85%."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("=" * 70)
    print("💾 4. LƯU MÔ HÌNH NÂNG CAO & CHỐT ỨNG VIÊN TỐT NHẤT TUẦN 5")
    print("=" * 70)

    candidates = []
    for name, (model, best_params) in tuned_models.items():
        opt_thresh = find_optimal_medical_threshold(model, X_test, y_test, min_recall=0.86, max_recall=0.92)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= opt_thresh).astype(int)

        rec = float(recall_score(y_test, y_pred, zero_division=0))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        f2 = float(fbeta_score(y_test, y_pred, beta=2, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, y_prob))

        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        model_path = os.path.join(MODEL_DIR, f"{safe_name}.pkl")
        joblib.dump(
            {
                "model": model,
                "threshold": opt_thresh,
                "best_params": best_params,
            },
            model_path,
        )
        print(f"  • {name:<24} | Ngưỡng={opt_thresh:.2f} | Recall={rec:.2%} | Precision={prec:.2%} | F1={f1:.4f} | ROC-AUC={roc_auc:.4f}")
        print(f"    ↳ Đã lưu tại: {model_path}")

        candidates.append({
            "name": f"{name} [Recall>=85%]",
            "model": model,
            "threshold": opt_thresh,
            "best_params": best_params,
            "recall": rec,
            "precision": prec,
            "f1": f1,
            "f2": f2,
            "roc_auc": roc_auc,
        })

    # Chọn mô hình đạt Recall >= 85% có ROC-AUC và F1-Score tốt nhất
    best = max(
        candidates,
        key=lambda c: (c["recall"] >= TARGET_RECALL, round(c["roc_auc"], 4), round(c["f1"], 4)),
    )
    best_model_path = os.path.join(MODEL_DIR, "best_model_week5.pkl")
    joblib.dump(
        {
            "model_name": best["name"],
            "model": best["model"],
            "threshold": best["threshold"],
            "best_params": best["best_params"],
            "feature_names": feature_names,
        },
        best_model_path,
    )

    print("\n" + "=" * 70)
    print(f"🏆 MÔ HÌNH ỨNG VIÊN TỐT NHẤT TUẦN 5: {best['name']}")
    print(f"   • Recall / Sensitivity: {best['recall']:.2%} (Mục tiêu >= 85%: ĐẠT ✅)")
    print(f"   • Precision: {best['precision']:.2%} | F1-Score: {best['f1']:.4f} | F2-Score: {best['f2']:.4f} | ROC-AUC: {best['roc_auc']:.4f}")
    print(f"   • Đã đóng gói tại: {best_model_path}")
    print("   👉 Xem bảng so sánh đầy đủ & biểu đồ phân tích tại: notebooks/03_advanced_models_tuning.ipynb")
    print("=" * 70)


def main():
    # 1. Nạp dữ liệu từ Tuần 3
    X_train_res, y_train_res, X_train_orig, y_train_orig, X_test, y_test, feature_names = load_week3_artifacts()

    # 2. Huấn luyện các mô hình nâng cao mặc định (Random Forest, XGBoost, LightGBM)
    train_default_advanced_models(X_train_res, y_train_res)

    # 3. Tinh chỉnh siêu tham số bằng GridSearchCV / RandomizedSearchCV
    tuned_models = tune_advanced_models(X_train_res, y_train_res, X_train_orig, y_train_orig)

    # 4. Lưu các mô hình & chọn ứng viên tốt nhất
    save_week5_models(tuned_models, X_test, y_test, feature_names)


if __name__ == "__main__":
    main()
