
import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

RANDOM_STATE  = 42
TARGET_RECALL = 0.85  # Ngưỡng mục tiêu y tế: Recall >= 85% - 90%

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR     = os.path.join(BASE_DIR, "outputs", "models")
FIG_DIR       = os.path.join(BASE_DIR, "outputs", "figures")
BASELINE_CSV  = os.path.join(BASE_DIR, "outputs", "baseline_comparison.csv")
FEATURE_CSV   = os.path.join(BASE_DIR, "outputs", "dt_feature_importance.csv")


# ──────────────────────────────────────────────────────────────
# 1. Nạp dữ liệu từ Tuần 3
# ──────────────────────────────────────────────────────────────
def load_week3_artifacts():
    """Nạp tập train (SMOTE), tập train gốc, tập test và tên đặc trưng từ Tuần 3."""
    train_res  = np.load(os.path.join(DATA_DIR, "train_resampled.npz"))
    train_orig = np.load(os.path.join(DATA_DIR, "train_original.npz"))
    test       = np.load(os.path.join(DATA_DIR, "test.npz"))
    feature_names = joblib.load(os.path.join(DATA_DIR, "feature_names.pkl"))

    X_train_res, y_train_res   = train_res["X"], train_res["y"]
    X_train_orig, y_train_orig = train_orig["X"], train_orig["y"]
    X_test, y_test             = test["X"], test["y"]

    print("=" * 70)
    print("📂 1. NẠP DỮ LIỆU ĐÃ TIỀN XỬ LÝ TỪ TUẦN 3")
    print("=" * 70)
    print(f"  • Tập Train (sau SMOTE): {X_train_res.shape[0]:,} mẫu × {X_train_res.shape[1]} features "
          f"(Không bệnh: {int((y_train_res == 0).sum()):,} | Đột quỵ: {int((y_train_res == 1).sum()):,})")
    print(f"  • Tập Train (gốc):       {X_train_orig.shape[0]:,} mẫu × {X_train_orig.shape[1]} features "
          f"(Không bệnh: {int((y_train_orig == 0).sum()):,} | Đột quỵ: {int((y_train_orig == 1).sum()):,})")
    print(f"  • Tập Test (thực tế):    {X_test.shape[0]:,} mẫu × {X_test.shape[1]} features "
          f"(Không bệnh: {int((y_test == 0).sum()):,} | Đột quỵ: {int((y_test == 1).sum()):,})")
    print()
    return X_train_res, y_train_res, X_train_orig, y_train_orig, X_test, y_test, feature_names


# ──────────────────────────────────────────────────────────────
# 2. Huấn luyện các mô hình Baseline
# ────────
# ──────────────────────────────────────────────────────
def train_logistic_regression(X_train, y_train):
    """Huấn luyện mô hình Baseline 1: Logistic Regression."""
    lr = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        C=1.0,
    )
    lr.fit(X_train, y_train)
    print("✅ Đã huấn luyện xong Logistic Regression (Baseline).")
    return lr


def train_decision_tree(X_train, y_train):
    """Huấn luyện mô hình Baseline 2: Decision Tree (giới hạn độ sâu tránh overfit)."""
    dt = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    dt.fit(X_train, y_train)
    print("✅ Đã huấn luyện xong Decision Tree (Baseline).")
    return dt


def train_baselines(X_train, y_train):
    """Huấn luyện bộ mô hình Baseline."""
    print("=" * 70)
    print("🤖 2. HUẤN LUYỆN MÔ HÌNH BASELINE (TRÊN TẬP TRAIN ĐÃ SMOTE)")
    print("=" * 70)
    lr = train_logistic_regression(X_train, y_train)
    dt = train_decision_tree(X_train, y_train)
    print()
    return {
        "Logistic Regression": lr,
        "Decision Tree": dt,
    }


# ──────────────────────────────────────────────────────────────
# 3. Bộ khung đánh giá Y tế (KHÔNG dùng Accuracy đơn thuần)
# ──────────────────────────────────────────────────────────────
def evaluate_medical_metrics(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    threshold: float = 0.5,
    X_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
    train_data_label: str = "SMOTE",
    note: str = "",
) -> dict:
    """
    Đo các chỉ số đánh giá y tế trên tập Test:
      - Recall / Sensitivity (Độ nhạy): TP / (TP + FN) -> Ưu tiên tối đa để không bỏ sót ca bệnh
      - Specificity (Độ đặc hiệu):      TN / (TN + FP)
      - Precision (Giá trị dự báo dương): TP / (TP + FP)
      - F1-Score:                       2 * Precision * Recall / (Precision + Recall)
      - ROC-AUC & PR-AUC:               Đánh giá khả năng phân loại trên mọi ngưỡng
      - Confusion Matrix:               TN, FP, FN (ca bệnh bị bỏ sót!), TP (ca bệnh phát hiện đúng)
    """
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_prob = y_pred.astype(float)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    recall_sens = recall_score(y_test, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision   = precision_score(y_test, y_pred, zero_division=0)
    f1          = f1_score(y_test, y_pred, zero_division=0)
    roc_auc     = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
    pr_auc      = average_precision_score(y_test, y_prob)

    recall_train = None
    if X_train is not None and y_train is not None:
        if hasattr(model, "predict_proba"):
            y_train_pred = (model.predict_proba(X_train)[:, 1] >= threshold).astype(int)
        else:
            y_train_pred = model.predict(X_train)
        recall_train = round(float(recall_score(y_train, y_train_pred, zero_division=0)), 4)

    status = "ĐẠT (>=85%)" if recall_sens >= TARGET_RECALL else "CHƯA ĐẠT (<85%)"

    print(f"──────────────────────────────────────────────────────────────────────")
    print(f"🩺 ĐÁNH GIÁ Y TẾ: {model_name} (Dữ liệu huấn luyện: {train_data_label} | Ngưỡng: {threshold:.2f})")
    print(f"──────────────────────────────────────────────────────────────────────")
    print(f"  ★ Recall / Sensitivity (Độ nhạy): {recall_sens:.4f} ({recall_sens:.2%})  --> [{status}]")
    print(f"  • Precision (Độ chuẩn xác):       {precision:.4f} ({precision:.2%})")
    print(f"  • F1-Score:                       {f1:.4f}")
    print(f"  • ROC-AUC:                        {roc_auc:.4f}")
    print(f"  • PR-AUC:                         {pr_auc:.4f}")
    print(f"  • Specificity (Độ đặc hiệu):      {specificity:.4f} ({specificity:.2%})")
    print(f"  • Confusion Matrix (Tập Test = {len(y_test)} ca, trong đó {tp + fn} ca đột quỵ thật):")
    print(f"      + Phát hiện đúng ca đột quỵ (TP): {tp:4d} / {tp + fn} ca")
    print(f"      + BỎ SÓT ca đột quỵ (FN - Nguy hiểm!): {fn:4d} / {tp + fn} ca")
    print(f"      + Cảnh báo nhầm (FP):             {fp:4d} / {tn + fp} ca")
    print(f"      + Dự đoán đúng an toàn (TN):      {tn:4d} / {tn + fp} ca")
    print()
    print("  Chi tiết Classification Report (Lớp 1 - Đột quỵ là trọng tâm):")
    print(classification_report(y_test, y_pred, target_names=["Không đột quỵ (0)", "Đột quỵ (1)"], zero_division=0))

    return {
        "Model": model_name,
        "Train_Data": train_data_label,
        "Threshold": round(float(threshold), 2),
        "Recall_Sensitivity": round(float(recall_sens), 4),
        "Precision": round(float(precision), 4),
        "F1-Score": round(float(f1), 4),
        "ROC-AUC": round(float(roc_auc), 4),
        "PR-AUC": round(float(pr_auc), 4),
        "Specificity": round(float(specificity), 4),
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "Recall_Train": recall_train,
        "Note": note,
    }


def demonstrate_accuracy_paradox(X_train_orig, y_train_orig, X_test, y_test):
    """Minh họa vì sao KHÔNG dùng Accuracy đơn thuần trên dữ liệu đột quỵ lệch (95% - 5%)."""
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train_orig, y_train_orig)
    y_pred = dummy.predict(X_test)
    naive_acc = (y_pred == y_test).mean()
    rec = recall_score(y_test, y_pred, zero_division=0)
    print("⚠️ MINH HỌA BẪY ĐỘ CHÍNH XÁC (ACCURACY PARADOX):")
    print(f"   Mô hình Dummy (luôn phán 'Không đột quỵ' cho mọi bệnh nhân):")
    print(f"   → Accuracy đạt tới {naive_acc:.2%} nhưng Recall (Độ nhạy) = {rec:.2%} (Bỏ sót 50/50 ca đột quỵ!).")
    print(f"   ⇒ Do đó, dự án loại bỏ Accuracy đơn thuần và lấy Recall/Sensitivity, Precision, F1-Score, ROC-AUC làm thước đo chuẩn.\n")


# ──────────────────────────────────────────────────────────────
# 4. Vẽ biểu đồ đánh giá Baseline
# ──────────────────────────────────────────────────────────────
def plot_baseline_charts(models: dict, X_test: np.ndarray, y_test: np.ndarray, feature_names: list):
    """Vẽ Confusion Matrix, ROC Curve và Feature Importance của Decision Tree."""
    os.makedirs(FIG_DIR, exist_ok=True)

    # 4.1 Confusion Matrix
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Không đột quỵ (0)", "Đột quỵ (1)"],
            yticklabels=["Không đột quỵ (0)", "Đột quỵ (1)"],
            ax=ax,
        )
        rec = recall_score(y_test, y_pred)
        ax.set_title(f"{name}\nRecall (Độ nhạy) = {rec:.1%} | FN (Bỏ sót) = {cm[1,0]}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Thực tế")
        ax.set_xlabel("Dự đoán")

    plt.tight_layout()
    cm_path1 = os.path.join(FIG_DIR, "confusion_matrices_baseline.png")
    cm_path2 = os.path.join(FIG_DIR, "week4_confusion_matrix.png")
    plt.savefig(cm_path1, dpi=150, bbox_inches="tight")
    plt.savefig(cm_path2, dpi=150, bbox_inches="tight")
    plt.close()

    # 4.2 ROC Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2980b9", "#e74c3c", "#27ae60"]
    for (name, model), color in zip(models.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=color, lw=2.2, label=f"{name} (ROC-AUC = {auc_val:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Ngẫu nhiên (AUC = 0.5000)")
    ax.axhline(TARGET_RECALL, color="#2ecc71", linestyle=":", lw=1.5, label=f"Mục tiêu Recall = {TARGET_RECALL:.0%}")
    ax.set_xlabel("1 - Specificity (False Positive Rate)")
    ax.set_ylabel("Sensitivity / Recall (True Positive Rate)")
    ax.set_title("Đường cong ROC – Mô hình Baseline (Tuần 4)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    roc_path1 = os.path.join(FIG_DIR, "roc_curve_baseline.png")
    roc_path2 = os.path.join(FIG_DIR, "week4_roc_curve.png")
    plt.savefig(roc_path1, dpi=150, bbox_inches="tight")
    plt.savefig(roc_path2, dpi=150, bbox_inches="tight")
    plt.close()

    # 4.3 Feature Importance từ Decision Tree (hỗ trợ Data Engineer Vũ Thị Giang)
    if "Decision Tree" in models:
        dt_model = models["Decision Tree"]
        fi_df = pd.DataFrame({
            "feature": feature_names,
            "importance": dt_model.feature_importances_,
        }).sort_values("importance", ascending=False)
        fi_df.to_csv(FEATURE_CSV, index=False)

        fig, ax = plt.subplots(figsize=(9, 6))
        sns.barplot(data=fi_df.head(12), x="importance", y="feature", palette="viridis", ax=ax)
        ax.set_title("Top Feature Importance – Decision Tree Baseline", fontsize=13, fontweight="bold")
        ax.set_xlabel("Mức độ quan trọng (Gini Importance)")
        ax.set_ylabel("Đặc trưng")
        plt.tight_layout()
        fi_path1 = os.path.join(FIG_DIR, "feature_importance_dt.png")
        fi_path2 = os.path.join(FIG_DIR, "week4_dt_feature_importance.png")
        plt.savefig(fi_path1, dpi=150, bbox_inches="tight")
        plt.savefig(fi_path2, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"📈 Đã lưu các biểu đồ đánh giá Tuần 4 vào: {FIG_DIR}")


# ──────────────────────────────────────────────────────────────
# 5. Lưu mô hình & Bảng tổng hợp chỉ số y tế
# ──────────────────────────────────────────────────────────────
def save_models_and_metrics(models: dict, results: list[dict]):
    """Lưu file .pkl của 2 mô hình Baseline và bảng CSV chỉ số y tế."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, model in models.items():
        safe_name = name.lower().replace(" ", "_")
        path1 = os.path.join(MODEL_DIR, f"{safe_name}_baseline.pkl")
        path2 = os.path.join(MODEL_DIR, f"baseline_{safe_name}.pkl")
        joblib.dump(model, path1)
        joblib.dump(model, path2)
        print(f"💾 Đã lưu mô hình {name} → {path1}")

    df_res = pd.DataFrame(results)
    df_res.to_csv(BASELINE_CSV, index=False)
    print(f"\n📋 Bảng tổng hợp các chỉ số y tế Baseline đã lưu → {BASELINE_CSV}")
    display_cols = [
        "Model", "Train_Data", "Recall_Sensitivity", "Precision",
        "F1-Score", "ROC-AUC", "Specificity", "TP", "FN", "FP", "TN"
    ]
    print(df_res[display_cols].to_string(index=False))


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    # 1. Nạp dữ liệu từ Tuần 3
    X_train_res, y_train_res, X_train_orig, y_train_orig, X_test, y_test, feature_names = load_week3_artifacts()

    # 2. Minh họa vì sao không dùng Accuracy đơn thuần
    demonstrate_accuracy_paradox(X_train_orig, y_train_orig, X_test, y_test)

    # 3. Huấn luyện 2 mô hình Baseline trên tập train đã SMOTE
    models = train_baselines(X_train_res, y_train_res)

    # 4. Đánh giá bằng bộ chỉ số y tế (Recall/Sensitivity, Precision, F1-Score, ROC-AUC)
    print("=" * 70)
    print("📊 3. KẾT QUẢ ĐO LƯỜNG CÁC CHỈ SỐ Y TẾ TRÊN TẬP TEST")
    print("=" * 70)
    results = []
    for name, model in models.items():
        metrics = evaluate_medical_metrics(
            model, X_test, y_test,
            model_name=name,
            threshold=0.5,
            X_train=X_train_res,
            y_train=y_train_res,
            train_data_label="SMOTE",
            note="Baseline model (Tuần 4)",
        )
        results.append(metrics)

    # 5. Vẽ biểu đồ & Lưu kết quả
    plot_baseline_charts(models, X_test, y_test, feature_names)
    save_models_and_metrics(models, results)

    # 6. Kết luận Baseline tốt nhất
    best_baseline = max(results, key=lambda r: (r["Recall_Sensitivity"], r["ROC-AUC"]))
    print("\n" + "=" * 70)
    print("🏆 KẾT LUẬN BASELINE TUẦN 4")
    print("=" * 70)
    print(f"  • Baseline tốt nhất: {best_baseline['Model']}")
    print(f"    - Recall / Sensitivity (Độ nhạy): {best_baseline['Recall_Sensitivity']:.2%} "
          f"(Phát hiện {best_baseline['TP']}/50 ca, bỏ sót {best_baseline['FN']} ca)")
    print(f"    - Precision: {best_baseline['Precision']:.2%} | F1-Score: {best_baseline['F1-Score']:.4f} | ROC-AUC: {best_baseline['ROC-AUC']:.4f}")
    print(f"  • Đánh giá mục tiêu y tế (Recall >= 85%): Hiện đạt {best_baseline['Recall_Sensitivity']:.2%}, "
          f"sẽ tiếp tục nâng cấp bằng Random Forest, XGBoost, LightGBM & Tuning ở Tuần 5.")
    print("=" * 70)


if __name__ == "__main__":
    main()
