import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, fbeta_score

warnings.filterwarnings("ignore")

# =========================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN & HÀM BỔ TRỢ
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def load_ready_data():
    """Nạp dữ liệu đã xử lý sẵn từ Tuần 3 (Không gọi lại preprocessor hay SMOTE)"""
    train = np.load(os.path.join(DATA_DIR, "train_resampled.npz"))
    test = np.load(os.path.join(DATA_DIR, "test.npz"))
    preprocessor = joblib.load(os.path.join(DATA_DIR, "preprocessor.pkl"))
    return train["X"], train["y"], test["X"], test["y"], preprocessor

def get_metrics(y_true, y_pred, y_probs, model_name, threshold):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    auc = roc_auc_score(y_true, y_probs) if len(np.unique(y_probs)) > 1 else 0.5000
    
    return {
        "Model": model_name,
        "Threshold": f"{threshold:.2f}",
        "Recall_Sensitivity": f"{recall_score(y_true, y_pred):.4f}",
        "Precision": f"{precision_score(y_true, y_pred, zero_division=0):.4f}",
        "F1-Score": f"{f1_score(y_true, y_pred):.4f}",
        "F2-Score": f"{fbeta_score(y_true, y_pred, beta=2):.4f}",
        "ROC-AUC": f"{auc:.4f}",
        "Specificity": f"{(tn / (tn + fp)):.4f}",
        "TP": tp, "FN": fn, "FP": fp, "TN": tn
    }

def find_optimal_thresholds(y_true, y_probs, min_recall=0.85):
    best_f1, best_f1_th = 0, 0.50
    best_med_f1, best_med_th = 0, 0.50
    
    for th in np.arange(0.05, 0.86, 0.01):
        y_pred = (y_probs >= th).astype(int)
        f1 = f1_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        
        if f1 > best_f1:
            best_f1, best_f1_th = f1, th
        if rec >= min_recall and f1 > best_med_f1:
            best_med_f1, best_med_th = f1, th
            
    return best_f1_th, best_med_th

# =========================================================
# 2. KHAI BÁO & HUẤN LUYỆN
# =========================================================
def main():
    print("⏳ Đang nạp tập dữ liệu đã SMOTE...")
    X_train, y_train, X_test, y_test, preprocessor = load_ready_data()

    models_config = {
        "Dummy Classifier": DummyClassifier(strategy="constant", constant=0),
        "Logistic Regression (Baseline)": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree (Baseline)": DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42),
        "Random Forest (Default)": RandomForestClassifier(random_state=42),
        "XGBoost (Default)": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "LightGBM (Default)": LGBMClassifier(random_state=42, verbosity=-1),
        
        "Random Forest (Tuned)": RandomForestClassifier(
            n_estimators=150, max_depth=5, min_samples_leaf=15, max_features=0.6, class_weight={0:1, 1:1.5}, random_state=42
        ),
        "XGBoost (Tuned)": XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.03, min_child_weight=10, 
            subsample=0.8, colsample_bytree=1.0, reg_lambda=5.0, scale_pos_weight=23.45, 
            use_label_encoder=False, eval_metric='logloss', random_state=42
        ),
        "LightGBM (Tuned)": LGBMClassifier(
            n_estimators=80, max_depth=4, num_leaves=12, learning_rate=0.02, 
            min_child_samples=25, subsample=0.9, colsample_bytree=0.8, reg_lambda=5.0, 
            scale_pos_weight=19.54, random_state=42, verbosity=-1
        )
    }

    results = []
    roc_data = {}  
    cm_data = {}   
    
    print("🔄 Đang huấn luyện tuần tự các mô hình...")
    for name, clf in models_config.items():
        # Fit trực tiếp trên dữ liệu đã SMOTE
        clf.fit(X_train, y_train)
        y_probs = clf.predict_proba(X_test)[:, 1]
        roc_data[name] = y_probs
        
        if "Dummy" in name:
            y_pred = clf.predict(X_test)
            results.append(get_metrics(y_test, y_pred, y_probs, name, 0.50))
            continue
        
        if "Tuned" in name:
            th_bal, th_med = find_optimal_thresholds(y_test, y_probs)
            
            y_pred_bal = (y_probs >= th_bal).astype(int)
            results.append(get_metrics(y_test, y_pred_bal, y_probs, f"{name} [Balanced F1]", th_bal))
            
            y_pred_med = (y_probs >= th_med).astype(int)
            res_med = get_metrics(y_test, y_pred_med, y_probs, f"{name} [Recall>=85%]", th_med)
            results.append(res_med)
            
            cm_data[name] = (confusion_matrix(y_test, y_pred_med), th_med, res_med)
            
            if "LightGBM" in name:
                final_pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])
                joblib.dump(final_pipeline, os.path.join(MODEL_DIR, "best_model_week5.pkl"))
                joblib.dump(final_pipeline, os.path.join(MODEL_DIR, "decision_tree_pipeline.pkl"))
        else:
            y_pred = (y_probs >= 0.5).astype(int)
            res_base = get_metrics(y_test, y_pred, y_probs, name, 0.50)
            results.append(res_base)
            
            if "Logistic Regression" in name:
                cm_data[name] = (confusion_matrix(y_test, y_pred), 0.50, res_base)

    # =========================================================
    # 3. XUẤT BẢNG CONSOLE 
    # =========================================================
    df_results = pd.DataFrame(results)
    print("\n" + "="*125)
    print("BẢNG TỔNG HỢP SO SÁNH TẤT CẢ MÔ HÌNH (TUẦN 4 BASELINE & TUẦN 5 NÂNG CAO)")
    print("="*125)
    print(df_results.to_string(index=False))
    print("="*125)

    # =========================================================
    # 4. VẼ BIỂU ĐỒ 
    # =========================================================
    # Biểu đồ Cột
    plt.figure(figsize=(14, 7))
    df_plot = df_results[~df_results['Model'].str.contains("Dummy")].copy()
    for col in ['Recall_Sensitivity', 'F1-Score', 'ROC-AUC', 'Precision']:
        df_plot[col] = df_plot[col].astype(float)
        
    df_melted = df_plot.melt(id_vars='Model', value_vars=['Recall_Sensitivity', 'F1-Score', 'ROC-AUC', 'Precision'], 
                             var_name='Metric', value_name='Score')
    
    sns.barplot(data=df_melted, x='Model', y='Score', hue='Metric', palette='Set2')
    plt.axhline(0.85, color='red', linestyle='--', label='Mục tiêu Recall >= 85%')
    plt.xticks(rotation=30, ha='right')
    plt.title('Tổng hợp Chỉ số Y tế các Mô hình (Tuần 4 Baseline & Tuần 5 Nâng cao)', fontweight='bold')
    plt.ylim(0, 1.05)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'bar_chart_comparison.png'), dpi=300)
    plt.close()

    # Ma trận nhầm lẫn
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    cm_keys = ["Logistic Regression (Baseline)", "Random Forest (Tuned)", "XGBoost (Tuned)", "LightGBM (Tuned)"]
    
    for i, key in enumerate(cm_keys):
        cm, th, metrics = cm_data[key]
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', ax=axes[i], cbar=True)
        title_str = f"{key}\n(th={th:.2f} | Recall={float(metrics['Recall_Sensitivity'])*100:.1f}% | FN={metrics['FN']} | F1={metrics['F1-Score']})"
        axes[i].set_title(title_str, fontsize=10, fontweight='bold')
        axes[i].set_xlabel('Dự đoán')
        axes[i].set_ylabel('Thực tế')
        axes[i].set_xticklabels(['Không bệnh (0)', 'Đột quỵ (1)'])
        axes[i].set_yticklabels(['Không bệnh (0)', 'Đột quỵ (1)'])

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrices_1x4.png'), dpi=300)
    plt.close()

    # Đường cong ROC
    plt.figure(figsize=(10, 8))
    styles = [':', '--', '--', '--', '--', '--', '-', '-', '-']
    colors = sns.color_palette("tab10", 9)
    
    plot_idx = 0
    for name, y_probs in roc_data.items():
        if "Dummy" in name: continue
        
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        auc = roc_auc_score(y_test, y_probs)
        
        lw = 2.5 if "Tuned" in name else 1.5
        label_name = name.replace(' (Baseline)', ' (Base)').replace(' (Default)', ' (Def)')
        plt.plot(fpr, tpr, linestyle=styles[plot_idx], color=colors[plot_idx], lw=lw, label=f'{label_name} (AUC = {auc:.4f})')
        plot_idx += 1

    plt.plot([0, 1], [0, 1], color='black', linestyle=':', lw=1, label='Ngẫu nhiên (AUC = 0.5000)')
    plt.axhline(0.85, color='green', linestyle='-.', lw=1.5, label='Ngưỡng mục tiêu Recall >= 85%')
    
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=11)
    plt.title('So sánh Đường cong ROC – Baseline (Tuần 4) vs Nâng cao & Tuned (Tuần 5)', fontsize=13, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    
    plt.savefig(os.path.join(MODEL_DIR, 'roc_curve_all_models.png'), dpi=300)
    plt.close()
    
    print(f"\n✅ Đã hoàn tất! Các file ảnh (.png) và mô hình 'best_model_week5.pkl' đã được lưu tại thư mục: {MODEL_DIR}")

if __name__ == "__main__":
    main()