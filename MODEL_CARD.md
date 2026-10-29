# Mô hình sàng lọc chính StrokeCare

## Mục đích

Hỗ trợ trình diễn đề tài sinh viên và nghiên cứu sàng lọc nguy cơ từ 10 thuộc tính của Stroke Prediction Dataset. Không xác nhận bệnh, không dự báo nguy cơ tuyệt đối theo thời hạn, chưa kiểm định lâm sàng tại Việt Nam.

## Artifact và tái lập

- Chính: outputs/models/model_pipeline.pkl; alias best_model_week5.pkl.
- Bản chạy nghiên cứu: outputs/final/model_pipeline.pkl; kết quả máy đọc validation_metrics.json; biểu đồ evaluation.png; giải thích SHAP tổng thể global_shap.png.
- Python 3.13, các thư viện và phiên bản theo requirements.txt; random seed 42; train/holdout stratified 80/20; 5 fold stratified shuffle trên train.
- Loại id; giữ 10 cột nguyên bản; median imputation + StandardScaler cho biến số, OneHotEncoder cho biến phân loại. Không học từ tên/email/khu vực tài khoản.
- RandomizedSearchCV so sánh trọng số lớp và SMOTE trong fold. Chọn XGBoost; n_estimators=180, max_depth=2, learning_rate=0.02, min_child_weight=15, reg_lambda=20, subsample=1, scale_pos_weight=15. Cấu hình thắng không SMOTE.
- Ngưỡng chính 0,1640415788 từ OOF train với chính sách ưu tiên độ nhạy >=97,5% trên train. Chính sách được tinh chỉnh qua quá trình phát triển; không coi tập kiểm tra phát triển là tập kiểm định mới.

## Kết quả đo

| Độ đo trên tập kiểm tra phát triển | Giá trị |
| --- | --- |
| Mẫu / ca dương tính | 1.022 / 50 |
| ROC-AUC | 0,850391 |
| PR-AUC (average precision) | 0,274849 |
| Recall | 90,00% |
| Precision | 8,82% |
| F1 | 0,160714 |
| F2 | 0,316901 |
| TN / FP / FN / TP | 507 / 465 / 5 / 45 |

So với baseline Logistic Regression, mô hình có AUC tốt hơn trên tập phát triển. Các chỉ số không thể hiện hiệu quả điều trị hoặc lợi ích lâm sàng. Chỉ có 50 ca dương tính nên một ca tương ứng 2 điểm phần trăm Recall; cần tập dữ liệu ngoài độc lập để đánh giá độ tin cậy.

## Sử dụng và giải thích

Joblib load artifact; gọi predict_proba với DataFrame một dòng chứa 10 thuộc tính. Ngưỡng lấy từ thuộc tính decision_threshold_; không dùng pipeline.predict vì mặc định classifier có thể dùng 0,5. Backend áp dụng ngưỡng lưu cùng mô hình và trả SHAP cho lớp dương tính. TreeExplainer với XGBoost trả đóng góp trên thang log-odds, không phải phần trăm. Điểm predict_proba chưa được calibration để diễn giải như nguy cơ mắc bệnh thật.

Bác sĩ có thể thay ngưỡng trong mỗi request; ngưỡng thực dùng và nhãn được lưu vào lịch sử. Các chỉ số trong bảng áp dụng cho ngưỡng mặc định, không áp dụng cho mọi ngưỡng tùy chỉnh.

## Giới hạn

Tập holdout đã được dùng trong các vòng phát triển trước và được quan sát khi chọn chính sách độ nhạy. Dù tham số và ngưỡng số được tính từ train OOF, kết quả cuối không phải ước lượng hoàn toàn độc lập. Không khẳng định đây là mô hình “hoàn hảo”. Do mất cân bằng lớp, nhiều dương tính giả; dữ liệu công khai khác với thực tế bệnh viện Việt Nam. Chưa đánh giá công bằng theo nhóm dân số hoặc triển khai lâm sàng.
