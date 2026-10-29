# Các điểm cần sửa trong báo cáo final

Ngày đối chiếu: 30/10/2026.

**Báo cáo:** `Report/BC_TTCSN_FINAL_NHOM_5.docx`.

Tài liệu này chỉ hướng dẫn bạn tự sửa nội dung trong Word. Không thay đổi báo cáo gốc, font, lề, style, mục lục hay mẫu của nhà trường. Định vị theo tên mục, bảng/hình và cụm từ tìm kiếm; không dùng số trang vì sẽ thay đổi khi bạn sửa.

Đối chiếu trực tiếp với mã nguồn, artifact `outputs/models/model_pipeline.pkl`, kết quả `outputs/final/validation_metrics.json`, các sơ đồ/ảnh nhúng trong báo cáo và bộ kiểm thử hiện tại. Không chỉ dựa vào README hoặc mô tả ở lần trao đổi trước.

## 1. Các giá trị cần thống nhất xuyên suốt

| Nội dung | Phiên bản hiện tại |
| --- | --- |
| Mô hình chính | Một pipeline XGBoost |
| Artifact chính | `outputs/models/model_pipeline.pkl` |
| Alias tương thích | `outputs/models/best_model_week5.pkl`, cùng mô hình mới |
| Ngưỡng mặc định | 0,16404157876968384, khoảng 16,40% điểm đầu ra |
| Phân loại | Nhãn 1 khi điểm >= ngưỡng áp dụng, nhãn 0 khi thấp hơn |
| Train / kiểm tra phát triển | 4.088 / 1.022 mẫu; 199 / 50 mẫu dương tính |
| Tìm siêu tham số | RandomizedSearchCV, StratifiedKFold 5-fold, scoring ROC-AUC |
| Chọn ngưỡng | Tối đa F2 trên OOF train trong các ngưỡng có Recall train >=97,5% |
| Chọn mô hình phục vụ | Precision OOF cao nhất trong RF/XGBoost/LightGBM tại ngưỡng đã chọn; hòa thì so ROC-AUC OOF |
| Imbalance của cấu hình thắng | scale_pos_weight=15, không SMOTE |
| Recall / Precision | 90,00% / 8,82% trên tập kiểm tra phát triển |
| ROC-AUC / Average Precision | 0,850391 / 0,274849 |
| F1 / F2 | 0,160714 / 0,316901 |
| Ma trận [[TN, FP], [FN, TP]] | [[507,465],[5,45]] |
| Specificity | 52,16% |
| Lưu lịch sử chính | history: id, user_id, action, timestamp |
| Biểu đồ lịch sử mới | JavaScript thuần và HTML Canvas |

**Không chỉ thay tên LightGBM thành XGBoost rồi giữ số cũ.** AUC sau làm tròn trùng 0,8504 nhưng thuộc hai lần thực nghiệm khác nhau.

## 2. Các điểm bắt buộc sửa về mô hình và thực nghiệm

### 01 — Tên mô hình triển khai

**Vị trí:** 2.2.4, 3.2.1, Kết luận và bảng tệp mô hình ở phụ lục.

**Đang ghi:** LightGBM Tuned là mô hình tối ưu; hoặc “LightGBM Tuned / XGBoost Tuned kèm Random Forest Pipeline”.

**Thay bằng:**

> Phiên bản hiện tại triển khai một pipeline XGBoost duy nhất làm mô hình suy luận chính. Logistic Regression là baseline; Random Forest và LightGBM là các mô hình đối chứng. File chính là outputs/models/model_pipeline.pkl; best_model_week5.pkl là tên tương thích chứa cùng pipeline mới.

Không gọi đây là Voting/Stacking hay hệ thống ghép dự đoán của ba pipeline. XGBoost là boosting nhưng ứng dụng chỉ nạp một pipeline phục vụ.

### 02 — Quy trình encoding, scaling và pipeline

**Vị trí:** 2.1.3.5; đối chiếu 2.1.3.2 và 2.2.1.

**Đang ghi:** One-Hot ba biến, Label Encoding ever_married/Residence_type; danh sách tổng thể thiếu StandardScaler.

**Thay danh sách quy trình bằng:**

1. Đọc CSV gốc; loại id, bỏ bản ghi trùng nếu có và tách nhãn stroke.
2. Chia dữ liệu phân tầng train/kiểm tra phát triển 80/20, random_state=42.
3. Trong mỗi fold train: điền trung vị rồi StandardScaler cho age, avg_glucose_level, bmi.
4. Điền giá trị phổ biến nhất rồi OneHotEncoder(handle_unknown='ignore', sparse_output=False) cho gender, ever_married, work_type, Residence_type, smoking_status.
5. Giữ hypertension và heart_disease ở dạng 0/1.
6. Khảo sát passthrough và SMOTE trong fold train, cùng các cấu hình trọng số lớp.
7. Tìm tham số, chọn ngưỡng từ train; đánh giá trên tập kiểm tra phát triển.
8. Lưu pipeline đã fit, ngưỡng và metadata.

Nếu nói pipeline có sampler, dùng tên `imblearn.pipeline.Pipeline`, không chỉ `sklearn.pipeline.Pipeline`. StandardScaler chuẩn hóa trung bình/độ lệch chuẩn, **không bảo đảm biến thành phân phối Gaussian** như cách nói “chuẩn hóa về phân phối chuẩn”.

### 03 — SMOTE không còn là bước bắt buộc của mô hình chính

**Vị trí:** 2.1.3.4, 2.2.1 bước 3, Hình 2.6 và các câu nói toàn bộ mô hình được train trên 7.778 mẫu.

**Thay bằng:**

> Nhóm khảo sát cấu hình không lấy mẫu lại và SMOTE(random_state=42, sampling_strategy=0.5) bên trong các fold train, đồng thời khảo sát trọng số lớp. sampling_strategy=0.5 là tỷ lệ số mẫu lớp thiểu số so với lớp đa số sau lấy mẫu lại, không phải hai lớp chiếm 50:50. XGBoost cuối không dùng SMOTE và sử dụng scale_pos_weight=15. Pipeline phục vụ được fit trên 4.088 mẫu train gốc.

`train_resampled.npz` gồm 7.778 mẫu là artifact tuần 3 cũ; `train_validated.py` không nạp nó. Nếu giữ Hình 2.6, ghi rõ là minh họa/thử nghiệm giai đoạn trước, không phải cấu hình thắng cuối.

### 04 — Nguồn dữ liệu và script huấn luyện

**Vị trí:** 2.2.1 và phụ lục tệp dữ liệu.

**Đang ghi:** mô hình cuối đọc stroke_clean.csv/npz và huấn luyện ở src/week5.py.

**Thay bằng:**

> Quy trình cuối ở src/train_validated.py đọc trực tiếp data/raw/healthcare-dataset-stroke-data.csv, loại id và bản ghi trùng trước khi chia dữ liệu. Các bộ biến đổi được fit trong fold train. src/week5_full.py là điểm gọi tương thích sang quy trình này. src/week4.py và các npz thuộc giai đoạn thực nghiệm trước.

Các số 5.110 / 4.088 / 1.022 và 249 / 199 / 50 mẫu dương tính vẫn đúng với lần chạy cuối, không cần đổi.

### 05 — Thuật toán tìm kiếm và hàm chấm điểm

**Vị trí:** 2.2.3, nhận xét dưới Bảng 3.1 và Kết luận.

**Đang ghi:** GridSearchCV/RandomizedSearchCV tối ưu F2; RF dùng GridSearchCV.

**Thay bằng:**

> Nhóm dùng RandomizedSearchCV với StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='roc_auc'. Logistic Regression khảo sát 8 cấu hình, mỗi mô hình cây khảo sát 12 cấu hình. Sau khi chọn tham số, nhóm tính dự đoán ngoài fold (OOF) trên train và hiệu chỉnh ngưỡng theo mục tiêu độ nhạy.

F2 được dùng **ở bước chọn ngưỡng**, không phải scoring của RandomizedSearchCV. Nếu giữ giải thích F2, viết “ưu tiên Recall thông qua beta=2”; không diễn giải máy móc “Recall quan trọng chính xác gấp 2 lần Precision”.

### 06 — Cơ chế chọn ngưỡng và chọn mô hình

**Vị trí:** 2.2.3 và 2.2.4.

**Đang ghi:** find_optimal_medical_threshold(), quét 0,05–0,85 bước 0,01; có hai chế độ Balanced F1/Medical và bảo đảm cứng Recall test 85–90%.

**Thay bằng:**

> choose_threshold() sử dụng đường Precision–Recall từ điểm OOF train. Trong các ngưỡng đạt Recall OOF >=97,5%, chọn ngưỡng có F2 lớn nhất. Ngưỡng lấy từ điểm dự đoán, không quét bước cố định 0,01. Trong nhóm Random Forest, XGBoost và LightGBM, ưu tiên Precision OOF tại ngưỡng đó; nếu bằng nhau thì so ROC-AUC OOF. Ngưỡng XGBoost cuối là 0,1640415788. Recall 90% trên tập kiểm tra phát triển là kết quả quan sát, không bảo đảm cho dữ liệu mới.

Bỏ mô tả phiên bản cuối có sẵn hai profile Balanced F1/Medical. Bác sĩ được nhập ngưỡng tùy chọn; các chỉ số ở bảng kết quả không tự áp dụng cho mọi ngưỡng tùy chỉnh. Cũng bỏ câu “chọn ngưỡng để tối đa ROC-AUC”: ROC-AUC của cùng tập điểm không thay đổi chỉ vì đổi một ngưỡng phân loại.

### 07 — Không gian và tham số ở Bảng 2.7

**Thay phần không gian khảo sát bằng:**

| Mô hình | Tham số cố định | Không gian khảo sát |
| --- | --- | --- |
| Logistic Regression | max_iter=2000, seed=42 | C=[0.01,0.1,1,10]; class_weight=[None,balanced] |
| Random Forest | n_estimators=180, seed=42, n_jobs=1 | max_depth=[3,5,7,None]; min_samples_leaf=[5,15,30]; max_features=[0.6,1.0]; class_weight=[None,balanced] |
| XGBoost | n_estimators=180, eval_metric=logloss, seed=42, n_jobs=1 | max_depth=[2,3,4]; learning_rate=[0.02,0.05]; min_child_weight=[5,15]; reg_lambda=[5,20]; subsample=[0.8,1.0]; scale_pos_weight=[1,5,15] |
| LightGBM | n_estimators=140, seed=42, n_jobs=1, verbosity=-1 | max_depth=[3,4,5]; num_leaves=[7,12]; learning_rate=[0.02,0.05]; min_child_samples=[25,50]; reg_lambda=[5,20]; scale_pos_weight=[1,5,15] |

Mỗi họ còn khảo sát sampler giữa passthrough và SMOTE(sampling_strategy=0.5). “seed” ở bảng này là random_state.

**Tham số được chọn trong từng họ:**

| Mô hình | Bộ tham số được chọn, ngoài các tham số cố định trên | Ngưỡng |
| --- | --- | ---: |
| Logistic Regression | C=1, class_weight=None, không SMOTE | 0,01713381 |
| Random Forest | max_depth=3, min_samples_leaf=5, max_features=0.6, class_weight=None, không SMOTE | 0,01348052 |
| XGBoost | max_depth=2, learning_rate=0.02, min_child_weight=15, reg_lambda=20, subsample=1.0, scale_pos_weight=15, không SMOTE | 0,16404158 |
| LightGBM | max_depth=3, num_leaves=12, learning_rate=0.02, min_child_samples=50, reg_lambda=20, scale_pos_weight=15, không SMOTE | 0,09241258 |

Không giữ n_estimators=150, learning_rate=0.03, scale_pos_weight=23.45/19.54 hay ngưỡng 0,18/0,15 của bảng cũ cho bản triển khai mới.

### 08 — Tập kiểm tra không còn là kiểm định hoàn toàn độc lập

**Vị trí:** 2.2.1, 2.2.4, 3.2.1, caption Bảng 3.1/Hình 3.5, Kết luận.

**Tìm:** “tập kiểm tra độc lập”, “cô lập hoàn toàn”, “phản ánh trung thực hiệu năng ngoài thực tế lâm sàng”, “tránh rò rỉ tuyệt đối”.

**Thay mô tả bằng:**

> Nhóm giữ cách chia train/kiểm tra phát triển 80/20. Trong mỗi lần huấn luyện, tiền xử lý và lấy mẫu lại chỉ fit trên train hoặc fold train. Tuy nhiên, tập kiểm tra phát triển đã được quan sát qua nhiều vòng thực nghiệm và điều chỉnh chính sách độ nhạy. Vì vậy kết quả cuối không phải kiểm định hoàn toàn độc lập. Cần một tập dữ liệu ngoài chưa sử dụng để đánh giá khả năng khái quát hóa.

OOF hiện cũng không phải nested CV: tham số được chọn bằng CV trên train, sau đó tính lại OOF với tham số đã chọn. Không trình bày điểm OOF như ước lượng hoàn toàn không thiên lệch.

### 09 — Thay kết quả ở Bảng 3.1

Giữ kiểu trình bày bảng của trường khi nhập lại các số dưới đây. Đây là bốn cấu hình cuối ở ngưỡng ưu tiên độ nhạy, không phải cùng ngưỡng 0,5.

| Mô hình | Ngưỡng | Recall | Precision | F1 | F2 | ROC-AUC | Specificity | TP | FN | FP | TN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0,01713381 | 86,00% | 8,62% | 0,1566 | 0,3076 | 0,8418 | 53,09% | 43 | 7 | 456 | 516 |
| Random Forest | 0,01348052 | 88,00% | 8,30% | 0,1517 | 0,3014 | 0,8426 | 50,00% | 44 | 6 | 486 | 486 |
| XGBoost | 0,16404158 | 90,00% | 8,82% | 0,1607 | 0,3169 | 0,8504 | 52,16% | 45 | 5 | 465 | 507 |
| LightGBM | 0,09241258 | 90,00% | 8,38% | 0,1533 | 0,3053 | 0,8501 | 49,38% | 45 | 5 | 492 | 480 |

**Caption đề xuất:**

> Bảng 3.1. Kết quả các mô hình ở cấu hình ưu tiên độ nhạy trên tập kiểm tra phát triển gồm 1.022 mẫu, trong đó 50 mẫu stroke=1. Ngưỡng tính từ OOF train; tập kiểm tra đã được dùng trong quá trình phát triển.

**Có thể thêm một câu dưới bảng thay vì thêm cột làm đổi bố cục:**

> Average Precision của Logistic Regression, Random Forest, XGBoost và LightGBM lần lượt là 0,2698; 0,2535; 0,2748; 0,2889. Mã nguồn lưu đại lượng dưới khóa pr_auc nhưng tính bằng average_precision_score, không phải tích phân hình thang của đường Precision–Recall.

Các dòng Decision Tree/Default/Balanced F1/Dummy trong bảng cũ không thuộc lần chạy cuối. Nếu giữ, tách rõ “thực nghiệm giai đoạn trước”, ghi nguồn/lần chạy, không dùng như bằng chứng trực tiếp cho tuning mới. Với Dummy luôn đoán 0, có thể nêu riêng kết quả lý thuyết: Accuracy=972/1022≈95,11%, Recall=0. Không ghi Logistic Regression mới là “Baseline - SMOTE”.

### 10 — Thay nhận xét so sánh mô hình

**Vị trí:** 2.2.4, nhận xét dưới Bảng 3.1 và 3.2.1.2.

**Bỏ:** “đánh bại hoàn toàn ... trên tất cả chỉ số”, “giảm 30 FP”, “44/50, FN=6” của model chính.

**Thay bằng:**

> XGBoost được chọn theo tiêu chí trên OOF train. Trên tập kiểm tra phát triển, mô hình đạt AUC 0,8504, Recall 90%, Precision 8,82%, F1 0,1607 và F2 0,3169. Mô hình phát hiện 45/50 mẫu dương tính, bỏ sót 5 mẫu và tạo 465 cảnh báo giả. So với Logistic Regression theo cùng chính sách ưu tiên độ nhạy, XGBoost phát hiện thêm 2 mẫu dương tính nhưng tăng 9 FP và có Specificity thấp hơn. LightGBM cũng đạt Recall 90% nhưng tạo nhiều FP hơn XGBoost (492 so với 465). Không mô hình nào vượt trội ở mọi độ đo.

Không gọi XGBoost có mọi chỉ số “lớn nhất”: LightGBM có OOF AUC/AP tốt hơn ở một số bảng. Quy tắc chọn cuối là Precision OOF dưới ràng buộc độ nhạy.

### 11 — Các ảnh kết quả thực nghiệm cũ

**Vị trí:** Hình 2.8–2.9, 3.4–3.6.

- Chụp lại kết quả lần chạy cuối hoặc dựng từ validation_metrics.json; không giữ ảnh Decision Tree/LightGBM Tuned rồi mô tả như bản cuối.
- Hình 3.4: dùng số liệu mới của Bảng 3.1.
- Hình 3.5: XGBoost chính dùng [[507,465],[5,45]], hàng=nhãn thật, cột=nhãn dự đoán. Nếu cần bốn ma trận, dùng đủ bốn ma trận trong JSON.
- Hình 3.6: đường ROC và nhãn/AUC của lần chạy mới.
- `outputs/final/evaluation.png` có ROC bốn mô hình và ma trận XGBoost trong một hình ghép. Có thể tách/dựng lại cho đúng bố cục/caption của trường.
- Các ảnh `bar_chart_comparison.png`, `confusion_matrices_1x4.png`, `roc_curve_all_models.png` còn trong outputs/models là ảnh lịch sử; không dùng cho bản mới khi chưa cập nhật.

### 12 — Bảng 2.6, Hình 2.7 và diễn giải feature importance

**Đang ghi:** age=33,65%, BMI=31,20%, glucose=25,48%, các nhóm chiếm trên 98%, “khớp hoàn toàn” với y văn.

Không giữ các số này cho model mới. Đề xuất đổi tiêu đề bảng thành “Mức đóng góp SHAP trung bình tuyệt đối trên mẫu khảo sát”, dùng `global_shap_` trong artifact hoặc `outputs/final/global_shap.png`.

| Đặc trưng | Mean absolute SHAP, thang đầu ra thô |
| --- | ---: |
| num__age | 1,559379 |
| num__bmi | 0,114511 |
| num__avg_glucose_level | 0,109938 |
| pass__hypertension | 0,055523 |
| cat__work_type_Self-employed | 0,034899 |
| cat__gender_Female | 0,002639 |

Các đặc trưng còn lại bằng 0 trên 200 mẫu khảo sát của artifact đang giao. Đây **không phải phần trăm**; không tự lấp lại tỷ trọng cũ. Không suy ra SHAP thấp nghĩa là thuộc tính không quan trọng về y học. Giữ đủ 10 đầu vào.

Lưu ý tái lập: artifact hiện giao thống kê SHAP trên 200 mẫu của bộ dữ liệu phát triển; script train hiện tính thống kê này trên 200 mẫu train. Nếu retrain, cập nhật bảng/hình theo artifact mới, không cố giữ các giá trị trên.

### 13 — Ý nghĩa SHAP

**Vị trí:** 3.2.1.3 và các mô tả biểu đồ giải thích.

**Đang ghi:** SHAP phân rã chênh lệch “xác suất bệnh nhân với xác suất nền”.

**Thay bằng:**

> Với XGBoost và TreeExplainer hiện dùng, SHAP giải thích đầu ra thô trên thang log-odds. Giá trị dương làm tăng điểm so với giá trị nền; giá trị âm làm giảm điểm. Backend cộng các cột One-Hot về nhóm thuộc tính gốc để hiển thị. Không diễn giải giá trị SHAP như phần trăm nguy cơ, tác dụng điều trị hoặc bằng chứng nhân quả.

### 14 — Đường dẫn và phạm vi của pipeline

**Vị trí:** 3.1.2.

**Đang ghi:** models/model_pipeline.pkl; toàn bộ logic AI được cô lập hoàn toàn trong pipeline; đóng gói tự nó loại bỏ leakage.

**Thay bằng:**

> FastAPI nạp outputs/models/model_pipeline.pkl và khởi tạo TreeExplainer trong RAM khi khởi động. Pipeline gồm tiền xử lý và classifier; Backend chuyển 10 trường thành DataFrame, gọi predict_proba, đọc decision_threshold_ để phân loại hoặc dùng ngưỡng riêng của bác sĩ. Backend tính/nhóm SHAP và lưu lịch sử. Đóng gói giúp nhất quán tiền xử lý; chống leakage còn phụ thuộc cách chia và fit trong CV.

Backend không dùng pipeline.predict() ở ngưỡng 0,5. Không khẳng định mọi thuật toán mới đều thay nóng mà không cần kiểm tra schema/ngưỡng/khả năng giải thích.

### 15 — Thuật ngữ kết quả và mức độ kết luận y tế

**Vị trí:** UC02, 3.2.1, 3.3.3 và Kết luận.

- Trong phần kết quả hệ thống, dùng “điểm nguy cơ/điểm xác suất đầu ra của mô hình, chưa hiệu chuẩn thành nguy cơ lâm sàng” thay khẳng định đây là xác suất mắc bệnh thật. Không bắt buộc đổi tên đề tài đã duyệt trên bìa.
- “972 người khỏe mạnh” → “972 mẫu nhãn stroke=0”; nhãn này không xác nhận không có bệnh khác.
- Thay “giải quyết triệt để”, “FP hoàn toàn không gây tổn hại”, “cứu thêm ca bệnh” bằng đoạn thận trọng:

> Kết quả cho thấy tiềm năng nghiên cứu sàng lọc ban đầu. Cảnh báo giả có thể làm tăng lo lắng và nhu cầu kiểm tra thêm, trong khi âm tính giả vẫn có thể xảy ra. Tác động thực tế cần được đánh giá với cơ sở y tế.

Đây là chỉnh phạm vi kết luận, không phải bổ sung kết quả thử nghiệm lâm sàng mới.

## 3. Các điểm cần sửa về chức năng và thiết kế

### 16 — Luồng chọn vai trò và demo

**Vị trí:** 2.3.1, UC01 và phụ lục A.

**Bổ sung:**

> Chế độ demo local cho phép đăng ký Người dùng/Bác sĩ và có ba nút đăng nhập nhanh bằng tài khoản mẫu Người dùng/Bác sĩ/Admin. Tài khoản demo dùng chung, dữ liệu khởi tạo là giả lập. Khi DEMO_MODE=0, đăng nhập demo tắt và quyền bác sĩ do Admin cấp. Đăng ký công khai không cấp quyền admin.

Tách rõ: `/auth/register` trả tài khoản HTTP 201, không tự trả JWT; `/auth/login` mới trả token. Demo dùng `/auth/demo/{role}`. Đăng ký không tự chuyển vào dashboard như cách UC01 hiện viết gộp.

### 17 — Chức năng bác sĩ mới

**Vị trí:** 2.3.1, UC04, sơ đồ use case và phần kết quả giao diện.

**Bổ sung đoạn:**

> Bác sĩ tạo hồ sơ bằng họ tên/khu vực mà không yêu cầu bệnh nhân đăng nhập, tìm theo tên/email, xem lịch sử và sàng lọc cho hồ sơ được chọn. Có thể chỉnh ngưỡng từng lần đo và nhập CSV UTF-8/XLSX tối đa 100 dòng, 2 MB. Tệp cần 10 cột sức khỏe; patient_id tùy chọn để ghi vào hồ sơ cũ. Nếu không có patient_id, mỗi dòng hợp lệ tạo hồ sơ mới. Kết quả hiển thị theo dòng thành công/lỗi và có bộ lọc ca vượt ngưỡng.

Không mô tả toàn bộ tệp là một transaction nguyên khối: các dòng thành công có thể đã được lưu trước khi phát hiện lỗi ở dòng khác. Mở rộng UC04 hoặc thêm luồng con theo quy ước đánh số nhóm/trường, không cần tự đổi toàn bộ mã UC.

### 18 — Ngưỡng màu và nút trong UC02

**Vị trí:** Bảng 2.8, đặc biệt bước 10 có công thức Word Math.

**Đang ghi:** đỏ >=50%, xanh <50%; nút “Gửi dữ liệu và phân tích”, “Nạp dữ liệu mẫu”.

**Thay bằng:**

> Gọi p là điểm mô hình, t là ngưỡng áp dụng. Đỏ khi prediction_result=1 (p>=t); vàng khi t/2<=p<t; xanh khi p<t/2. Đây là quy ước giao diện, không phải phân tầng nguy cơ lâm sàng. Ngưỡng mặc định t≈0,16404; nếu bác sĩ chọn ngưỡng khác thì dùng ngưỡng đó.

Nút mới: “Phân tích nguy cơ”, “Điền dữ liệu mẫu”. Bổ sung ngoại lệ 503 khi model/SHAP chưa sẵn sàng; 422 khi sai schema; 403 khi tài khoản khóa. Không có kết quả nguy cơ 0% giả lập khi nạp model lỗi.

### 19 — UC03 không dùng Chart.js nữa

**Vị trí:** Bảng 2.9, bước 6.

**Thay bằng:**

> Frontend dùng JavaScript thuần với HTML Canvas để vẽ diễn biến điểm nguy cơ. Bảng và biểu đồ không cần tải Chart.js từ CDN.

Bảng mới gồm ngày đo, đường huyết, BMI, điểm, ngưỡng và kết quả; không còn cột huyết áp riêng như mô tả cũ. Đầu vào hypertension chỉ là tiền sử tăng huyết áp 0/1, không phải số đo huyết áp tâm thu/tâm trương. Ngày hiển thị có năm/giây và UTC+7; biểu đồ tăng theo thời gian, bảng gần đây mới nhất trước.

### 20 — Schema CSDL khác trực tiếp với mô hình đang vẽ

**Vị trí:** 2.3.3, Hình 2.23–2.26.

**Đang ghi:** username; Predictions chứa 10 cột riêng; History có prediction_id/result/confidence/message/explanation_data; quan hệ users→predictions→history.

**Thay bằng schema thật:**

| Bảng | Cột trong models.py | Vai trò |
| --- | --- | --- |
| users | id, full_name, email, hashed_password, role, is_active, region, created_at | Tài khoản/hồ sơ bệnh nhân role=user; email đăng nhập; không có username |
| history | id, user_id, action, timestamp | Luồng sàng lọc chính ghi vào đây; user_id→users.id |
| patients | id, patient_code, full_name, age, gender | Có khai báo sẵn, chưa được API hồ sơ hiện tại dùng làm nguồn chính |
| predictions | id, patient_id, input_data, result, confidence, created_at | Có khai báo, patient_id→patients.id; /predict hiện không ghi bảng này |

`history.action` chứa JSON được mã hóa: performed_by, input (10 trường), prediction_result, threshold, model_name, shap, glucose, bmi, bp, risk. performed_by nằm trong JSON, không phải foreign key riêng. risk trong lịch sử là điểm nhân 100; risk_probability response thuộc [0,1].

**Vẽ lại ERD:** users.id→history.user_id (1–n) là luồng chính. Nếu vẽ đủ schema thì thêm patients.id→predictions.patient_id và ghi hai bảng chưa dùng trong luồng chính. Không vẽ History nối Predictions bằng prediction_id vì cột đó không tồn tại. Hình bảng Users phải bổ sung is_active, bỏ username; tên id và kiểu Text/DateTime theo models.py.

### 21 — Sửa các bước lưu/truy vấn trong use case và UML

- UC02 bước 8: “lưu vào Predictions rồi History” → “lưu đầu vào, nhãn, ngưỡng, SHAP vào JSON trong history.action; gắn user_id với hồ sơ được sàng lọc”.
- Hình 2.15 (tuần tự UC03) đang join History/Predictions: đổi thành truy vấn History trực tiếp theo user_id.
- UC04/Hình 2.18: lấy users với role=user, không phải tùy chọn User/Patient; endpoint thật là `/doctor/patient/{patient_id}/history`.
- UC05 phụ lục A: “truy vấn Users, Predictions, History” → “duyệt hồ sơ users(role=user) và history tương ứng”.

### 22 — Phạm vi xuất báo cáo

**Vị trí:** UC05 phụ lục A.

**Thay/bổ sung:**

> Bác sĩ gọi GET /doctor/export?format=xlsx hoặc format=pdf kèm JWT. Báo cáo gồm mã bệnh nhân, họ tên, email, ngày đo, đường huyết, BMI, điểm và nhãn sàng lọc. API xuất tổng hợp lịch sử của tất cả hồ sơ role=user, không tự giới hạn theo từ khóa đang tìm trên màn hình. PDF backend dùng chữ không dấu để tương thích font chuẩn; tính năng In/Lưu PDF trình duyệt có thể giữ tiếng Việt.

Không ghi file xuất chứa đầy đủ 10 chỉ số hoặc biểu đồ SHAP nếu thực tế chưa có. Dùng “báo cáo kết quả sàng lọc” thay “báo cáo chẩn đoán” trong phần giải thích chức năng khi phù hợp tên UC đã duyệt.

### 23 — UC07 chưa có giám sát tải API đầy đủ

**Đang ghi:** giám sát tải API, nhật ký hệ thống và kiểm tra phiên bản mô hình.

**Thay bằng:**

> Admin xem số tài khoản/bác sĩ/lượt sàng lọc, ca vượt ngưỡng theo tuổi/khu vực, lịch sử toàn hệ thống, thuật toán/file mô hình, ngưỡng, chỉ số tập kiểm tra phát triển và trạng thái SQLite. Có thể nạp lại model và xóa một lịch sử theo mã.

Chưa có dashboard RPS/CPU/RAM/latency, audit log đầy đủ mọi thao tác hay model registry/versioning chuyên dụng. Không coi tên file là bằng chứng hệ thống quản lý phiên bản mô hình.

### 24 — Phạm vi bảo mật

**Vị trí:** 1.4, 2.3.1, 3.3.3.2, Kết luận.

**Bỏ:** “bảo mật tuyệt đối”, “chuẩn y khoa”, “JWT/bcrypt mã hóa bệnh án”, “đã tuân thủ HIPAA”.

**Thay bằng:**

> JWT phục vụ xác thực/phân quyền, mật khẩu được băm bcrypt. Tên, khu vực và JSON sức khỏe trong history.action được mã hóa đối xứng Fernet bằng cryptography; một số cột ở bảng dự phòng cũng dùng kiểu mã hóa. Email vẫn tra cứu được, không mã hóa toàn bộ SQLite. Chạy local bằng HTTP, chưa có TLS. Đây là mức bảo mật học thuật, chưa chứng minh tuân thủ tiêu chuẩn y tế/HIPAA.

Bổ sung bảo quản `.data_key`/DATA_ENCRYPTION_KEY và `.jwt_secret`/JWT_SECRET_KEY, không đưa khóa thật vào báo cáo/ảnh/Git. Mất khóa dữ liệu có thể không đọc lại được dữ liệu. Mã hóa không đồng nghĩa ẩn danh hóa: hồ sơ vẫn gắn tài khoản.

### 25 — Cập nhật UML và kiến trúc

**Vị trí:** Hình 2.10–2.22, 3.1–3.2.

- Hình 2.10: thêm bác sĩ sàng lọc cho hồ sơ, nhập danh sách, chỉnh ngưỡng; thêm thống kê admin.
- Hình 2.12: bổ sung ngưỡng từ artifact/ngưỡng bác sĩ và lỗi 503. Các Controller có thể giữ như thành phần trách nhiệm thiết kế; không ghi đó là các class riêng đã tồn tại nếu code không có.
- Hình 2.15: bỏ join Predictions như mục 21.
- Hình 2.18: history dùng user_id của hồ sơ role=user; endpoint patient số ít đúng code.
- Hình 2.21: bổ sung khóa/mở khóa và bảo vệ admin tự khóa.
- Hình 3.1: Backend↔AI Core đang ghi HTTP/JSON; sửa thành “gọi pipeline/SHAP trực tiếp trong cùng tiến trình Python”. HTTP/JSON là trình duyệt↔FastAPI, không có model server riêng.
- Khối AI Core ghi XGBoost chính. SMOTE chỉ là lựa chọn khảo sát khi train, không chạy trong suy luận, không được chọn trong cấu hình thắng.
- Hình 3.2: dữ liệu thô→preprocessor→predict_proba→áp dụng ngưỡng; SHAP là nhánh giải thích ở backend, không cho SMOTE sinh mẫu khi người dùng dự đoán.

Các ảnh sơ đồ đã được kiểm tra về nội dung; tài liệu này không thay hay đánh giá lại format trang Word.

### 26 — Thay ảnh giao diện cũ

**Vị trí:** Hình 2.27–2.29, 3.7–3.8; bổ sung minh họa bác sĩ/admin nếu cần.

Chụp lại với dữ liệu giả:

1. Đăng nhập có ba nút demo; form đăng ký chọn Người dùng/Bác sĩ.
2. Sàng lọc mới, ngưỡng 16,40% mặc định, thanh SHAP.
3. Lịch sử có thẻ thống kê, biểu đồ và bảng mới.
4. Bác sĩ có tạo hồ sơ, CSV/XLSX, bộ lọc và xuất báo cáo.
5. Admin có thống kê, tài khoản, XGBClassifier/model_pipeline.pkl và ngưỡng mới.

Bạn tự chèn/căn ảnh theo mẫu trường. `outputs/ui-login.png` và `outputs/ui-history.png` là ảnh QA màn hình hẹp trước đó, không nhất thiết hợp bố cục báo cáo; nên chụp mới bản demo sạch.

## 4. Kiểm thử, cài đặt và kết luận

### 27 — Không dùng số thời gian chưa có benchmark tương ứng

**Vị trí:** 2.2.2, 3.2.1.3, 3.3.3.2, Kết luận.

**Đang ghi:** dưới 0,05 giây; trung bình 0,15–0,35 giây; dưới 1,5 giây; mọi trường hợp dưới 2 giây.

**Thay bằng:**

> Các lượt kiểm thử API local gần đây quan sát thời gian dự đoán kèm SHAP khoảng 0,06–0,10 giây. Đây là đo đơn lẻ trên môi trường phát triển, chưa phải trung bình benchmark nhiều mẫu/kiểm thử tải. Phản hồi dưới 2 giây là mục tiêu, chưa chứng minh cho mọi tải và môi trường.

Muốn giữ con số “trung bình” chính thức thì đo lại có số request, cấu hình máy, warm/cold start và p95. Không lấy một lượt test làm benchmark.

### 28 — Cập nhật testcase và bằng chứng Pass

**Vị trí:** 3.3.2.

- KT01: thêm đăng ký bác sĩ khi demo bật, từ chối đăng ký admin, demo tắt thì không đăng nhập nhanh.
- KT02: tên nút mới; tuổi 0–120, BMI 10–100, glucose 1–1000 mg/dL; enum, thiếu trường, NaN/Infinity; 503 khi model lỗi; ngưỡng bác sĩ lưu đúng lịch sử. Đây là giới hạn phần mềm, không chứng minh mọi giá trị trong khoảng hợp lý lâm sàng.
- KT03: API `/history` không JWT trả 401; trang LICH_SU.HTML đưa người chưa đăng nhập về auth. Không đồng nhất API JSON với trang HTML tự chuyển hướng.
- KT04: thêm tạo hồ sơ, lưu kết quả đúng bệnh nhân, CSV có dòng sai, XLSX, thiếu cột/quá giới hạn và lọc ca vượt ngưỡng.
- KT05: khóa/mở khóa, token cũ bị chặn sau khóa, ngăn admin tự khóa.
- Thêm testcase xlsx/pdf export, thống kê, reload và kiểm tra dữ liệu mã hóa.

Bằng chứng hiện tại: `python -m unittest discover -s tests -v` có **một phương thức tích hợp test_complete_flow chứa nhiều assertion**. Không gọi đó là hàng chục testcase tự động độc lập hoặc ghi mọi trường hợp đề xuất trên đã tự động Pass. Phân biệt kịch bản đã đo/chạy với kịch bản cần bạn bổ sung minh chứng.

### 29 — Công nghệ, phiên bản và môi trường

**Vị trí:** Bảng 1.5, 3.1.1 và 3.3.1.2.

- FastAPI 0.110.0 → 0.157.0 nếu ghi theo requirements.txt hiện tại.
- Metadata lần huấn luyện ghi Python 3.13.15. Không khẳng định 3.11–3.13 đều kiểm thử khi chưa có minh chứng.
- Frontend HTML/CSS/JavaScript thuần; Canvas. Không nói toàn bộ hệ thống viết bằng Python.
- Bảng thư viện bổ sung LightGBM và cryptography/Fernet nếu muốn liệt kê đủ thực tế.
- CPU/RAM 16 GB/SSD 500 GB chỉ ghi là cấu hình thực nghiệm nếu đúng máy nhóm dùng; mã nguồn không xác minh phần cứng, không chứng minh đó là cấu hình tối thiểu.
- Figma, PR review, Postman, Excel quản lý testcase chỉ ghi đã dùng nếu nhóm có minh chứng; không suy từ đề cương rằng đã thực hiện.

### 30 — Bổ sung danh sách API ở phụ lục B

Đổi “endpoint 3–9 yêu cầu JWT” thành “endpoint yêu cầu JWT theo cột quyền”. Bảng đầy đủ đề xuất:

| Method | Endpoint | Quyền/phạm vi |
| --- | --- | --- |
| GET | /config | Công khai, trạng thái demo |
| POST | /auth/register | Công khai, user/doctor theo chế độ; không admin |
| POST | /auth/login | Công khai, xác thực email/mật khẩu |
| POST | /auth/demo/{role} | Chỉ khả dụng trong demo; user/doctor/admin |
| GET | /auth/me | Đã đăng nhập |
| POST | /predict | Đã đăng nhập; patient_id/cutoff chỉ cho doctor |
| GET | /history | Lịch sử riêng của tài khoản |
| GET | /doctor/patients?q= | Doctor, tìm hồ sơ role=user |
| POST | /doctor/patients | Doctor, tạo hồ sơ |
| GET | /doctor/patient/{patient_id}/history | Doctor |
| GET | /doctor/template | Doctor, CSV mẫu |
| POST | /doctor/batch?cutoff= | Doctor, multipart CSV/XLSX |
| GET | /doctor/export?format=xlsx hoặc pdf | Doctor |
| GET | /admin/users | Admin |
| PUT | /admin/users/{id} | Admin, role/is_active |
| DELETE | /admin/users/{id} | Admin |
| GET | /admin/stats | Admin |
| GET | /admin/history | Admin, toàn bộ lịch sử |
| GET | /admin/system | Admin, model/ngưỡng/metadata/SQLite |
| POST | /admin/model/reload | Admin |
| DELETE | /admin/history/{id} | Admin |
| GET | /health | Công khai, readiness model |
| GET | /docs | Swagger |
| GET | /web/ | Giao diện static; API vẫn xác thực riêng |

Response /predict: prediction_result, risk_probability, threshold, message, shap, shap_unit. shap_unit=raw_model_output, không phải phần trăm. /predict không giới hạn chỉ role=user như bảng cũ; doctor mới được chọn hồ sơ khác/ngưỡng riêng.

### 31 — Vai trò các tệp ở phụ lục dữ liệu/mô hình

| Tệp | Mô tả nên dùng |
| --- | --- |
| data/raw/healthcare-dataset-stroke-data.csv | Nguồn training cuối |
| src/train_validated.py | Tìm tham số, OOF, ngưỡng, model, metrics/artifact |
| src/week5_full.py | Entry point tương thích gọi train_validated.main |
| src/week4.py | Baseline giai đoạn trước, không tạo bảng cuối mới |
| outputs/models/model_pipeline.pkl | XGBoost phục vụ chính, tiền xử lý/ngưỡng/metadata |
| outputs/models/best_model_week5.pkl | Alias cùng phiên bản XGBoost mới |
| outputs/models/legacy_before_final.pkl | Bản cũ đối chiếu, không phục vụ chính |
| outputs/final/validation_metrics.json | Kết quả lần cuối đã giao |
| outputs/models/validation_metrics.json | Kết quả trong thư mục phục vụ, retrain có thể cập nhật |
| outputs/final/evaluation.png | ROC ứng viên và confusion matrix XGBoost |
| outputs/final/global_shap.png | SHAP tổng thể mẫu khảo sát của bản đã giao |
| train_resampled.npz, test.npz, preprocessor.pkl, feature_names.pkl | Artifact tuần 3, không phải đầu vào trực tiếp training cuối |

Không viết script cuối nạp test.npz để lập bảng mới. Nó chia lại từ CSV; seed/tỷ lệ có thể trùng nhưng đường xử lý khác.

### 32 — Cài đặt, chạy và hướng dẫn demo

**Vị trí:** phụ lục hướng dẫn và Bảng PL3.

**Thay bước train bằng:**

> Nếu thiếu model_pipeline.pkl hoặc dùng --retrain, launcher tạo artifact tuần 3 rồi chạy src/train_validated.py. Script chính tự chia dữ liệu CSV, tìm kiếm và lưu model_pipeline.pkl cùng alias best_model_week5.pkl. Khi có file chính và dữ liệu không cần làm mới, launcher bỏ qua train.

Bổ sung: dừng phiên cũ, chạy `python run.py`, Ctrl+F5 nếu trình duyệt giữ giao diện cũ; ba nút demo; `DEMO_MODE=0` để tắt demo. Thêm URL `http://127.0.0.1:8000/web/` bên cạnh frontend 5500 và /docs.

Tách `git clone <repo-url>cd ...python run.py` thành các dòng, điền repo thật khi công bố. Python 3.13 là bản đã dùng; macOS/Linux/Python khác nên ghi chưa kiểm thử nếu nhóm chưa chạy.

### 33 — Đoạn kết quả đạt được trong Kết luận

**Có thể thay hai đoạn mô hình/kết quả cũ bằng:**

> Nhóm đã xây dựng phiên bản demo hỗ trợ sàng lọc nguy cơ đột quỵ bằng FastAPI, SQLite và giao diện HTML/CSS/JavaScript cho người dùng, bác sĩ và quản trị viên. Quy trình so sánh Logistic Regression, Random Forest, XGBoost và LightGBM bằng CV phân tầng trên train, khảo sát trọng số lớp và SMOTE, sau đó hiệu chỉnh ngưỡng trên dự đoán ngoài fold. XGBoost được chọn triển khai; cấu hình cuối dùng trọng số lớp, không SMOTE. Trên tập kiểm tra phát triển 1.022 mẫu với 50 mẫu dương tính, mô hình đạt ROC-AUC 0,8504 và Recall 90%, nhận diện 45 mẫu, bỏ sót 5 mẫu, tạo 465 cảnh báo giả. Hệ thống có SHAP, lịch sử, nhập CSV/XLSX và ngưỡng bác sĩ; admin quản lý tài khoản, thống kê và nạp lại mô hình. Tập kiểm tra đã được dùng qua nhiều vòng phát triển, nên kết quả chưa thay thế kiểm định độc lập hoặc đánh giá lâm sàng.

### 34 — Hạn chế và hướng phát triển

- “Sẽ thử trọng số lớp” → “mở rộng khảo sát phương pháp imbalance”; trọng số lớp đã dùng.
- “Sẽ mã hóa dữ liệu” → “mở rộng phạm vi mã hóa, quản lý khóa và HTTPS”; đã mã hóa một số cột nhưng chưa email/toàn file SQLite.
- Giữ hướng hiệu chuẩn xác suất, dữ liệu ngoài/Vietnam, đánh giá công bằng theo nhóm, tác động cảnh báo giả.
- Nêu rõ chưa dự báo nguy cơ theo thời hạn 10 năm và chưa đối sánh trực tiếp với thang điểm lâm sàng.

## 5. Những lỗi nội dung/đối chiếu bạn nên tự chỉnh trong Word

### 35 — Tài liệu tham khảo và số trích dẫn không khớp

Các lỗi đối chiếu nội bộ đã thấy:

- Thân bài [1] nói GBD 2019, nhưng danh mục [1] là sách Machine Learning cơ bản.
- [2] nói INTERSTROKE, nhưng danh mục [2] là sách học máy.
- [3] nói Framingham, nhưng danh mục [3] là sách CSDL.
- [6] nói SHAP, nhưng danh mục [6] là hướng dẫn Bộ Y tế.
- [7] nói LIME, nhưng danh mục [7] là Kaggle.
- [12], [13], [14] xuất hiện trong nội dung, trong khi danh mục chỉ đến [11].

**Cách sửa:** lập lại danh sách nguồn thật đã đọc và gắn đúng số. Bổ sung nguồn gốc cho GBD/INTERSTROKE/Framingham/SHAP/LIME/công trình ở Bảng 1.2 nếu giữ; bỏ nhận định chưa có nguồn. Không tự gán nguồn dựa trên tên tác giả chưa xác minh. Lần rà soát này không kiểm chứng thay nhóm toàn bộ nghiên cứu/y văn bên ngoài.

### 36 — So sánh Framingham đang vượt quá bằng chứng

**Vị trí:** 3.3.3.1.

**Thay mở đầu bằng:**

> Nhóm so sánh định tính một số đặc điểm đầu vào và cách trình bày giữa hệ thống nghiên cứu và thang điểm truyền thống. Chưa chạy đối chứng trên cùng quần thể, chưa tính Framingham trên bộ dữ liệu và chưa đủ cơ sở kết luận vượt trội lâm sàng.

Bỏ chữ “DOCX” cuối hai tiêu đề cột. Những khẳng định về hiệu năng, tính xâm lấn và bỏ sót của thang điểm truyền thống cần nguồn phù hợp; không suy từ kết quả của nhóm. Điểm mô hình không có cùng thời hạn 10 năm để so trực tiếp hai con số.

### 37 — Số mục, caption và lỗi chữ

Bạn tự thao tác theo template trường, không cần thay thiết kế báo cáo:

- Bảng 3.3 bị lặp cho lịch trình/phần cứng/phần mềm/công cụ; danh mục thiếu Bảng 3.2. Đánh lại tuần tự và các dẫn chiếu.
- Bảng 2.5 nằm trong Danh mục hình ảnh: chuyển đúng Danh mục bảng.
- Hai mục cùng “Phụ lục C”: có thể đổi hướng dẫn cài đặt thành D theo quy định trường.
- PL3 dùng cho hai bảng khác nhau: đánh lại mã phụ lục.
- 3.3.1.2 lặp tiêu đề; “5. Chức năng Quản lý người dùng” chưa theo chuỗi 3.3.2.x.
- `df.infor()` → `df.info()`; “minh minh bạch” → “minh bạch”; “Nguời” → “Người”; “thực thể liên kế” → “thực thể liên kết”; “chạy локально” → “chạy cục bộ”.
- Hình 3.3 kết thúc bằng “tại đường dẫn” nhưng thiếu URL: thêm `http://127.0.0.1:8000/docs`.
- Không xóa công thức vì bản text có chỗ trống: báo cáo dùng Word Math. Ví dụ UC02 thật sự có >=50% / <50%; sửa nội dung ngưỡng bên trong công thức và giữ format.
- Sau khi tự sửa nội dung, cập nhật trường mục lục/danh mục bảng/hình và kiểm tra phân trang theo mẫu nhà trường.

## 6. Thứ tự sửa đề xuất

1. Tên model, ngưỡng, quy trình và bảng kết quả: 01–14.
2. CSDL/ERD và cách lưu/truy vấn: 20–21.
3. Demo, bác sĩ/admin, ảnh giao diện, API và cài đặt: 16–26, 30–32.
4. Kết luận, giới hạn, benchmark và mức độ tuyên bố: 15, 24, 27–29, 33–36.
5. Nguồn tham khảo, caption, đánh số và trường tự động: 35, 37.

**Có thể giữ về cơ bản:** tên đề tài đã duyệt; phạm vi hỗ trợ sàng lọc; dataset 5.110/249; 10 đầu vào; 21 cột trong artifact hiện tại; split 80/20 seed 42; median BMI; One-Hot năm biến; StandardScaler ba biến số; JWT/bcrypt; ba vai trò; tuyên bố không thay thế bác sĩ. Không cần viết lại toàn bộ lý thuyết hoặc đổi công nghệ.

## 7. Xác nhận phạm vi rà soát

Chỉ tạo Markdown hướng dẫn này và bản trích xuất/ảnh kiểm tra nội bộ ở outputs. Không sửa file Word, format, code hoặc artifact trong lượt rà soát này; không huấn luyện lại.

SHA-256 file Word khi đối chiếu: `8ca58d0c37a523644e5197f9ec7169ea6f9b0a56a023435ec6032af7b9766ba5`.
