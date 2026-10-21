
## 1. Tổng quan tập dữ liệu

| Thuộc tính | Giá trị |
|------------|---------|
| Tên dataset | Stroke Prediction Dataset |
| Nguồn | Kaggle – fedesoriano |
| Định dạng | CSV |
| Số mẫu (rows) | 5.110 |
| Số cột (columns) | 12 |
| Bài toán | Phân loại nhị phân (Binary Classification) |
| Biến mục tiêu | `stroke` (0 = Không đột quỵ, 1 = Có đột quỵ) |
| Tỷ lệ lớp dương | 249 / 5.110 ≈ **4,87%** |
| Tỷ lệ lớp âm | 4.861 / 5.110 ≈ **95,13%** |
| Missing values | Chỉ cột `bmi` (201 giá trị ≈ 3,93%) |

Tập dữ liệu chứa các yếu tố nguy cơ lâm sàng phổ biến của đột quỵ: tuổi, tăng huyết áp, bệnh tim, đường huyết, chỉ số BMI và tình trạng hút thuốc. Sự mất cân bằng lớp nghiêm trọng là thách thức chính cần xử lý trong quá trình xây dựng mô hình.

---

## 2. Mô tả chi tiết từng cột (Data Dictionary)

### 2.1. `id`

- **Tên đầy đủ:** Patient Identifier  
- **Kiểu dữ liệu:** `int64` (số nguyên)  
- **Đơn vị:** Không có  
- **Khoảng giá trị hợp lệ:** 67 – 72.940  
- **Mô tả:** Mã định danh duy nhất của từng bệnh nhân trong tập dữ liệu.  
- **Vai trò:** **Không sử dụng làm feature**. Cột này sẽ bị loại bỏ trước khi huấn luyện mô hình.  
- **Ghi chú Frontend (tooltip):** “Mã định danh bệnh nhân – không dùng để dự đoán.”

---

### 2.2. `gender`

- **Tên đầy đủ:** Gender  
- **Kiểu dữ liệu:** `object` (categorical / nominal)  
- **Đơn vị:** Không có  
- **Khoảng giá trị hợp lệ:** `Male`, `Female`, `Other`  
- **Mô tả:** Giới tính của bệnh nhân.  
- **Phân bố:** Nữ chiếm đa số. Chỉ có **1 mẫu** mang giá trị `Other` (rất hiếm).  
- **Cách mã hóa đề xuất:** One-Hot Encoding  
- **Ghi chú Frontend (tooltip):** “Giới tính: Nam / Nữ / Khác.”

---

### 2.3. `age`

- **Tên đầy đủ:** Age  
- **Kiểu dữ liệu:** `float64` (numerical continuous)  
- **Đơn vị:** Năm tuổi  
- **Khoảng giá trị hợp lệ:** **0,08 – 82**  
- **Mô tả:** Tuổi của bệnh nhân. Giá trị thập phân nhỏ hơn 1 là hợp lệ (bao gồm trẻ sơ sinh và trẻ nhỏ).  
- **Thống kê mô tả:**  
  - Trung bình ≈ 43,2  
  - Trung vị ≈ 45  
  - Nhóm đột quỵ có tuổi trung vị ≈ **71**, nhóm không đột quỵ ≈ **43**  
- **Vai trò:** **Yếu tố nguy cơ mạnh nhất** theo kết quả EDA và feature importance.  
- **Ghi chú Frontend (tooltip):** “Tuổi bệnh nhân (năm). Nguy cơ đột quỵ tăng mạnh theo tuổi.”

---

### 2.4. `hypertension`

- **Tên đầy đủ:** Hypertension (Tăng huyết áp)  
- **Kiểu dữ liệu:** `int64` (binary)  
- **Đơn vị:** Không có  
- **Khoảng giá trị hợp lệ:** `0` hoặc `1`  
- **Mô tả:**  
  - `0` = Không có tiền sử tăng huyết áp  
  - `1` = Có tiền sử tăng huyết áp  
- **Phân bố:** Phần lớn bệnh nhân không tăng huyết áp.  
- **Tỷ lệ đột quỵ:**  
  - Không tăng huyết áp: ≈ **4,0%**  
  - Có tăng huyết áp: ≈ **13,3%** (tăng khoảng 3,3 lần)  
- **Ghi chú Frontend (tooltip):** “Tiền sử tăng huyết áp: 0 = Không, 1 = Có. Làm tăng nguy cơ đột quỵ rõ rệt.”

---

### 2.5. `heart_disease`

- **Tên đầy đủ:** Heart Disease (Bệnh tim mạch)  
- **Kiểu dữ liệu:** `int64` (binary)  
- **Đơn vị:** Không có  
- **Khoảng giá trị hợp lệ:** `0` hoặc `1`  
- **Mô tả:**  
  - `0` = Không có tiền sử bệnh tim  
  - `1` = Có tiền sử bệnh tim  
- **Tỷ lệ đột quỵ:**  
  - Không bệnh tim: ≈ **4,2%**  
  - Có bệnh tim: ≈ **17,0%** (tăng khoảng 4 lần)  
- **Ghi chú Frontend (tooltip):** “Tiền sử bệnh tim: 0 = Không, 1 = Có. Yếu tố làm tăng nguy cơ đột quỵ mạnh.”

---

### 2.6. `ever_married`

- **Tên đầy đủ:** Ever Married (Tình trạng hôn nhân)  
- **Kiểu dữ liệu:** `object` (categorical / binary)  
- **Đơn vị:** Không có  
- **Khoảng giá trị hợp lệ:** `Yes`, `No`  
- **Mô tả:** Bệnh nhân đã từng kết hôn hay chưa.  
- **Phân bố:** Người đã kết hôn chiếm đa số (tương quan với tuổi).  
- **Cách mã hóa đề xuất:** Label Encoding (`Yes` → 1, `No` → 0)  
- **Ghi chú Frontend (tooltip):** “Đã từng kết hôn: Yes / No.”

---

### 2.7. `work_type`

- **Tên đầy đủ:** Work Type (Loại hình công việc)  
- **Kiểu dữ liệu:** `object` (categorical / nominal)  
- **Đơn vị:** Không có  
- **Khoảng giá trị hợp lệ:**  
  - `Private` – Làm việc tư nhân  
  - `Self-employed` – Tự kinh doanh  
  - `Govt_job` – Công chức / viên chức  
  - `children` – Trẻ em  
  - `Never_worked` – Chưa từng làm việc  
- **Phân bố:** `Private` chiếm tỷ lệ lớn nhất.  
- **Cách mã hóa đề xuất:** One-Hot Encoding  
- **Ghi chú Frontend (tooltip):** “Loại công việc: Private / Self-employed / Govt_job / children / Never_worked.”

---

### 2.8. `Residence_type`

- **Tên đầy đủ:** Residence Type (Nơi cư trú)  
- **Kiểu dữ liệu:** `object` (categorical / binary)  
- **Đơn vị:** Không có  
- **Khoảng giá trị hợp lệ:** `Urban` (Thành thị), `Rural` (Nông thôn)  
- **Phân bố:** Hai nhóm gần cân bằng.  
- **Cách mã hóa đề xuất:** Label Encoding (`Urban` → 1, `Rural` → 0)  
- **Ghi chú Frontend (tooltip):** “Nơi ở: Urban (Thành thị) / Rural (Nông thôn).”

---

### 2.9. `avg_glucose_level`

- **Tên đầy đủ:** Average Glucose Level (Mức đường huyết trung bình)  
- **Kiểu dữ liệu:** `float64` (numerical continuous)  
- **Đơn vị:** **mg/dL** (milligram trên decilít)  
- **Khoảng giá trị hợp lệ:** **55,12 – 271,74**  
- **Mô tả:** Mức đường huyết trung bình trong máu.  
- **Thống kê mô tả:**  
  - Trung bình ≈ 106,1  
  - Trung vị ≈ 91,9  
  - Phân phối **lệch phải** (phần lớn ở mức bình thường, một nhóm giá trị cao)  
  - Nhóm đột quỵ có median cao hơn (≈ 105 so với ≈ 91,5)  
- **Ghi chú Frontend (tooltip):** “Đường huyết trung bình (mg/dL). Giá trị cao làm tăng nguy cơ đột quỵ.”

---

### 2.10. `bmi`

- **Tên đầy đủ:** Body Mass Index (Chỉ số khối cơ thể)  
- **Kiểu dữ liệu:** `float64` (numerical continuous)  
- **Đơn vị:** **kg/m²**  
- **Khoảng giá trị hợp lệ:** **10,3 – 97,6** (trên các giá trị không thiếu)  
- **Mô tả:** Chỉ số BMI = cân nặng (kg) / chiều cao² (m²).  
- **Missing values:** **201 giá trị** (≈ 3,93%) – cột duy nhất bị thiếu dữ liệu.  
- **Đặc điểm:** Phân phối lệch phải, tồn tại nhiều outlier cao.  
- **Chiến lược imputation đề xuất:** **Median** (bền vững hơn mean khi có outlier và phân phối lệch).  
- **Ghi chú Frontend (tooltip):** “Chỉ số BMI (kg/m²). Có thể thiếu dữ liệu; giá trị cao liên quan đến nguy cơ mạch máu.”

---

### 2.11. `smoking_status`

- **Tên đầy đủ:** Smoking Status (Tình trạng hút thuốc)  
- **Kiểu dữ liệu:** `object` (categorical / nominal)  
- **Đơn vị:** Không có  
- **Khoảng giá trị hợp lệ:**  
  - `never smoked` – Chưa bao giờ hút thuốc  
  - `formerly smoked` – Đã từng hút thuốc  
  - `smokes` – Đang hút thuốc  
  - `Unknown` – Không có thông tin  
- **Phân bố:** `never smoked` và `Unknown` chiếm tỷ lệ cao. Nhóm `formerly smoked` có xu hướng tỷ lệ đột quỵ cao hơn một chút.  
- **Cách mã hóa đề xuất:** One-Hot Encoding  
- **Ghi chú Frontend (tooltip):** “Tình trạng hút thuốc: never smoked / formerly smoked / smokes / Unknown.”


### 2.12. `stroke` (Target)

- **Tên đầy đủ:** Stroke (Đột quỵ)  
- **Kiểu dữ liệu:** `int64` (binary – **biến mục tiêu**)  
- **Đơn vị:** Không có  
- **Khoảng giá trị hợp lệ:** `0` hoặc `1`  
- **Mô tả:**  
  - `0` = Không bị đột quỵ  
  - `1` = Đã từng bị đột quỵ  
- **Phân bố:**  
  - Lớp 0: 4.861 mẫu (**95,13%**)  
  - Lớp 1: 249 mẫu (**4,87%**)  
- **Ghi chú quan trọng:** Mất cân bằng lớp cực đoan → không dùng Accuracy làm độ đo chính; ưu tiên **Recall**, F1-score và AUC-ROC.  
- **Ghi chú Frontend (tooltip):** “Kết quả đột quỵ: 0 = Không, 1 = Có. Đây là biến cần dự đoán.”


## 3. Phân nhóm thuộc tính

### Biến số liên tục (Numerical Continuous)
- `age` (năm)
- `avg_glucose_level` (mg/dL)
- `bmi` (kg/m²)

### Biến nhị phân (Binary 0/1)
- `hypertension`
- `heart_disease`
- `stroke` (target)

### Biến phân loại (Categorical)
- Nominal (≥ 3 nhóm): `gender`, `work_type`, `smoking_status` → **One-Hot Encoding**
- Binary string: `ever_married`, `Residence_type` → **Label Encoding** (map 0/1)

### Biến loại bỏ
- `id`

## 4. Feature Importance cơ bản (từ Decision Tree)

Trong các mô hình dựa trên cây quyết định (Decision Tree, Random Forest, XGBoost…), thuộc tính **feature importance** đo mức độ đóng góp của từng biến vào việc giảm độ không thuần khiết (impurity) khi phân nhánh.

Cách lấy feature importance với scikit-learn:

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# Lấy độ quan trọng
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print(feat_imp)
```

**Kỳ vọng từ EDA (thứ tự quan trọng sơ bộ):**
1. `age` – mạnh nhất  
2. `avg_glucose_level`  
3. `bmi`  
4. `hypertension`, `heart_disease`  
5. Các biến categorical còn lại  


## 5. Hướng dẫn sử dụng cho Frontend (Tooltip)

```json
{
  "age": "Tuổi bệnh nhân (năm). Nguy cơ đột quỵ tăng mạnh theo tuổi.",
  "hypertension": "Tiền sử tăng huyết áp: 0 = Không, 1 = Có. Làm tăng nguy cơ đột quỵ rõ rệt.",
  "heart_disease": "Tiền sử bệnh tim: 0 = Không, 1 = Có. Yếu tố làm tăng nguy cơ đột quỵ mạnh.",
  "avg_glucose_level": "Đường huyết trung bình (mg/dL). Giá trị cao làm tăng nguy cơ đột quỵ.",
  "bmi": "Chỉ số BMI (kg/m²). Có thể thiếu dữ liệu; giá trị cao liên quan đến nguy cơ mạch máu.",
  "smoking_status": "Tình trạng hút thuốc: never smoked / formerly smoked / smokes / Unknown.",
  "gender": "Giới tính: Nam / Nữ / Khác.",
  "ever_married": "Đã từng kết hôn: Yes / No.",
  "work_type": "Loại công việc: Private / Self-employed / Govt_job / children / Never_worked.",
  "Residence_type": "Nơi ở: Urban (Thành thị) / Rural (Nông thôn)."
}

