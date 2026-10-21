<div align="center">

# 🧠 Stroke Risk Screening System (cách chạy dự án trước tuần 3)

**Hệ thống hỗ trợ sàng lọc sớm nguy cơ đột quỵ dựa trên dữ liệu y tế bằng Machine Learning**

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-F7931E?logo=scikitlearn&logoColor=white)
![JavaScript](https://shields.io/badge/JavaScript-F7DF1E?logo=JavaScript&logoColor=000&style=flat-square)
![Status](https://img.shields.io/badge/status-in%20development-yellow)

</div>

> ⚠️ **Tuyên bố miễn trừ trách nhiệm (Medical Disclaimer)**
> Hệ thống chỉ mang tính chất tham khảo và hỗ trợ sàng lọc, không thay thế hoàn toàn chỉ định chẩn đoán chính thức từ Bác sĩ chuyên khoa.
> Dự án phục vụ mục đích học tập (đồ án Thực tập cơ sở ngành), **chưa được kiểm chứng lâm sàng** và không dùng để ra quyết định điều trị.

**Mục lục:** [Bài toán](#1-bài-toán) · [Kiến trúc](#2-kiến-trúc-hệ-thống-4-khối) · [Cấu trúc thư mục](#3-cấu-trúc-thư-mục) · [Cài đặt](#4-cài-đặt) · [Quy trình làm việc](#5-quy-trình-làm-việc) · [Nhóm thực hiện](#6-nhóm-thực-hiện)

---

## 1. Bài toán

Từ **10 chỉ số** người dùng nhập, hệ thống ước lượng **điểm nguy cơ đột quỵ**, gán nhãn nguy cơ cao/thấp và **giải thích lý do bằng SHAP**.

| | Mô tả |
|---|---|
| **Đầu vào** | `age`, `gender`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status` |
| **Đầu ra** | Điểm nguy cơ + nhãn nguy cơ + đóng góp của từng yếu tố (SHAP) |
| **Dữ liệu** | [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) (Kaggle) |
| **Thách thức chính** | Mất cân bằng lớp nghiêm trọng (~5% ca đột quỵ) → đánh giá bằng Precision / Recall / F1, **không dùng riêng Accuracy** |

**Mục tiêu kỹ thuật bắt buộc**

| Yêu cầu | Chỉ tiêu |
|---|---|
| Độ nhạy y tế | Recall ≥ 85–90% trên lớp nguy cơ đột quỵ |
| Thời gian phản hồi | ≤ 2 giây (bao gồm biểu đồ SHAP) |
| Bảo mật | HTTPS/TLS, mã hóa dữ liệu ở tầng CSDL, ẩn danh hóa dữ liệu dùng để huấn luyện lại |
| Phân quyền | 3 vai trò: Người dùng, Bác sĩ (chỉ xem thống kê/lịch sử), Admin (quản lý tài khoản) |
| Theo dõi | Lưu lịch sử và vẽ biểu đồ biến thiên nguy cơ qua các lần nhập liệu |

---

## 2. Kiến trúc hệ thống 4 khối

```mermaid
flowchart LR
    FE["Khối 1: Frontend<br/>React"] -- "JSON 10 trường" --> BE["Khối 2: Backend<br/>FastAPI"]
    BE -- "DataFrame 1 dòng" --> AI["Khối 3: AI<br/>model_pipeline.pkl + SHAP"]
    AI -- "điểm nguy cơ + SHAP" --> BE
    BE <--> DB[("Khối 4: Database")]
    BE -- "kết quả + giải thích" --> FE
```

| Khối | Công nghệ | Trách nhiệm |
|---|---|---|
| **Frontend** | HTML & JavaScript | Form nhập liệu có validation và chú thích đơn vị, hiển thị kết quả + biểu đồ SHAP, lịch sử/biểu đồ xu hướng, Admin Panel, disclaimer |
| **Backend** | FastAPI | Nhận JSON, xác thực & phân quyền (JWT), gọi khối AI, lưu/đọc CSDL, API thống kê |
| **AI** | scikit-learn Pipeline, SHAP | Tiền xử lý + mô hình đóng gói trong **một** file `.pkl`; cung cấp hàm dự đoán kèm giải thích |
| **Database** | PostgreSQL / MySQL | Tài khoản, hồ sơ, kết quả dự đoán, lịch sử |

**Nguyên tắc tích hợp (quan trọng)**

- Backend chỉ chuyển JSON thành **DataFrame 1 dòng, đúng 10 cột thô** (đúng tên, đúng kiểu) rồi đưa cho khối AI.
- Backend **không** encode/scale/impute và **không** dùng `pd.get_dummies`; mọi biến đổi nằm trong Pipeline (fit một lần lúc huấn luyện).
- Backend **không** gọi `predict()` trực tiếp mà gọi hàm `predict_with_explanation()` do khối AI cung cấp (trả điểm nguy cơ, nhãn, SHAP; ngưỡng phân loại lưu cùng mô hình).
- Mã hóa bảo mật (*encryption*) xử lý ở tầng CSDL và HTTPS; biến đổi đặc trưng (*encoding*) là việc nội bộ của Pipeline.

> Đặc tả đầy đủ: [`info/DataSpecV3.docx`](info/DataSpecV3.docx)

---

## 3. Cấu trúc thư mục

```text
.
├── backend/                 # Mã nguồn phía máy chủ FastAPI
│   └── app/
│       ├── database.py      # Cấu hình SQLite / SQLAlchemy
│       ├── main.py          # Các API Endpoint (Auth, Predict, History)
│       ├── models.py        # Định nghĩa các bảng Database (User, History)
│       ├── schemas.py       # Pydantic validation cho request/response
│       └── security.py      # Mã hóa mật khẩu & tạo JWT Token
├── frontend/                # Giao diện người dùng (Vanilla HTML & JS)
│   ├── DANGKY DANGNHAP.HTML # Giao diện Đăng nhập / Đăng ký
│   ├── TRANG KTRA.HTML      # Giao diện Nhập liệu & Dự đoán AI
│   └── LICH_SU.HTML         # Giao diện Lịch sử & Biểu đồ Chart.js
├── models/
│   └── model_pipeline.pkl   # Mô hình AI đóng gói sẵn
├── notebooks/               # Khám phá và xử lý dữ liệu (EDA)
├── src/                     # Các hàm tiền xử lý dùng chung
├── requirements.txt         # Danh sách thư viện Python
└── README.md
```

| Thư mục | Vai trò | Quy ước |
|---|---|---|
| `data/raw` | Bản gốc của dataset (`healthcare-dataset-stroke-data.csv`), tải bằng `01_eda.py` | **Không sửa tay.** Mọi thay đổi phải qua script để tái lập được |
| `data/processed` | `stroke_clean.csv`: dữ liệu đã bỏ cột `id`, giữ 10 cột đặc trưng + nhãn `stroke` | **Chưa** encode/scale/impute (`bmi` vẫn còn giá trị thiếu). Việc này thuộc Pipeline để chỉ fit trên tập train, tránh rò rỉ dữ liệu (*data leakage*) |
| `src` | Mã nguồn tái sử dụng. `preprocessing.py` định nghĩa `build_preprocessor()`: số (`age`, `avg_glucose_level`, `bmi`) → impute median + `StandardScaler`; phân loại (5 cột) → impute + `OneHotEncoder(handle_unknown="ignore")`; `hypertension`, `heart_disease` → giữ nguyên | Huấn luyện và Backend dùng **cùng một** định nghĩa tiền xử lý |
| `models` | Artifact mô hình mà ứng dụng nạp lúc chạy. Backend chỉ nạp file này | `.pkl` phụ thuộc phiên bản scikit-learn → cài đúng phiên bản trong `requirements.txt` |


Hệ thống cung cấp giao diện Web trực quan bằng HTML/JS thuần giúp người dùng thao tác dễ dàng:
- **Đăng ký / Đăng nhập:** Xác thực người dùng bằng JSON Web Token (JWT).
- **Trang Sàng lọc (`TRANG KTRA.HTML`):** Nhập 10 thông số sức khỏe, gửi yêu cầu qua API FastAPI để nhận kết quả xác suất đột quỵ và biểu đồ giải thích yếu tố SHAP.
- **Trang Lịch sử (`LICH_SU.HTML`):** Truy vấn cơ sở dữ liệu để vẽ biểu đồ biến thiên nguy cơ qua các lần kiểm tra bằng Chart.js.

---

## 4. Cài đặt

### Bước 1: Clone dự án và tạo môi trường ảo
```bash

git clone <repo-url>
cd DuDoanDotQuy

#Ai đã từng chạy dự án có thể nhập lệnh deactivate để thoát khỏi môi trường ảo trước, xong sau đó xóa file môi trường ảo để cài lại và thống nhất requirements mới nhất.
python -m venv venv
venv\Scripts\activate          # Trên Windows (CMD / PowerShell)
# source venv/bin/activate     # Trên macOS / Linux
```

### Bước 2: Cài đặt thư viện phụ thuộc
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Cách chạy dự án

Để ứng dụng chạy mượt mà, bạn cần khởi động **Backend (FastAPI)** trước, sau đó mở **Frontend (HTML/JS)**.

### Bước 1: Khởi động máy chủ Backend
Mở Terminal, di chuyển vào thư mục backend và chạy lệnh Uvicorn:
```bash
cd backend
uvicorn app.main:app --reload
```
*Máy chủ Backend sẽ hoạt động tại địa chỉ: `http://127.0.0.1:8000`*

### Bước 2: Mở giao diện Frontend
Bạn có thể mở giao diện bằng một trong hai cách sau:
- **Cách khuyên dùng (Ổn định nhất):** Dùng Python HTTP Server tích hợp:
  ```bash
  cd frontend
  python -m http.server 5500
  ```
  Sau đó mở trình duyệt và truy cập: `http://localhost:5500/DANGKY DANGNHAP.HTML`
- **Cách truyền thống:** Sử dụng tiện ích Live Server, xong open file `DANGKY DANGNHAP.HTML`  with Live Server 
- **Cách nhanh:** Nhấp đúp trực tiếp vào file `DANGKY DANGNHAP.HTML` trong thư mục `frontend` bằng trình duyệt web (Edge / Chrome).

---

## 5. Quy trình làm việc

- Nhánh `main` luôn chạy được; mỗi người làm trên nhánh riêng: `feature/eda`, `feature/model`, `feature/backend`, `feature/frontend`.
- Mọi thay đổi vào `main` qua **Pull Request** và có ít nhất 1 người review.
- Không commit dữ liệu nhạy cảm, mật khẩu, khóa mã hóa hay file `.env`.

---

## 6. Nhóm thực hiện

| Thành viên | Vai trò |
|---|---|
| Nguyễn Hoàng Hải | Leader, tích hợp hệ thống |
| Vũ Thị Giang | Data Engineer |
| Hoàng Nhật Giang | ML Engineer |
| Trần Xuân Hai | Backend Developer |
| Nguyễn Văn Hoàng | Frontend Developer |

Kế hoạch chi tiết 10 tuần: [`info/Ke_hoach_10_tuan_Chan_doan_Dot_quy V3.docx`](info/Ke_hoach_10_tuan_Chan_doan_Dot_quy%20V3.docx)
