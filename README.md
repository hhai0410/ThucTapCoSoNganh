<div align="center">

# 🧠 Stroke Risk Screening System

**Hệ thống hỗ trợ sàng lọc sớm nguy cơ đột quỵ dựa trên dữ liệu y tế bằng Machine Learning**

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-Frontend-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)
![HTML5](https://img.shields.io/badge/HTML5-Structure-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-Styling-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![Chart.js](https://img.shields.io/badge/Chart.js-Visualization-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white)
![Status](https://img.shields.io/badge/Status-In_Development-F1C40F?style=for-the-badge)

</div>

> ⚠️ **Tuyên bố miễn trừ trách nhiệm (Medical Disclaimer)**
> Hệ thống chỉ mang tính chất tham khảo và hỗ trợ sàng lọc, không thay thế hoàn toàn chỉ định chẩn đoán chính thức từ Bác sĩ chuyên khoa.
> Dự án phục vụ mục đích học tập (đồ án Thực tập cơ sở ngành), **chưa được kiểm chứng lâm sàng** và không dùng để ra quyết định điều trị.

**Mục lục:** [Chạy nhanh](#chạy-nhanh-1-lệnh) · [Bài toán](#1-bài-toán) · [Kiến trúc](#2-kiến-trúc-hệ-thống-4-khối) · [Cấu trúc thư mục](#3-cấu-trúc-thư-mục) · [Giao diện](#4-giao-diện-và-tính-năng) · [Chạy thủ công](#5-cài-đặt-và-chạy-thủ-công) · [Xử lý sự cố](#6-xử-lý-sự-cố) · [Quy trình làm việc](#7-quy-trình-làm-việc) · [Nhóm](#8-nhóm-thực-hiện)

---

## Chạy nhanh (1 lệnh)

**Yêu cầu:** [Git](https://git-scm.com/), [Python](https://www.python.org/downloads/) 3.13 (tối thiểu 3.10; khi cài trên Windows nhớ tick **Add Python to PATH**) và kết nối Internet ở lần chạy đầu.

```bash
git clone <repo-url>
cd DuDoanDotQuy
python run.py
```

- **Windows:** có thể bấm đúp vào file `run.bat` thay cho lệnh `python run.py` (nếu máy không nhận `python`, thử `py run.py`).
- **macOS / Linux:** dùng `python3 run.py`.

Sau khi chạy xong, trình duyệt tự mở trang đăng nhập. Nhấn `Ctrl + C` trong terminal để tắt dự án.

**`run.py` tự làm lần lượt các bước sau** (bước nào đã xong ở lần trước sẽ được bỏ qua nên các lần chạy sau rất nhanh):

| Bước | Việc làm | Bỏ qua khi |
|---|---|---|
| 1 | Tạo môi trường ảo `venv/` | Đã có `venv/` |
| 2 | Cài thư viện từ `requirements.txt` | `requirements.txt` không đổi từ lần cài trước |
| 3 | Tải dữ liệu gốc (`notebooks/01_eda.py`) rồi làm sạch (`notebooks/03_eda.py`) | Đã có `data/raw/...csv` và `data/processed/stroke_clean.csv` |
| 4 | Chia train/test + SMOTE (`src/week3.py`), huấn luyện và lưu `models/model_pipeline.pkl` (`src/week4.py`) | Đã có `models/model_pipeline.pkl` |
| 5 | Bật Backend (cổng `8000`) và Frontend (cổng `5500`), mở trình duyệt | Không bao giờ bỏ qua (trừ khi dùng `--setup-only`) |

> Lần chạy đầu tiên mất vài phút (cài thư viện, tải dữ liệu, huấn luyện mô hình).
> `02_eda.py` chỉ vẽ biểu đồ khám phá và mở cửa sổ biểu đồ nên **không** chạy tự động, dùng `--eda` nếu muốn xem.

**Các tùy chọn thêm:**

| Lệnh | Tác dụng |
|---|---|
| `python run.py --dry-run` | Chỉ in ra kế hoạch sẽ làm gì, chưa làm gì cả |
| `python run.py --retrain` | Huấn luyện lại mô hình (chạy lại `week3.py` và `week4.py`) |
| `python run.py --refresh-data` | Tải lại dữ liệu gốc, làm sạch lại và huấn luyện lại |
| `python run.py --eda` | Chạy thêm `02_eda.py` để xem biểu đồ khám phá dữ liệu |
| `python run.py --setup-only` | Chỉ chuẩn bị (cài đặt, dữ liệu, huấn luyện), không bật máy chủ |
| `python run.py --no-browser` | Không tự mở trình duyệt |
| `python run.py --reload` | Backend tự khởi động lại khi sửa code (dùng khi phát triển) |

**Sau khi chạy, truy cập:**

| Thành phần | Địa chỉ |
|---|---|
| Ứng dụng web | http://localhost:5500/DANGKY%20DANGNHAP.HTML |
| Tài liệu API (Swagger) | http://127.0.0.1:8000/docs |

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
    FE["Khối 1: Frontend<br/>HTML + JavaScript"] -- "JSON 10 trường" --> BE["Khối 2: Backend<br/>FastAPI"]
    BE -- "DataFrame 1 dòng" --> AI["Khối 3: AI<br/>model_pipeline.pkl + SHAP"]
    AI -- "điểm nguy cơ + SHAP" --> BE
    BE <--> DB[("Khối 4: Database<br/>SQLite")]
    BE -- "kết quả + giải thích" --> FE
```

| Khối | Công nghệ | Trách nhiệm |
|---|---|---|
| **Frontend** | HTML, CSS, JavaScript thuần, Chart.js | Form nhập liệu có validation và chú thích đơn vị, hiển thị kết quả + biểu đồ SHAP, lịch sử/biểu đồ xu hướng, disclaimer |
| **Backend** | FastAPI, SQLAlchemy | Nhận JSON, xác thực & phân quyền (JWT), gọi khối AI, lưu/đọc CSDL |
| **AI** | scikit-learn Pipeline, SHAP | Tiền xử lý + mô hình đóng gói trong **một** file `.pkl`; cung cấp hàm dự đoán kèm giải thích |
| **Database** | SQLite (qua SQLAlchemy) | Tài khoản, kết quả dự đoán, lịch sử. Dự kiến nâng cấp lên PostgreSQL/MySQL khi triển khai thật |

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
├── run.py                   # ⭐ Chạy cả dự án bằng 1 lệnh (cài đặt → dữ liệu → huấn luyện → máy chủ)
├── run.bat                  # Bấm đúp để chạy trên Windows (gọi run.py)
├── requirements.txt         # Danh sách thư viện Python
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
├── src/
│   ├── preprocessing.py     # build_preprocessor(): ColumnTransformer dùng chung
│   ├── week3.py             # Chia train/test + cân bằng lớp SMOTE
│   └── week4.py             # Huấn luyện pipeline, lưu models/model_pipeline.pkl
├── notebooks/               # Khám phá và xử lý dữ liệu (01_eda, 02_eda, 03_eda)
├── data/                    # (tự tạo khi chạy) raw/ và processed/
├── models/                  # model_pipeline.pkl - mô hình AI đóng gói sẵn
├── outputs/                 # Biểu đồ, log, kết quả trung gian
├── info/                    # Tài liệu: Data Spec, đề tài, kế hoạch 10 tuần
└── README.md
```

| Thư mục | Vai trò | Quy ước |
|---|---|---|
| `data/raw` | Bản gốc của dataset (`healthcare-dataset-stroke-data.csv`), tải bằng `01_eda.py` | **Không sửa tay.** Mọi thay đổi phải qua script để tái lập được |
| `data/processed` | `stroke_clean.csv`: dữ liệu đã bỏ cột `id`, giữ 10 cột đặc trưng + nhãn `stroke` | **Chưa** encode/scale/impute (`bmi` vẫn còn giá trị thiếu). Việc này thuộc Pipeline để chỉ fit trên tập train, tránh rò rỉ dữ liệu (*data leakage*) |
| `src` | Mã nguồn tái sử dụng. `preprocessing.py` định nghĩa `build_preprocessor()`: số (`age`, `avg_glucose_level`, `bmi`) → impute median + `StandardScaler`; phân loại (5 cột) → impute + `OneHotEncoder(handle_unknown="ignore")`; `hypertension`, `heart_disease` → giữ nguyên | Huấn luyện và Backend dùng **cùng một** định nghĩa tiền xử lý |
| `models` | Artifact mô hình mà ứng dụng nạp lúc chạy. Backend chỉ nạp file này | `.pkl` phụ thuộc phiên bản scikit-learn → cài đúng phiên bản trong `requirements.txt`. Nếu `.pkl` không được commit lên Git, lần chạy đầu `run.py` sẽ tự huấn luyện ra |

> Các thư mục/tệp sinh ra khi chạy (`venv/`, `data/`, `__pycache__/`, file CSDL `*.db`) không nên commit lên Git, hãy khai báo trong `.gitignore`.

---

## 4. Giao diện và tính năng

Hệ thống cung cấp giao diện Web bằng HTML/JS thuần giúp người dùng thao tác dễ dàng:

- **Đăng ký / Đăng nhập:** xác thực người dùng bằng JSON Web Token (JWT).
- **Trang Sàng lọc (`TRANG KTRA.HTML`):** nhập 10 thông số sức khỏe, gửi yêu cầu qua API FastAPI để nhận **điểm nguy cơ** và biểu đồ giải thích các yếu tố SHAP.
- **Trang Lịch sử (`LICH_SU.HTML`):** truy vấn cơ sở dữ liệu để vẽ biểu đồ biến thiên nguy cơ qua các lần kiểm tra bằng Chart.js.

---

## 5. Cài đặt và chạy thủ công

Chỉ cần khi bạn muốn tự chạy từng bước thay vì dùng `python run.py`.

<details>
<summary><b>Bấm để xem các bước thủ công</b></summary>

### Bước 1: Tạo môi trường ảo và cài thư viện

```bash
git clone <repo-url>
cd DuDoanDotQuy

python -m venv venv
venv\Scripts\activate          # Windows (CMD / PowerShell)
# source venv/bin/activate     # macOS / Linux

python -m pip install --upgrade pip
pip install -r requirements.txt
```

*(Tùy chọn)* Chọn interpreter trong VS Code: `Ctrl + Shift + P` → **Python: Select Interpreter** → chọn `./venv/Scripts/python.exe`.

### Bước 2: Tạo `data/raw` và `data/processed`

Chạy từ **thư mục gốc** của dự án (script dùng đường dẫn tương đối):

```bash
python notebooks/01_eda.py   # Tải dataset gốc vào data/raw/
python notebooks/02_eda.py   # (Tùy chọn) Khám phá, vẽ biểu đồ
python notebooks/03_eda.py   # Làm sạch, xuất ra data/processed/
```

### Bước 3: Chia tập dữ liệu và cân bằng lớp (SMOTE)

```bash
python src/week3.py
```

### Bước 4: Huấn luyện mô hình

```bash
python src/week4.py          # Lưu mô hình vào models/model_pipeline.pkl
```

### Bước 5: Khởi động Backend (FastAPI)

```bash
cd backend
uvicorn app.main:app --reload
```

Backend chạy tại `http://127.0.0.1:8000` (tài liệu API: `/docs`).

### Bước 6: Mở Frontend

Mở một terminal khác (đã kích hoạt venv):

```bash
cd frontend
python -m http.server 5500
```

Truy cập `http://localhost:5500/DANGKY%20DANGNHAP.HTML`.

Cách khác: dùng tiện ích **Live Server** của VS Code, chuột phải vào `DANGKY DANGNHAP.HTML` → *Open with Live Server*.

> Không nên mở file HTML bằng cách nhấp đúp (địa chỉ `file://`): trình duyệt có thể chặn các lời gọi API tới Backend.

</details>

---

## 6. Xử lý sự cố

| Triệu chứng | Cách xử lý |
|---|---|
| `python` không được nhận diện | Cài Python và tick **Add Python to PATH**, hoặc dùng `py run.py` (Windows) / `python3 run.py` (macOS, Linux) |
| Báo cổng `8000` hoặc `5500` đang bị chiếm | Dự án đang chạy ở terminal khác: tắt bằng `Ctrl + C`. Hoặc tìm tiến trình: `netstat -ano \| findstr :8000` rồi `taskkill /PID <số PID> /F` (Windows) |
| Lỗi tải dữ liệu từ Kaggle | Tải thủ công `healthcare-dataset-stroke-data.csv` từ trang dataset, đặt vào `data/raw/` rồi chạy lại `python run.py` |
| `run.py` báo thiếu thư viện | Thêm thư viện được nêu vào `requirements.txt` (mỗi dòng một thư viện) rồi chạy lại |
| Cảnh báo/lỗi phiên bản scikit-learn khi nạp `model_pipeline.pkl` | Huấn luyện lại bằng đúng thư viện đang cài: `python run.py --retrain` |
| Trang web không gọi được API | Kiểm tra Backend đang chạy ở cổng `8000` và bạn mở web qua `http://localhost:5500/...` (không phải `file://`) |
| Muốn làm lại từ đầu | Xóa `venv/`, `data/`, `models/model_pipeline.pkl` rồi chạy lại `python run.py` |

---

## 7. Quy trình làm việc

- Nhánh `main` luôn chạy được; mỗi người làm trên nhánh riêng: `feature/eda`, `feature/model`, `feature/backend`, `feature/frontend`.
- Mọi thay đổi vào `main` qua **Pull Request** và có ít nhất 1 người review.
- Không commit dữ liệu nhạy cảm, mật khẩu, khóa mã hóa hay file `.env`.
- `run.py` là điểm vào duy nhất để chạy dự án. Khi thêm bước xử lý mới hoặc đổi tên file/cổng, sửa phần **CẤU HÌNH** ở đầu `run.py`; khi thêm thư viện mới, thêm vào `requirements.txt`.

---

## 8. Nhóm thực hiện

| Thành viên | Vai trò |
|---|---|
| Nguyễn Hoàng Hải | Leader, tích hợp hệ thống |
| Vũ Thị Giang | Data Engineer |
| Hoàng Nhật Giang | ML Engineer |
| Trần Xuân Hai | Backend Developer |
| Nguyễn Văn Hoàng | Frontend Developer |

Kế hoạch chi tiết 10 tuần: [`info/Ke_hoach_10_tuan_Chan_doan_Dot_quy V3.docx`](info/Ke_hoach_10_tuan_Chan_doan_Dot_quy%20V3.docx)
