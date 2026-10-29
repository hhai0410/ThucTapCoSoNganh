# Cập nhật sau yêu cầu hoàn thiện demo

Phiên bản demo hoàn chỉnh được mô tả trong README.md; phương pháp và kết quả mô hình trong MODEL_CARD.md. Các phát hiện bên dưới là lịch sử rà soát ban đầu, không còn mô tả đầy đủ trạng thái mới.

Đã bổ sung chọn vai trò Người dùng/Bác sĩ, ba nút đăng nhập demo, dữ liệu mẫu, thiết kế dùng chung cho ba phân hệ, nhập CSV/XLSX, ngưỡng bác sĩ, tạo hồ sơ, thống kê admin, mã hóa tên/khu vực/nội dung lịch sử và artifact XGBoost mới làm mô hình chính. Model mới đạt Recall 90%, AUC 0,8504 trên tập kiểm tra phát triển; không phải xác minh độc lập số liệu trong báo cáo cũ.

Đã kiểm thử API với database tạm, cả CSV và XLSX (lượt cuối thành công, dự đoán khoảng 0,067 giây), và kiểm tra trực tiếp giao diện đăng nhập, chọn role, tạo hồ sơ, sàng lọc + SHAP, dashboard admin. Các giới hạn còn lại: chưa có HTTPS local, email chưa mã hóa, PDF backend không dấu, precision thấp và chưa có kiểm định lâm sàng/ngoài mẫu.

Lưu ý quản lý mã nguồn: backend/app.db vốn đã được Git theo dõi từ phiên bản cũ. Quy tắc .gitignore không tự bỏ theo dõi file đã có trong Git; không đưa database vận hành và các khóa mã hóa vào bản chia sẻ công khai.

---

# Rà soát dự án ngày 30/10/2026

Đã đối chiếu ba tài liệu trong info và BC_TTCSN_FINAL_NHOM_5.docx. Ưu tiên phạm vi báo cáo final và DataSpecV3 khi các tài liệu có mô tả mâu thuẫn (16/21 cột nội bộ, yêu cầu mở rộng so với sản phẩm cuối). Không sửa nội dung báo cáo gốc.

## Lỗi đã sửa

| Phát hiện | Thay đổi |
| --- | --- |
| Predict dùng 0,5 thay ngưỡng 0,18 trong báo cáo | Mô hình hiện tại được gắn decision_threshold_=0.18; API và màu cảnh báo dùng nhãn trả về; mô hình huấn luyện mới dùng ngưỡng riêng lưu cùng artifact |
| Không nạp được model vẫn trả 200 với nguy cơ 0% | Trả HTTP 503; /health phản ánh trạng thái model |
| TreeExplainer khởi tạo lại mỗi lần gọi | Nạp sẵn cùng mô hình; reload thay bộ model/explainer/ngưỡng đồng bộ |
| Không có chức năng bác sĩ/admin dù báo cáo ghi đã Pass | Bổ sung danh sách, tìm kiếm, hồ sơ, lịch sử bệnh nhân; phân quyền endpoint; quản lý tài khoản, cấu hình và reload |
| Đăng ký tự khai role bất kỳ | Công khai chỉ được đăng ký user; admin cấp quyền; thêm công cụ tạo admin local |
| JWT có secret cố định trong mã | Đọc biến môi trường hoặc khóa local ngẫu nhiên; không commit khóa |
| Database phụ thuộc thư mục chạy | Đường dẫn tuyệt đối tới backend/app.db; hỗ trợ DATABASE_URL |
| Thiếu trạng thái khóa và thời điểm tạo mặc định cố định | Bổ sung migration is_active không xóa dữ liệu; default thời gian dùng callable; khóa chặn cả token cũ |
| Admin tự xóa/hạ quyền/khóa | Chặn với 400; xóa tài khoản dọn lịch sử liên quan |
| Validation thiếu email, tên trống, số vô hạn, BMI <10 | Đồng bộ backend và frontend; giới hạn mật khẩu bcrypt 72 byte UTF-8 |
| Lịch sử chỉ giữ vài chỉ số | Bản ghi mới giữ đủ 10 đầu vào, nhãn, ngưỡng và SHAP; bản ghi cũ vẫn đọc được |
| Thời gian lịch sử không có năm và chưa đổi giờ | Hiển thị UTC+7, có năm/giây, sắp xếp ổn định theo timestamp/id |
| Chuyển trang đăng nhập sai tên file | Sửa tên có dấu cách; xử lý hết hạn/khóa; bổ sung trạng thái lịch sử rỗng |
| Giao diện hard-code 50% | Dùng prediction_result từ mô hình; lịch sử mới lưu ngưỡng theo lần đo |
| SHAP dễ bị hiểu thành phần trăm | Ghi đơn vị đầu ra thô, không coi giá trị SHAP là phần trăm nguy cơ |
| Huấn luyện dò ngưỡng trên test | Thay bằng GridSearchCV 5-fold F2 và ngưỡng trên OOF train; preprocess/SMOTE chỉ fit trong fold; test chỉ đánh giá |
| Không lưu số liệu huấn luyện dạng máy đọc | validation_metrics.json chứa tham số, ngưỡng, Recall/Precision/F1/F2/AUC và confusion matrix |
| Launcher có thể giữ model cũ sau tái tạo dữ liệu | Dữ liệu cần làm sạch lại thì model cũng được huấn luyện lại |

## Kiểm chứng

- API integration test: một kịch bản tổng hợp nhiều kiểm tra, chạy bằng unittest trên database tạm. Đã Pass các luồng auth, dữ liệu sai, dự đoán + SHAP, lưu lịch sử, cách ly tài khoản, 401/403, khóa/mở khóa, bảo vệ admin, reload và cấu trúc Excel.
- Lượt đo dự đoán API trong kiểm thử: khoảng 0,082 giây trên máy hiện tại. Đây là một lượt đo local, không phải benchmark tải hay cam kết mọi request.
- pip check: không phát hiện xung đột dependency trong venv hiện tại.
- compileall backend/src/run.py/tests và kiểm tra cú pháp JavaScript nhúng ở tất cả trang: đạt.
- run.py --dry-run: chạy thành công.

## Sai lệch báo cáo và giới hạn cần ghi nhận

1. Đánh giá artifact gốc hiện có bằng classifier trên data/processed/test.npz ở ngưỡng 0,18: Recall=1,0; ROC-AUC=0,810041; confusion matrix [[169,803],[0,50]]. Không trùng bảng báo cáo (AUC 0,8504, Recall 88%, [[538,434],[6,44]]). Không có đủ bằng chứng để xác nhận artifact và tập test lưu sẵn chính là phiên bản dùng cho bảng đó. Không được coi số liệu báo cáo là kết quả kiểm thử code hiện tại.
2. Quy trình huấn luyện mới hợp lệ hơn về tách dữ liệu nhưng kết quả có thể khác báo cáo. Các tham số được chọn trên train; OOF dùng để chọn ngưỡng không phải nested-CV ước lượng tổng quát hóa. Chỉ số cuối trên test mới là đánh giá holdout; không sửa ngưỡng theo test để ép Recall đạt mục tiêu.
3. SQLite hiện chưa mã hóa dữ liệu tại rest và local HTTP chưa có TLS. JWT/bcrypt không đồng nghĩa mã hóa bệnh án; không tuyên bố đạt HIPAA hay đã triển khai bảo mật y tế đầy đủ. Báo cáo cũng nêu nâng cấp mã hóa/cloud là hướng phát triển.
4. Xuất Excel qua backend; PDF qua /doctor/export?format=pdf (chữ được chuyển không dấu để tương thích font chuẩn); có thêm In / Lưu PDF ở trình duyệt để giữ tiếng Việt. Chưa kiểm tra bố cục in trên trình duyệt.
5. Biểu đồ lịch sử cá nhân vẫn dùng Chart.js CDN; bảng vẫn hoạt động khi không nạp được thư viện. Chưa kiểm thử trình duyệt tự động/toàn bộ layout, tải đồng thời, hay triển khai HTTPS.
6. Bản ghi lịch sử cũ không có đủ đầu vào/SHAP để phục hồi; không tự tạo dữ liệu giả. is_active được thêm tự động cho SQLite cũ khi khởi động.
7. Tính năng tải CSV hàng loạt và chỉnh ngưỡng thủ công theo bác sĩ trong đề cương ban đầu không nằm trong danh sách API final; chưa bổ sung vào phiên bản này.
8. Không có cam kết “không còn mọi lỗi”; kết luận chỉ áp dụng cho những kiểm tra đã chạy và bằng chứng nêu trên.

Huấn luyện kiểm tra riêng đã hoàn tất cả 5 thuật toán. Holdout Recall: Logistic Regression 84%, Decision Tree 84%, Random Forest 82%, XGBoost 82%, LightGBM 84%. Chưa đạt mục tiêu holdout 85–90%; không hạ ngưỡng dựa trên test. Mô hình đang phục vụ vẫn là artifact LightGBM cũ cùng ngưỡng báo cáo 0,18.

Kết quả huấn luyện mới đầy đủ: outputs/validated/validation_metrics.json. Trong nhóm mô hình cây, XGBoost được chọn bằng OOF ROC-AUC trên train; ngưỡng 0,363178; holdout Recall 82%, AUC 0,831821. Đã kiểm tra predict_proba và TreeExplainer của artifact mới. Không thay model đang phục vụ.
