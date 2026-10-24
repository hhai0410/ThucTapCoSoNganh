#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py - CHẠY TOÀN BỘ DỰ ÁN BẰNG MỘT LỆNH
==========================================

    python run.py            (Windows có thể bấm đúp vào run.bat)

Script tự làm lần lượt (bước nào đã xong từ lần trước thì bỏ qua):
    1. Tạo môi trường ảo ./venv
    2. Cài thư viện từ requirements.txt (chỉ cài lại khi file này thay đổi)
    3. Chuẩn bị dữ liệu : notebooks/01_eda.py -> notebooks/03_eda.py
    4. Huấn luyện model : src/week3.py -> src/week5.py -> outputs/models/best_model_week5.pkl
    5. Bật Backend (FastAPI, cổng 8000) và Frontend (cổng 5500), mở trình duyệt

Chỉ dùng thư viện chuẩn của Python nên chạy được ngay cả khi chưa cài gì.
Nhấn Ctrl+C để tắt cả hai máy chủ.

Ví dụ:
    python run.py --dry-run        # xem script SẼ làm gì, chưa làm gì cả
    python run.py --retrain        # huấn luyện lại model
    python run.py --refresh-data   # tải lại dữ liệu gốc rồi làm lại tất cả
    python run.py --setup-only     # chỉ chuẩn bị, không bật máy chủ
"""

import argparse
import hashlib
import os
import signal
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.parse import quote

# ============================================================
# CẤU HÌNH - đổi tên file / cổng thì chỉ cần sửa ở đây
# ============================================================
ROOT = Path(__file__).resolve().parent          # thư mục gốc của repo
VENV_DIR = ROOT / "venv"
REQUIREMENTS = ROOT / "requirements.txt"

# Các file "đầu ra": nếu đã tồn tại thì bước tạo ra nó được bỏ qua
RAW_CSV = ROOT / "data" / "raw" / "healthcare-dataset-stroke-data.csv"
CLEAN_CSV = ROOT / "data" / "processed" / "stroke_clean.csv"
MODEL_PKL = ROOT / "outputs" / "models" / "best_model_week5.pkl"   # model cuối do week5 lưu

SCRIPT_DOWNLOAD = ROOT / "notebooks" / "01_eda.py"   # tải dữ liệu gốc
SCRIPT_EDA = ROOT / "notebooks" / "02_eda.py"        # vẽ biểu đồ (chỉ chạy khi có --eda)
SCRIPT_CLEAN = ROOT / "notebooks" / "03_eda.py"      # làm sạch dữ liệu
SCRIPT_WEEK3 = ROOT / "src" / "week3.py"             # chia train/test + SMOTE (tạo data/processed/*.npz)
# Script huấn luyện + so sánh mô hình, lưu model vào outputs/models/.
# Ưu tiên src/week5.py; nếu repo đặt tên src/week5_full.py thì dùng tên đó.
SCRIPT_TRAIN_CANDIDATES = [ROOT / "src" / "week5.py", ROOT / "src" / "week5_full.py"]

BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"
BACKEND_APP = "app.main:app"        # <tên file>:<biến FastAPI> trong backend/
BACKEND_PORT = 8000
FRONTEND_PORT = 5500
LOGIN_PAGE = "DANGKY DANGNHAP.HTML"  # trang mở đầu tiên trong thư mục frontend/

# Thư viện bắt buộc (tên khi import, tên khi pip install).
# Thiếu thì script báo rõ thiếu gì, thay vì để lỗi khó hiểu xuất hiện giữa chừng.
REQUIRED_SERVE = [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("sqlalchemy", "sqlalchemy"),
    ("sklearn", "scikit-learn"),
    ("pandas", "pandas"),
]
REQUIRED_TRAIN = [
    ("imblearn", "imbalanced-learn"),
    ("kagglehub", "kagglehub"),
    ("xgboost", "xgboost"),
    ("lightgbm", "lightgbm"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
]


# ============================================================
# Hàm tiện ích
# ============================================================
def say(msg=""):
    print(msg, flush=True)


def step(number, total, title):
    say()
    say(f"[{number}/{total}] {title}")
    say("-" * 60)


def fail(msg, code=1):
    say()
    say("LỖI: " + msg)
    sys.exit(code)


def venv_python():
    """Đường dẫn python bên trong venv (khác nhau giữa Windows và macOS/Linux)."""
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def make_env(show_plots=False):
    """Biến môi trường cho các tiến trình con."""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"                 # in tiếng Việt không bị lỗi font trên Windows
    env["PYTHONIOENCODING"] = "utf-8"
    if not show_plots:
        # Agg = vẽ biểu đồ "ngầm", không mở cửa sổ -> script không bị treo chờ đóng cửa sổ
        env["MPLBACKEND"] = "Agg"
    return env


def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def requirements_hash():
    return hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()


def train_script():
    """Chọn script huấn luyện: week5.py, hoặc week5_full.py nếu repo dùng tên đó."""
    for candidate in SCRIPT_TRAIN_CANDIDATES:
        if candidate.exists():
            return candidate
    return SCRIPT_TRAIN_CANDIDATES[0]       # không có file nào -> run_script sẽ báo lỗi rõ ràng


def backend_command(py, reload):
    cmd = [str(py), "-m", "uvicorn", BACKEND_APP, "--port", str(BACKEND_PORT)]
    if reload:
        cmd.append("--reload")
    return cmd


def frontend_command(py):
    return [str(py), "-m", "http.server", str(FRONTEND_PORT)]


# ============================================================
# Lập kế hoạch: bước nào cần chạy?
# ============================================================
def make_plan(args):
    """Trả về dict mô tả việc cần làm, dùng chung cho chế độ chạy thật và --dry-run."""
    need_raw = args.refresh_data or not RAW_CSV.exists()
    need_clean = need_raw or not CLEAN_CSV.exists()
    need_model = args.retrain or args.refresh_data or not MODEL_PKL.exists()
    return {
        "create_venv": not venv_python().exists(),
        "need_raw": need_raw,
        "need_clean": need_clean,
        "need_model": need_model,
        "run_eda": args.eda,
    }


def print_plan(plan, args):
    def yn(flag, why_yes, why_no):
        return ("SẼ CHẠY  : " + why_yes) if flag else ("BỎ QUA   : " + why_no)

    say("Kế hoạch (chưa thực hiện gì):")
    say("  " + yn(plan["create_venv"], "tạo môi trường ảo ./venv", "đã có ./venv"))
    say("  " + yn(True, "cài/kiểm tra thư viện từ requirements.txt (bỏ qua nếu file không đổi)", ""))
    say("  " + yn(plan["need_raw"], "01_eda.py - tải dữ liệu gốc", "đã có data/raw/..."))
    say("  " + yn(plan["run_eda"], "02_eda.py - vẽ biểu đồ khám phá", "chỉ chạy khi thêm --eda"))
    say("  " + yn(plan["need_clean"], "03_eda.py - làm sạch dữ liệu", "đã có data/processed/stroke_clean.csv"))
    say("  " + yn(plan["need_model"], f"week3.py + {train_script().name} - huấn luyện model", "đã có outputs/models/best_model_week5.pkl"))
    if args.setup_only:
        say("  BỎ QUA   : bật máy chủ (do --setup-only)")
    else:
        say(f"  SẼ CHẠY  : Backend  http://127.0.0.1:{BACKEND_PORT}  (API docs: /docs)")
        say(f"  SẼ CHẠY  : Frontend http://localhost:{FRONTEND_PORT}")


# ============================================================
# Các bước thực thi
# ============================================================
def check_python():
    if sys.version_info < (3, 10):
        fail(f"Cần Python 3.10 trở lên (đang dùng {sys.version.split()[0]}). Nhóm dùng Python 3.13.")
    if sys.version_info[:2] != (3, 13):
        say(f"Lưu ý: nhóm phát triển bằng Python 3.13, máy bạn đang dùng {sys.version.split()[0]}.")
        say("       Thường vẫn chạy được; nếu cài thư viện bị lỗi, hãy cài Python 3.13.")


def ensure_venv():
    py = venv_python()
    if py.exists():
        say("Đã có môi trường ảo ./venv - bỏ qua.")
        return py
    say("Đang tạo môi trường ảo ./venv ...")
    result = subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)])
    if result.returncode != 0 or not py.exists():
        fail("Không tạo được môi trường ảo.\n"
             "  Trên Ubuntu/Debian hãy chạy: sudo apt install python3-venv\n"
             "  Rồi xóa thư mục venv (nếu có) và chạy lại: python run.py")
    # Nâng cấp pip cho đỡ cảnh báo; thất bại (mất mạng...) thì bỏ qua, không sao.
    subprocess.run([str(py), "-m", "pip", "install", "--upgrade", "pip",
                    "--quiet", "--disable-pip-version-check", "--retries", "1", "--timeout", "10"])
    return py


def install_requirements(py):
    if not REQUIREMENTS.exists():
        fail("Không thấy requirements.txt ở thư mục gốc của dự án.")
    stamp = VENV_DIR / ".requirements.sha256"
    current = requirements_hash()
    if stamp.exists() and stamp.read_text().strip() == current:
        say("Thư viện đã được cài (requirements.txt không đổi) - bỏ qua.")
        return
    say("Đang cài thư viện từ requirements.txt (lần đầu có thể mất vài phút) ...")
    result = subprocess.run([str(py), "-m", "pip", "install", "-r", str(REQUIREMENTS),
                             "--disable-pip-version-check"])
    if result.returncode != 0:
        fail("Cài thư viện thất bại. Kiểm tra kết nối Internet và nội dung requirements.txt,\n"
             "  rồi chạy lại: python run.py")
    stamp.write_text(current)


def check_modules(py, modules, purpose):
    """Kiểm tra các thư viện cần thiết đã nằm trong venv chưa."""
    code = ("import importlib.util, sys\n"
            "print(','.join(m for m in sys.argv[1:] if importlib.util.find_spec(m) is None))")
    result = subprocess.run([str(py), "-c", code] + [m for m, _ in modules],
                            capture_output=True, text=True)
    missing = [m for m in result.stdout.strip().split(",") if m]
    if missing:
        pip_names = [pip_name for m, pip_name in modules if m in missing]
        fail(f"Thiếu thư viện cho phần {purpose}: {', '.join(pip_names)}\n"
             "  Hãy thêm vào requirements.txt (mỗi thư viện một dòng) rồi chạy lại: python run.py")
    say(f"Đủ thư viện cho phần {purpose}.")


def run_script(py, script, label, show_plots=False, hint=None):
    if not script.exists():
        fail(f"Không tìm thấy {script.relative_to(ROOT)} (bước: {label}).")
    say(f"Đang chạy {script.relative_to(ROOT)} ...")
    result = subprocess.run([str(py), str(script)], cwd=ROOT, env=make_env(show_plots))
    if result.returncode != 0:
        msg = f"{script.relative_to(ROOT)} bị lỗi (xem thông báo phía trên)."
        if hint:
            msg += "\n" + hint
        fail(msg)


def prepare_data(py, plan, total):
    step(3, total, "Chuẩn bị dữ liệu")
    if plan["need_raw"]:
        run_script(py, SCRIPT_DOWNLOAD, "tải dữ liệu gốc", hint=(
            "  Nếu lỗi do tải từ Kaggle, hãy tải thủ công file healthcare-dataset-stroke-data.csv tại:\n"
            "  https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset\n"
            "  rồi đặt vào thư mục data/raw/ và chạy lại: python run.py"))
    else:
        say("Đã có data/raw/healthcare-dataset-stroke-data.csv - bỏ qua bước tải.")
    if plan["run_eda"]:
        run_script(py, SCRIPT_EDA, "khám phá dữ liệu", show_plots=True)
    if plan["need_clean"]:
        run_script(py, SCRIPT_CLEAN, "làm sạch dữ liệu")
    else:
        say("Đã có data/processed/stroke_clean.csv - bỏ qua bước làm sạch.")


def train_model(py, plan, total):
    step(4, total, "Huấn luyện mô hình")
    if plan["need_model"]:
        run_script(py, SCRIPT_WEEK3, "chia train/test + SMOTE")
        run_script(py, train_script(), "huấn luyện và so sánh mô hình")
    else:
        say("Đã có outputs/models/best_model_week5.pkl - bỏ qua huấn luyện (thêm --retrain để huấn luyện lại).")
    if not MODEL_PKL.exists():
        fail("Chạy xong các bước huấn luyện nhưng không thấy outputs/models/best_model_week5.pkl.\n"
             f"  Kiểm tra {train_script().name} có lưu model đúng đường dẫn này không.")


def stop_all(procs):
    for _, p in procs:
        if p.poll() is None:
            p.terminate()
    deadline = time.time() + 5
    for _, p in procs:
        try:
            p.wait(timeout=max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            p.kill()


def wait_for_port(port, proc, timeout):
    """Chờ tới khi cổng mở (máy chủ sẵn sàng). Trả về False nếu tiến trình chết hoặc quá hạn."""
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            return False
        if port_in_use(port):
            return True
        time.sleep(0.5)
    return False


def start_servers(py, args, total):
    step(5, total, "Khởi động máy chủ")
    if not (BACKEND_DIR / "app").exists():
        fail("Không thấy thư mục backend/app.")
    if not FRONTEND_DIR.exists():
        fail("Không thấy thư mục frontend.")
    for name, port in (("Backend", BACKEND_PORT), ("Frontend", FRONTEND_PORT)):
        if port_in_use(port):
            fail(f"Cổng {port} ({name}) đang bị chiếm - có thể dự án đang chạy ở một cửa sổ terminal khác.\n"
                 "  Hãy tắt cửa sổ đó (Ctrl+C) rồi chạy lại.")

    def _terminate(signum, frame):       # đóng cửa sổ terminal / kill cũng tắt gọn hai máy chủ
        raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGTERM, _terminate)
    except (ValueError, OSError):
        pass

    env = make_env()
    procs = []
    try:
        backend = subprocess.Popen(backend_command(py, args.reload), cwd=BACKEND_DIR, env=env)
        procs.append(("Backend", backend))
        say("Đang chờ Backend khởi động (nạp model có thể mất vài giây) ...")
        if not wait_for_port(BACKEND_PORT, backend, timeout=90):
            stop_all(procs)
            fail("Backend không khởi động được (xem thông báo lỗi phía trên).")

        frontend = subprocess.Popen(frontend_command(py), cwd=FRONTEND_DIR, env=env)
        procs.append(("Frontend", frontend))
        if not wait_for_port(FRONTEND_PORT, frontend, timeout=15):
            stop_all(procs)
            fail("Frontend không khởi động được (xem thông báo lỗi phía trên).")

        url = f"http://localhost:{FRONTEND_PORT}/{quote(LOGIN_PAGE)}"
        say()
        say("=" * 60)
        say("  DỰ ÁN ĐANG CHẠY")
        say(f"  Ứng dụng : {url}")
        say(f"  API docs : http://127.0.0.1:{BACKEND_PORT}/docs")
        say("  Nhấn Ctrl+C để tắt.")
        say("=" * 60)
        if not args.no_browser:
            webbrowser.open(url)

        while True:                          # giữ script sống, theo dõi 2 máy chủ
            time.sleep(1)
            for name, p in procs:
                if p.poll() is not None:
                    say(f"\n{name} đã dừng (mã thoát {p.returncode}). Đang tắt phần còn lại ...")
                    return
    except KeyboardInterrupt:
        say("\nĐang tắt các máy chủ ...")
    finally:
        stop_all(procs)
        say("Đã tắt.")


# ============================================================
# Chương trình chính
# ============================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Chạy toàn bộ dự án (cài đặt + dữ liệu + huấn luyện + máy chủ) bằng một lệnh.")
    p.add_argument("--retrain", action="store_true", help="huấn luyện lại model (chạy lại week3 + week5)")
    p.add_argument("--refresh-data", action="store_true",
                   help="tải lại dữ liệu gốc, làm sạch lại và huấn luyện lại")
    p.add_argument("--eda", action="store_true", help="chạy thêm 02_eda.py (mở cửa sổ biểu đồ)")
    p.add_argument("--setup-only", action="store_true", help="chỉ chuẩn bị, không bật máy chủ")
    p.add_argument("--no-browser", action="store_true", help="không tự mở trình duyệt")
    p.add_argument("--reload", action="store_true", help="Backend tự khởi động lại khi sửa code (dùng khi phát triển)")
    p.add_argument("--dry-run", action="store_true", help="chỉ in kế hoạch, không làm gì cả")
    return p.parse_args(argv)


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):      # in tiếng Việt được trên console Windows cũ
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    args = parse_args(argv)
    os.chdir(ROOT)

    say("Stroke Risk Screening System - chạy dự án bằng một lệnh")
    plan = make_plan(args)
    if args.dry_run:
        print_plan(plan, args)
        return 0

    total = 4 if args.setup_only else 5
    check_python()

    step(1, total, "Môi trường ảo (venv)")
    py = ensure_venv()

    step(2, total, "Thư viện (requirements.txt)")
    install_requirements(py)
    check_modules(py, REQUIRED_SERVE, "chạy ứng dụng")
    if plan["need_raw"] or plan["need_clean"] or plan["need_model"]:
        check_modules(py, REQUIRED_TRAIN, "chuẩn bị dữ liệu và huấn luyện")

    prepare_data(py, plan, total)
    train_model(py, plan, total)

    if args.setup_only:
        say()
        say("Chuẩn bị xong. Chạy 'python run.py' để bật dự án.")
        return 0
    start_servers(py, args, total)
    return 0


if __name__ == "__main__":
    sys.exit(main())