"""Cấu hình trung tâm của hệ thống.

Mọi hằng số dùng chung (đường dẫn, phiên bản, random_state, giới hạn) đặt tại đây
để các module khác import, tránh "hard-code" rải rác trong mã nguồn.
API Key KHÔNG bao giờ được ghi ở đây: chỉ đọc từ biến môi trường / file .env.
"""
from __future__ import annotations

import os
from pathlib import Path

try:  # python-dotenv là tùy chọn khi chạy test
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

ROOT_DIR = Path(__file__).resolve().parents[2]
if load_dotenv is not None:
    load_dotenv(ROOT_DIR / ".env", override=False)

# ---------------------------------------------------------------- Thông tin hệ thống
APP_NAME = "ITJobTrend"
APP_SHORT = "ITJobTrend"
APP_VERSION = "2.0.0"
PROJECT_TITLE_VI = (
    "Hệ thống phân tích, khai phá và dự báo xu hướng tuyển dụng ngành CNTT "
    "từ dữ liệu tin đăng việc làm tích hợp trí tuệ nhân tạo"
)
PROJECT_TITLE_EN = "Intelligent IT Recruitment Analytics and Forecasting System with AI Assistant"
COURSE = "Chuyên đề tốt nghiệp 2 - Chuyên ngành Khoa học dữ liệu"

# ---------------------------------------------------------------- Đường dẫn
STORAGE_DIR = Path(os.getenv("APP_STORAGE_DIR", ROOT_DIR / "storage"))
MODELS_DIR = STORAGE_DIR / "models"
EXPORT_DIR = STORAGE_DIR / "exports"
LOG_DIR = STORAGE_DIR / "logs"
DB_PATH = STORAGE_DIR / "app.db"
ASSETS_DIR = ROOT_DIR / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"
SAMPLE_DATA = ROOT_DIR / "data" / "vn" / "sample_it_jobs.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

for _d in (STORAGE_DIR, MODELS_DIR, EXPORT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- Thực nghiệm
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
MAX_TRAIN_ROWS = 60_000          # giới hạn số dòng huấn luyện để giao diện phản hồi nhanh
MAX_CLUSTER_ROWS = 50_000
SILHOUETTE_SAMPLE = 5_000
PREVIEW_ROWS = 20

# ---------------------------------------------------------------- Dự báo
MIN_POINTS = {"naive": 3, "moving_average": 4, "linear_trend": 4, "holt": 8, "arima": 20, "sarima": 0}
SARIMA_MIN_SEASONS = 3            # SARIMA cần >= 3 chu kỳ mùa vụ đầy đủ

# ---------------------------------------------------------------- Miền tuyển dụng
USD_TO_VND = float(os.getenv("USD_TO_VND", "25000"))   # tỷ giá quy đổi (giả định, có thể sửa trong .env)
SALARY_MIN_PLAUSIBLE = 0.5       # triệu VND / tháng
SALARY_MAX_PLAUSIBLE = 500.0
WEEKS_PER_MONTH = 4.33
WORKDAYS_PER_MONTH = 26
HOURS_PER_MONTH = 208

# ---------------------------------------------------------------- Upload
MAX_UPLOAD_MB = 200             # khớp server.maxUploadSize trong .streamlit/config.toml
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp1258", "cp1252", "latin-1")
CSV_DELIMITERS = (",", ";", "\t", "|")

# ---------------------------------------------------------------- AI
AI_MAX_CONTEXT_CHARS = 12_000    # giới hạn kích thước ngữ cảnh đưa vào Gemini
AI_MAX_HISTORY_TURNS = 8
AI_TIMEOUT_SECONDS = 60
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def get_gemini_key() -> str | None:
    """Dùng chung cơ chế của ITJobTrend (src/config.py): ưu tiên key nhập tạm trong phiên, sau đó tới file .env."""
    from src.config import get_gemini_key as _base_key
    return _base_key() or None


def get_gemini_model() -> str:
    from src.config import get_gemini_model as _base_model
    return _base_model() or DEFAULT_GEMINI_MODEL


def mask_secret(value: str | None) -> str:
    """Hiển thị an toàn: chỉ cho biết có/không và 4 ký tự cuối."""
    if not value:
        return "(chưa cấu hình)"
    return "•" * 8 + value[-4:]
