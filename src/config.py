"""Cấu hình chung của ứng dụng (đọc từ file .env)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "models"

# Nạp biến môi trường từ file .env (nếu có)
load_dotenv(ROOT_DIR / ".env", override=True)


def get_gemini_key() -> str:
    """Trả về API key Gemini. Ưu tiên key người dùng nhập tạm trong phiên làm việc."""
    try:
        import streamlit as st

        k = st.session_state.get("gemini_key_override", "")
        if k:
            return k.strip()
    except Exception:
        pass
    load_dotenv(ROOT_DIR / ".env", override=True)
    return os.getenv("GEMINI_API_KEY", "").strip()


def get_gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"


APP_NAME = "ITJobTrend"
APP_TAGLINE = "Phân tích & dự báo xu hướng tuyển dụng ngành CNTT từ dữ liệu tin đăng việc làm"
