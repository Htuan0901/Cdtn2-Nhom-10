"""Cấu hình chung của ứng dụng (đọc từ file .env)."""
import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "models"

# Nạp biến môi trường từ file .env (nếu có). override=False để KHÔNG ghi đè biến môi trường / Secrets
# do nền tảng deploy cung cấp (Streamlit Cloud, Hugging Face...) bằng giá trị rỗng trong .env.
load_dotenv(ROOT_DIR / ".env", override=False)


def get_gemini_key() -> str:
    """Trả về API key Gemini. Ưu tiên key người dùng nhập tạm trong phiên làm việc."""
    try:
        import streamlit as st

        k = st.session_state.get("gemini_key_override", "")
        if k:
            return k.strip()
    except Exception:
        pass
    # Đọc lại .env mỗi lần (sửa key không cần khởi động lại); nếu .env trống thì dùng biến môi trường / Secrets
    env_file = ROOT_DIR / ".env"
    k = (dotenv_values(env_file).get("GEMINI_API_KEY") or "").strip() if env_file.exists() else ""
    if k:
        return k
    k = os.getenv("GEMINI_API_KEY", "").strip()
    if k:
        return k
    try:  # st.secrets (file .streamlit/secrets.toml hoặc Secrets trên Streamlit Community Cloud)
        import streamlit as st

        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        return ""


def get_setting(name: str, default: str = "") -> str:
    """Đọc một cấu hình: biến môi trường (kể cả từ .env) trước, sau đó tới st.secrets (Streamlit Cloud)."""
    v = os.getenv(name, "").strip()
    if v:
        return v
    try:
        import streamlit as st

        v = str(st.secrets.get(name, "")).strip()
    except Exception:
        v = ""
    return v or default


def is_public() -> bool:
    """True khi app chạy ở chế độ công khai trên web (APP_PUBLIC=1 trong Secrets / biến môi trường).

    Ở chế độ này app ẩn các nhật ký dùng chung (hoạt động, huấn luyện, lỗi) và bật giới hạn câu hỏi Gemini.
    """
    return get_setting("APP_PUBLIC", "0").lower() in ("1", "true", "yes", "on")


DEFAULT_PUBLIC_CHAT_LIMIT = 20


def get_chat_limit() -> int:
    """Số câu hỏi tối đa gửi tới Gemini (bằng key của nhóm) trong một phiên. 0 = không giới hạn.

    Đặt bằng GEMINI_MAX_QUESTIONS. Nếu không đặt: chế độ công khai mặc định 20 câu, chạy trên máy cá nhân không giới hạn.
    """
    raw = get_setting("GEMINI_MAX_QUESTIONS", "")
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return DEFAULT_PUBLIC_CHAT_LIMIT if is_public() else 0


def get_gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"


APP_NAME = "ITJobTrend"
APP_TAGLINE = "Phân tích & dự báo xu hướng tuyển dụng ngành CNTT từ dữ liệu tin đăng việc làm"
