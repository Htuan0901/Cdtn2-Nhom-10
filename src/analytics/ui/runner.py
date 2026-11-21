"""Chạy một trang phân tích mở rộng: bắt lỗi để không làm sập ứng dụng, ghi vào nhật ký lỗi (SQLite + file log)."""
import streamlit as st

from src.analytics import state
from src.analytics.db.database import log_error


def run_page(render) -> None:
    state.init_state(st.session_state)
    try:
        render()
    except Exception as exc:  # noqa: BLE001
        # Ngoại lệ điều khiển của Streamlit (rerun/stop/switch_page) phải được truyền tiếp
        if type(exc).__module__.startswith("streamlit") or type(exc).__name__ in ("RerunException", "StopException"):
            raise
        log_error(render.__module__, exc)
        st.error(f"Đã xảy ra lỗi: {exc}")
        with st.expander("Chi tiết kỹ thuật"):
            st.exception(exc)
