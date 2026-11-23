"""Quản lý dữ liệu trong phiên làm việc: tự nạp dữ liệu đã xử lý (data/processed) nếu có."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .pipeline import load_processed


@st.cache_data(show_spinner="Đang nạp dữ liệu đã xử lý…")
def _cached_load(_ver: float):
    return load_processed()


def _version() -> float:
    from .pipeline import LI_FILE, HN_FILE
    try:
        return LI_FILE.stat().st_mtime + HN_FILE.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def set_data(li: pd.DataFrame, hn: pd.DataFrame, log: pd.DataFrame, source: str):
    st.session_state.update(li=li, hn=hn, clean_log=log, data_source=source)
    for k in ["salary_bundle", "fc_result"]:
        st.session_state.pop(k, None)


def get_data(stop: bool = True):
    """Trả về (li, hn, log). Nếu phiên chưa có dữ liệu thì tự nạp từ data/processed."""
    if st.session_state.get("li") is None:
        res = _cached_load(_version())
        if res is not None:
            li, hn, log = res
            set_data(li, hn, log, "Dữ liệu đã xử lý sẵn (data/processed)")
    if st.session_state.get("li") is None:
        if stop:
            st.info("Chưa có dữ liệu. Vào **Nhập dữ liệu** để chạy pipeline từ 2 file Kaggle.", icon=":material/upload_file:")
            st.page_link("views/upload.py", label="Đi tới Nhập dữ liệu", icon=":material/arrow_forward:")
            st.stop()
        return None, None, None
    return st.session_state["li"], st.session_state["hn"], st.session_state.get("clean_log")
