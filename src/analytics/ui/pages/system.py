"""MODULE 11 - System Information: phiên bản, thư viện, trạng thái Gemini, lịch sử hoạt động, huấn luyện, lỗi."""
import platform
import sys
from importlib import metadata

import pandas as pd
import streamlit as st

from src.analytics import state
from src.analytics.ai.gemini_client import GeminiClient, sdk_available
from src.analytics.config import APP_NAME, APP_VERSION, DB_PATH, PROJECT_TITLE_VI, get_gemini_key, get_gemini_model, mask_secret
from src.analytics.db import database as db
from src.analytics.forecasting.forecaster import has_statsmodels
from src.analytics.ml.trainer import HAS_XGB
from src.analytics.ui import components as ui
from src.analytics.ui.theme import BLUE, NAVY, TEAL

LIBS = ["streamlit", "pandas", "numpy", "scikit-learn", "xgboost", "statsmodels", "scipy", "plotly", "matplotlib", "seaborn",
        "google-genai", "reportlab", "python-dotenv", "pyarrow", "joblib", "pytest"]


def _version(lib: str) -> str:
    try:
        return metadata.version(lib)
    except metadata.PackageNotFoundError:
        return "chưa cài"


def render():
    ss = st.session_state
    ui.page_header("Thông tin hệ thống", "Phiên bản, thư viện, trạng thái kết nối AI, lịch sử hoạt động và nhật ký lỗi.", "settings", badge=f"v{APP_VERSION}")
    key = get_gemini_key()
    ui.kpi_cards([
        {"label": "Phiên bản hệ thống", "value": APP_VERSION, "icon": "settings", "accent": NAVY, "hint": APP_NAME},
        {"label": "Python", "value": platform.python_version(), "icon": "code", "accent": BLUE, "hint": platform.system()},
        {"label": "Gemini API Key", "value": "<span style='font-size:1rem'>" + mask_secret(key) + "</span>", "icon": "chat", "accent": TEAL,
         "hint": f"Model: {get_gemini_model()}"},
        {"label": "Cơ sở dữ liệu", "value": "SQLite", "icon": "rows", "accent": BLUE, "hint": "Hoạt động" if db.health_check() else "Lỗi"},
    ])
    st.caption(PROJECT_TITLE_VI)
    t1, t2, t3, t4, t5 = st.tabs(["Trạng thái module", "Thư viện", "Lịch sử hoạt động", "Lịch sử huấn luyện", "Nhật ký lỗi"])
    with t1:
        has = state.has_data(ss)
        res = {k: state.get_result(ss, k) is not None for k in state.RESULT_KEYS}
        rows = [
            ("Dashboard", has, "Cần dataset"), ("Data Upload", True, "Sẵn sàng"), ("Làm sạch dữ liệu", has, f"{len(ss.get('ws_clean_log') or [])} bước đã áp dụng"),
            ("EDA", has, "Sẵn sàng" if has else "Cần dataset"),
            ("Thị trường CNTT", has and ("salary_mid" in ss["df"] or "is_it" in ss["df"]), "Đã trích xuất kỹ năng" if res["skills"] else "Chưa trích xuất kỹ năng"),
            ("Data Mining", res["mining"], "Đã chạy" if res["mining"] else "Chưa chạy"),
            ("Machine Learning", res["ml"], f"XGBoost: {'có' if HAS_XGB else 'chưa cài'}"),
            ("Dự báo", res["forecast"], f"statsmodels: {'có' if has_statsmodels() else 'chưa cài'}"),
            ("Phát hiện bất thường", res["anomaly"], "Đã chạy" if res["anomaly"] else "Chưa chạy"),
            ("AI Assistant", bool(key) and sdk_available(), "Sẵn sàng" if key and sdk_available() else "Chế độ offline (thiếu API Key hoặc SDK)"),
            ("Báo cáo PDF", has, "Sẵn sàng" if has else "Cần dataset"),
        ]
        ui.show_df(pd.DataFrame([{"Module": m, "Trạng thái": "Hoạt động" if ok else "Chưa sẵn sàng", "Ghi chú": n} for m, ok, n in rows]))
        ui.section("Dataset hiện tại")
        if has:
            m = ss["meta"]
            ui.show_df(pd.DataFrame([("Tên", m.get("name")), ("Dataset ID", m.get("dataset_id")), ("Dòng × cột", f"{len(ss['df']):,} × {ss['df'].shape[1]}"),
                                     ("Cột thời gian", ss.get("time_col")), ("Biến mục tiêu", ss.get("target_col")),
                                     ("Cập nhật", ss.get("updated_at"))], columns=["Thuộc tính", "Giá trị"]).astype(str))
        else:
            st.caption("Chưa có dataset.")
        ui.section("Kiểm tra kết nối Gemini")
        st.caption("API Key được đọc từ file .env (biến GEMINI_API_KEY), không hiển thị và không ghi vào log.")
        if ui.button("Kiểm tra kết nối", key="ping", full=False, disabled=not key):
            with st.spinner("Đang gọi Gemini..."):
                ok, msg = GeminiClient().ping()
            (st.success if ok else st.error)(msg)
    with t2:
        ui.show_df(pd.DataFrame([{"Thư viện": l, "Phiên bản": _version(l)} for l in LIBS]))
        st.caption(f"Python {sys.version.split()[0]} · CSDL: {DB_PATH}")
    with t3:
        ui.show_df(db.read_table("activity_log", 300), height=420)
    with t4:
        th = db.read_table("training_history", 300)
        ui.show_df(th, height=420)
    with t5:
        er = db.read_table("error_log", 200)
        if er.empty:
            ui.callout("Chưa ghi nhận lỗi nào.", "ok")
        else:
            ui.show_df(er.drop(columns=["trace"]), height=300)
            with st.expander("Chi tiết traceback lỗi gần nhất"):
                st.code(er.iloc[0]["trace"] or "(không có)")
