"""ITJobTrend – Ứng dụng đồ án Chuyên đề tốt nghiệp 2 (Khoa học dữ liệu).
Phân tích & dự báo xu hướng tuyển dụng ngành CNTT từ dữ liệu tin đăng việc làm.

Chạy:  streamlit run app.py

Kiến trúc:
- views/*.py      : các trang giao diện (mỗi file một trang).
- src/*.py        : pipeline Kaggle (LinkedIn + Hacker News), kỹ năng, dự báo theo tháng, mô hình lương, chatbot.
- src/analytics/  : các module phân tích tổng quát (upload CSV bất kỳ, làm sạch tương tác, EDA tự động, K-Means/PCA,
                    Machine Learning, ARIMA/SARIMA, Isolation Forest, Data Analysis Agent, SQLite, báo cáo PDF).
"""
import streamlit as st

from src.analytics import state as ws_state
from src.analytics.ui import nav
from src.config import APP_NAME, ROOT_DIR
from src.state import get_data
from src.ui import fmt_int, inject_css

st.set_page_config(page_title=f"{APP_NAME} – Xu hướng tuyển dụng CNTT", page_icon=str(ROOT_DIR / "assets" / "logo.png"),
                   layout="wide", initial_sidebar_state="expanded")
inject_css()
ws_state.init_state(st.session_state)

st.logo(str(ROOT_DIR / "assets" / "logo.svg"), size="large", icon_image=str(ROOT_DIR / "assets" / "icon.svg"))

# (khóa, file, tiêu đề, icon Material - không dùng emoji)
PAGE_DEFS = {
    "Tổng quan": [
        ("home", "views/home.py", "Trang chủ", ":material/home:"),
        ("overview", "views/overview.py", "Dashboard tổng quan dữ liệu", ":material/dashboard:"),
    ],
    "Dữ liệu": [
        ("upload", "views/upload.py", "Nhập dữ liệu Kaggle", ":material/cloud_download:"),
        ("dataset", "views/dataset.py", "Bộ dữ liệu phân tích", ":material/upload_file:"),
        ("cleaning", "views/cleaning.py", "Làm sạch (pipeline Kaggle)", ":material/cleaning_services:"),
        ("cleaning_interactive", "views/cleaning_interactive.py", "Làm sạch tương tác", ":material/tune:"),
    ],
    "Phân tích": [
        ("eda", "views/eda.py", "EDA thị trường CNTT", ":material/query_stats:"),
        ("eda_auto", "views/eda_auto.py", "EDA tự động", ":material/analytics:"),
        ("skills", "views/skills.py", "Trích xuất kỹ năng", ":material/extension:"),
        ("jobs", "views/vn_market.py", "Thị trường tuyển dụng (VN)", ":material/work:"),
        ("dashboard", "views/dashboard.py", "Dashboard thị trường", ":material/space_dashboard:"),
        ("mining", "views/mining.py", "Khai phá dữ liệu", ":material/hub:"),
    ],
    "Mô hình": [
        ("forecast", "views/forecast.py", "Xu hướng & dự báo", ":material/trending_up:"),
        ("forecast_general", "views/forecast_general.py", "Dự báo tổng quát", ":material/timeline:"),
        ("salary", "views/salary.py", "Dự đoán mức lương", ":material/payments:"),
        ("ml", "views/ml.py", "Machine Learning", ":material/model_training:"),
        ("anomaly", "views/anomaly.py", "Phát hiện bất thường", ":material/crisis_alert:"),
    ],
    "Trợ lý & Báo cáo": [
        ("chat", "views/chat.py", "Trợ lý AI (Gemini)", ":material/auto_awesome:"),
        ("export", "views/export.py", "Xuất kết quả & báo cáo", ":material/description:"),
    ],
    "Hệ thống": [
        ("system", "views/system.py", "Thông tin hệ thống", ":material/settings:"),
    ],
}

pages = {}
for section, items in PAGE_DEFS.items():
    pages[section] = []
    for key, path, title, icon in items:
        page = st.Page(path, title=title, icon=icon, default=(key == "home"))
        nav.PAGES[key] = page
        pages[section].append(page)
current = st.navigation(pages)

with st.sidebar:
    li, hn, _ = get_data(stop=False)
    st.markdown("**Trạng thái dữ liệu**")
    if li is None:
        st.caption(":material/radio_button_unchecked: Chưa có dữ liệu Kaggle")
    else:
        st.caption(f":material/check_circle: LinkedIn (CNTT): **{fmt_int(len(li))}** tin")
        st.caption(f":material/check_circle: Hacker News: **{fmt_int(len(hn))}** tin")
    ss = st.session_state
    if ss.get("df") is not None and ss.get("meta"):
        st.caption(f":material/dataset: Bộ phân tích: **{ss['meta'].get('name', '')[:34]}** ({fmt_int(len(ss['df']))} dòng)")
    else:
        st.caption(":material/dataset: Bộ phân tích: chưa chọn")
    if ss.get("fc_result"):
        st.caption(f":material/trending_up: Dự báo: **{ss['fc_result']['best']}**")
    if ss.get("salary_bundle"):
        st.caption(f":material/payments: Mô hình lương: **{ss['salary_bundle']['best']}**")

current.run()
