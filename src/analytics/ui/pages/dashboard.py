"""MODULE 1 - Dashboard tổng quan: KPI và biểu đồ lấy trực tiếp từ dataset hiện tại."""
import pandas as pd
import streamlit as st

from src.analytics import state
from src.analytics.analysis import eda
from src.analytics.data import schema
from src.analytics.data.cleaning import quality_summary
from src.analytics.db.database import count_trained_models
from src.analytics.ui import charts
from src.analytics.ui import components as ui
from src.analytics.ui.theme import BLUE, DANGER, NAVY, TEAL, WARNING


def render():
    ss = st.session_state
    ui.page_header("Dashboard tổng quan dữ liệu", "KPI chất lượng, phân phối, tương quan và xu hướng của bộ dữ liệu phân tích đang chọn (lấy trực tiếp từ dữ liệu, không dùng số liệu giả).",
                   "dashboard")
    df = ui.require_data()
    if df is None:
        return

    q = quality_summary(df)
    n_models = count_trained_models(ss.get("dataset_id"))
    ui.kpi_cards([
        {"label": "Tổng số bản ghi", "value": ui.fmt_num(q["rows"]), "icon": "rows", "accent": BLUE, "hint": "Số dòng hiện tại"},
        {"label": "Tổng số thuộc tính", "value": ui.fmt_num(q["cols"]), "icon": "columns", "accent": NAVY, "hint": "Số cột"},
        {"label": "Ô dữ liệu thiếu", "value": ui.fmt_num(q["missing_cells"]), "icon": "missing", "accent": WARNING,
         "hint": f"{q['missing_pct']}% tổng số ô"},
        {"label": "Tỷ lệ dữ liệu hợp lệ", "value": f"{q['valid_pct']}%", "icon": "check", "accent": TEAL, "hint": "Ô không bị thiếu"},
        {"label": "Bản ghi trùng lặp", "value": ui.fmt_num(q["duplicates"]), "icon": "copy", "accent": DANGER, "hint": "Bỏ qua cột ID khi so sánh"},
        {"label": "Số lượng biến số", "value": ui.fmt_num(len(schema.numeric_columns(df))), "icon": "hash", "accent": BLUE,
         "hint": "Cột số dùng cho phân tích"},
        {"label": "Mô hình đã huấn luyện", "value": ui.fmt_num(n_models), "icon": "cpu", "accent": NAVY, "hint": "Trên dataset này"},
        {"label": "Cập nhật dữ liệu", "value": f"<span style='font-size:1.05rem'>{ss.get('updated_at') or '—'}</span>",
         "icon": "clock", "accent": TEAL, "hint": f"{len(ss.get('ws_clean_log') or [])} bước làm sạch đã áp dụng"},
    ])

    if "salary_mid" in df.columns or "is_it" in df.columns:
        _job_strip(df)

    nums = schema.numeric_columns(df)
    cats = schema.categorical_columns(df, max_unique=40)
    tcands = schema.time_candidates(df)
    tcol = ss.get("time_col") if ss.get("time_col") in df.columns else (tcands[0] if tcands else None)

    c1, c2 = st.columns(2)
    with c1:
        ui.section("Phân bố các biến số")
        if nums:
            col = st.selectbox("Biến", nums, key="dash_hist", label_visibility="collapsed")
            ui.plot(charts.histogram(df, col))
        else:
            ui.callout("Dataset chưa có biến số. Hãy chuyển kiểu dữ liệu ở trang Làm sạch hoặc tạo đặc trưng tuyển dụng.", "warn")
    with c2:
        ui.section("Xu hướng dữ liệu theo thời gian")
        if tcol:
            opts = ["(Số bản ghi)"] + nums
            v = st.selectbox("Chỉ số", opts, key="dash_trend", label_visibility="collapsed")
            ts, freq = eda.time_aggregate(df, tcol, None if v == opts[0] else v)
            ui.plot(charts.line(ts, "Thời gian", ts.columns[1], f"{ts.columns[1]} - {eda.FREQ_LABEL.get(freq, freq).lower()} (cột {tcol})"))
        else:
            ui.callout("Không phát hiện cột thời gian trong dữ liệu ⇒ không vẽ xu hướng theo thời gian.", "info")

    c3, c4 = st.columns(2)
    with c3:
        ui.section("Ma trận tương quan", "Pearson, chỉ các biến số hợp lệ")
        corr = eda.correlation_matrix(df, nums[:15])
        if corr.empty:
            ui.callout("Cần ít nhất 2 biến số có biến thiên để tính tương quan.", "info")
        else:
            ui.plot(charts.heatmap(corr, "Hệ số tương quan Pearson", height=420))
    with c4:
        ui.section("Phân phối các chỉ số", "Boxplot chuẩn hóa z-score để so sánh trên cùng thang đo")
        if nums:
            ui.plot(charts.box(df, nums[:8], "Phân phối các chỉ số", normalize=len(nums) > 1))
        else:
            ui.callout("Không có biến số.", "info")

    c5, c6 = st.columns([3, 2])
    with c5:
        ui.section("So sánh dữ liệu theo nhóm")
        if cats:
            a, b = st.columns(2)
            g = a.selectbox("Nhóm theo", cats, key="dash_group")
            v = b.selectbox("Chỉ số", ["(Số bản ghi)"] + nums, key="dash_group_val")
            if v == "(Số bản ghi)":
                tab = eda.group_stats(df, g, None, top=15)
                ui.plot(charts.bar(tab, g, "Số bản ghi", horizontal=True, height=420))
            else:
                tab = eda.group_stats(df, g, v, top=15).sort_values("Median", ascending=False)
                ui.plot(charts.bar(tab, g, "Median", f"Trung vị {v} theo {g}", horizontal=True, height=420))
        else:
            ui.callout("Không có cột phân loại phù hợp (≤ 40 giá trị) để so sánh nhóm.", "info")
    with c6:
        ui.section("Các phát hiện quan trọng", "Tự động từ dữ liệu")
        ui.insights(eda.key_findings(df))
        sk = state.get_result(ss, "skills")
        if sk is not None and not sk["counts"].empty:
            top = sk["counts"].iloc[0]
            ui.insights([f"Kỹ năng được yêu cầu nhiều nhất: **{top['Kỹ năng']}** ({top['Tỷ lệ tin (%)']}% số tin)."])


def _job_strip(df: pd.DataFrame):
    ui.section("Chỉ số thị trường tuyển dụng", "Từ các đặc trưng đã chuẩn hóa")
    items = []
    if "is_it" in df.columns:
        items.append({"label": "Tin ngành CNTT", "value": ui.fmt_num(int(df["is_it"].sum())), "icon": "code", "accent": BLUE,
                      "hint": f"{100 * df['is_it'].mean():.1f}% tổng số tin"})
    if "salary_mid" in df.columns:
        s = df["salary_mid"].dropna()
        items.append({"label": "Lương trung vị", "value": f"{s.median():.1f} tr" if len(s) else "—", "icon": "money", "accent": TEAL,
                      "hint": f"Triệu VND/tháng · {len(s):,} tin có lương"})
    if "exp_years" in df.columns:
        items.append({"label": "Kinh nghiệm trung vị", "value": f"{df['exp_years'].median():.1f} năm", "icon": "clock", "accent": NAVY,
                      "hint": f"{100 * (df['exp_years'] == 0).mean():.1f}% tin không yêu cầu kinh nghiệm"})
    if "province" in df.columns:
        top = df["province"].value_counts()
        items.append({"label": "Địa phương dẫn đầu", "value": f"<span style='font-size:1.1rem'>{top.index[0]}</span>", "icon": "map",
                      "accent": BLUE, "hint": f"{100 * top.iloc[0] / len(df):.1f}% số tin · {df['province'].nunique()} địa phương"})
    ui.kpi_cards(items)
