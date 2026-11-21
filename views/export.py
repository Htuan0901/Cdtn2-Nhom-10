import io

import pandas as pd
import streamlit as st

from src import forecast as fc
from src.pipeline import explode_skills
from src.skills import skills_matrix
from src.state import get_data
from src.ui import page_header

page_header("📤", "Xuất kết quả & báo cáo", "Tải dữ liệu sạch, bảng đặc trưng kỹ năng, kết quả dự báo, báo cáo Excel tổng hợp "
            "và báo cáo PDF từ bộ dữ liệu phân tích.")

li, hn, log = get_data()


@st.cache_data(show_spinner=False)
def _csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


c1, c2 = st.columns(2)
with c1:
    st.markdown("#### 🗂️ Dữ liệu")
    st.download_button("⬇️ LinkedIn CNTT đã làm sạch (CSV)", _csv(li), "linkedin_it_clean.csv", "text/csv", width="stretch")
    st.download_button("⬇️ Hacker News đã làm sạch (CSV)", _csv(hn), "hn_jobs_clean.csv", "text/csv", width="stretch")
    if st.button("Tạo bảng đặc trưng kỹ năng (feature table)", width="stretch"):
        feat = pd.concat([li[["job_id", "role", "experience", "state", "salary_usd"]], skills_matrix(li["skills"])], axis=1)
        st.download_button("⬇️ feature_table_linkedin.csv", _csv(feat), "feature_table_linkedin.csv", "text/csv", width="stretch")
with c2:
    st.markdown("#### 📈 Kết quả mô hình")
    r = st.session_state.get("fc_result")
    if r:
        st.download_button(f"⬇️ So sánh mô hình dự báo – {r['target']}", _csv(r["results"]), "forecast_backtest.csv", "text/csv",
                           width="stretch")
    if st.session_state.get("fc_future") is not None:
        st.download_button("⬇️ Giá trị dự báo tương lai", _csv(st.session_state["fc_future"].reset_index(names="month")),
                           "forecast_future.csv", "text/csv", width="stretch")
    b = st.session_state.get("salary_bundle")
    if b:
        st.download_button("⬇️ So sánh mô hình lương", _csv(b["results"]), "salary_models.csv", "text/csv", width="stretch")
    if not r and not b:
        st.info("Chạy trang **Xu hướng & dự báo** hoặc **Dự đoán mức lương** để có kết quả mô hình.")

st.markdown("#### 📑 Báo cáo Excel tổng hợp")
st.caption("Gồm nhiều sheet: nhật ký làm sạch, top kỹ năng, lương theo vị trí/cấp bậc/bang, số tin HN theo tháng, xếp hạng xu hướng, kết quả mô hình.")
if st.button("📑 Tạo báo cáo Excel", type="primary"):
    with st.spinner("Đang tạo…"):
        buf = io.BytesIO()
        sal = li.dropna(subset=["salary_usd"])
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            if log is not None and len(log):
                log.to_excel(w, sheet_name="Nhat_ky_lam_sach", index=False)
            sk = explode_skills(li)["skill"].value_counts().rename_axis("Kỹ năng").reset_index(name="Số tin")
            sk["% tin"] = (sk["Số tin"] / len(li) * 100).round(2)
            sk.to_excel(w, sheet_name="Top_ky_nang", index=False)
            for col, name in [("role", "Luong_theo_vi_tri"), ("experience", "Luong_theo_cap_bac"), ("state", "Luong_theo_bang")]:
                (sal.groupby(col)["salary_usd"].describe()[["count", "25%", "50%", "75%", "mean"]].round(0)
                 .sort_values("50%", ascending=False).to_excel(w, sheet_name=name))
            hn.groupby("month").size().rename("Số tin").to_frame().to_excel(w, sheet_name="HN_theo_thang")
            fc.trend_ranking(hn, min_posts=300).to_excel(w, sheet_name="Xu_huong_ky_nang", index=False)
            if r:
                r["results"].to_excel(w, sheet_name="Mo_hinh_du_bao", index=False)
            if b:
                b["results"].to_excel(w, sheet_name="Mo_hinh_luong", index=False)
        st.download_button("⬇️ Tải ITJobTrend_BaoCao.xlsx", buf.getvalue(), "ITJobTrend_BaoCao.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ============================ BÁO CÁO PDF (module mở rộng) ============================
st.divider()
st.markdown("#### Báo cáo PDF (bộ dữ liệu phân tích)")
st.caption("Báo cáo 11 phần tiếng Việt có biểu đồ, dựng từ kết quả THỰC TẾ của các module phân tích tổng quát "
           "(làm sạch tương tác, EDA tự động, khai phá, ML, dự báo, bất thường) và kèm kết quả dự báo theo tháng / mô hình lương "
           "của trang Xu hướng & dự báo / Dự đoán mức lương nếu đã chạy. Phần chưa chạy được ghi rõ 'chưa thực hiện'.")
from src.analytics.ui.pages import report as _report  # noqa: E402
from src.analytics.ui.runner import run_page  # noqa: E402

run_page(lambda: _report.render(embedded=True))
