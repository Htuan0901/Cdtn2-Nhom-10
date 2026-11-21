import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.pipeline import EXP_ORDER
from src.salary_model import predict_salary, train_salary_models, salary_trend_table
from src.skills import SKILL_NAMES
from src.state import get_data
from src.ui import SERIES, fmt_int, fmt_usd, kpi_cards, page_header, style_fig

page_header("💵", "Dự đoán mức lương", "Mô hình hồi quy dự đoán lương (USD/năm) từ vị trí, cấp bậc, bang, hình thức làm việc và kỹ năng "
            "(LinkedIn). So sánh với baseline bằng MAE, RMSE, R².")

li, _, _ = get_data()
sal = li[li["salary_usd"].notna()]
st.caption(f"Dữ liệu: {fmt_int(len(sal))} tin CNTT có công bố lương. Chia ngẫu nhiên 60% train / 20% validation / 20% test. "
           "Chọn mô hình theo validation MAE, test chỉ để đánh giá.")

if st.button("🧠 Huấn luyện & so sánh mô hình", type="primary") or st.session_state.get("salary_bundle") is None:
    with st.spinner("Đang huấn luyện 4 mô hình…"):
        st.session_state["salary_bundle"] = train_salary_models(li)
b = st.session_state["salary_bundle"]
res = b["results"]
best = res[res["Mô hình"] == b["best"]].iloc[0]
base = res[res["Mô hình"].str.startswith("Baseline")].iloc[0]
kpi_cards([
    ("Mô hình tốt nhất", b["best"]),
    ("MAE", fmt_usd(best["MAE (USD)"]), f"Baseline: {fmt_usd(base['MAE (USD)'])}"),
    ("Giảm sai số so với baseline", f"{best['Giảm MAE so với baseline (%)']:.1f}%", "", "up"),
    ("R²", f"{best['R²']:.3f}", "tỉ lệ biến thiên giải thích được"),
    ("Train / Val / Test", f"{fmt_int(b['n_train'])} / {fmt_int(b['n_validation'])} / {fmt_int(b['n_test'])}"),
])

t1, t_trend, t2, t3 = st.tabs(["📋 So sánh mô hình", "📈 Xu hướng lương", "🔍 Yếu tố ảnh hưởng", "🎯 Dự đoán thử"])
with t1:
    st.caption("Xếp hạng theo Validation MAE (USD). MAE/RMSE/R² là kết quả test giữ lại độc lập; "
               "mô hình tốt nhất không được chọn bằng test.")
    st.dataframe(res.style.format({"MAE (USD)": "{:,.0f}", "RMSE (USD)": "{:,.0f}", "R²": "{:.3f}",
                                   "Validation MAE (USD)": "{:,.0f}", "Validation R²": "{:.3f}", "Thời gian (s)": "{:.1f}",
                                   "Giảm MAE so với baseline (%)": "{:+.1f}"})
                 .highlight_min(subset=["MAE (USD)", "RMSE (USD)"], color="#d9f2e6"), width="stretch", hide_index=True)
    c1, c2 = st.columns(2)
    fig = px.bar(res, x="Mô hình", y="MAE (USD)", color="Mô hình", title="MAE trên tập test (càng thấp càng tốt)",
                 text=res["MAE (USD)"].map(fmt_usd))
    fig.update_layout(showlegend=False, xaxis_title="")
    c1.plotly_chart(style_fig(fig, 360), width="stretch")
    p = b["pred_test"][b["best"]]
    d = pd.DataFrame({"Thực tế": b["y_test"], "Dự đoán": p}).sample(min(1500, len(p)), random_state=0)
    fig = px.scatter(d, x="Thực tế", y="Dự đoán", opacity=0.4, title=f"Thực tế vs dự đoán – {b['best']}")
    mx = float(np.percentile(d.values, 99))
    fig.add_shape(type="line", x0=0, y0=0, x1=mx, y1=mx, line=dict(dash="dash", color="#98a2b3"))
    c2.plotly_chart(style_fig(fig, 360), width="stretch")


with t_trend:
    st.markdown("##### Trung vị lương theo tháng hoặc quý, phân nhóm vị trí")
    c_freq, c_count = st.columns([1, 1])
    period = c_freq.radio("Gộp theo", ["Tháng", "Quý"], horizontal=True, key="salary_trend_period")
    threshold = c_count.slider("Tối thiểu số tin mỗi vị trí/kỳ", 1, 30, 5, key="salary_trend_threshold")
    default_roles = sal["role"].value_counts().head(5).index.tolist()
    roles = st.multiselect("Nhóm vị trí", sorted(sal["role"].dropna().unique()),
                           default=default_roles, key="salary_trend_roles")
    trend = salary_trend_table(sal, "MS" if period == "Tháng" else "QS", threshold, roles)
    if trend["Kỳ"].nunique() < 3:
        st.warning("Dữ liệu lương theo thời gian chỉ phủ dưới 3 kỳ đủ mẫu với bộ lọc hiện tại. "
                   "Đồ thị chỉ mô tả các kỳ quan sát được, chưa đủ cơ sở kết luận xu hướng dài hạn.")
    if trend.empty:
        st.info("Không có nhóm nào đủ số tin với bộ lọc hiện tại. Hãy giảm ngưỡng hoặc chọn thêm vị trí.")
    else:
        fig = px.line(trend, x="Kỳ", y="Lương trung vị (USD/năm)", color="Vị trí", markers=True,
                      hover_data={"Số tin": True},
                      title="Xu hướng lương trung vị theo vị trí (USD/năm)")
        fig.update_layout(yaxis_tickprefix="$", xaxis_title="Kỳ đăng tin")
        st.plotly_chart(style_fig(fig, 440), width="stretch")
        st.dataframe(trend.style.format({"Lương trung vị (USD/năm)": "{:,.0f}"}),
                     hide_index=True, width="stretch")
    date_values = pd.to_datetime(sal["listed_date"], errors="coerce").dropna()
    if not date_values.empty:
        st.caption(f"Phạm vi mẫu LinkedIn: {date_values.min():%d/%m/%Y} – {date_values.max():%d/%m/%Y}. "
                   "Mỗi điểm là trung vị của các tin có công bố lương, không phải lương thị trường đại diện. "
                   "Số tin từng kỳ chênh lệch lớn, nên tránh suy luận xu hướng dài hạn.")

with t2:
    eff = b["effect"]
    eff = eff[~eff.index.str.contains("infrequent")]
    top = pd.concat([eff.head(12), eff.tail(12)])
    d = top.rename("pct").reset_index().rename(columns={"index": "Đặc trưng"})
    d["Chiều"] = np.where(d["pct"] > 0, "Tăng lương", "Giảm lương")
    fig = px.bar(d, x="pct", y="Đặc trưng", orientation="h", color="Chiều",
                 color_discrete_map={"Tăng lương": "#1baf7a", "Giảm lương": "#e34948"},
                 title="Tác động ước tính lên lương (%) – hệ số mô hình Ridge, các yếu tố khác giữ nguyên",
                 labels={"pct": "% thay đổi lương", "Đặc trưng": ""})
    st.plotly_chart(style_fig(fig, 720), width="stretch")
    st.caption("Ridge học trên log(lương) nên hệ số đổi được sang % thay đổi lương: (e^β − 1) × 100. "
               "Nhóm danh mục so với mức tham chiếu ngầm định của mã hóa one-hot, nên nên đọc theo thứ tự tương đối.")

with t3:
    c1, c2, c3 = st.columns(3)
    role = c1.selectbox("Vị trí", sorted(li["role"].unique()), index=sorted(li["role"].unique()).index("Software Engineer")
                        if "Software Engineer" in li["role"].unique() else 0)
    exp = c2.selectbox("Cấp bậc", [e for e in EXP_ORDER if e in li["experience"].unique()], index=3)
    states = ["Khác"] + b["top_states"]
    state = c3.selectbox("Bang", states, index=states.index("CA") if "CA" in states else 0)
    c4, c5 = st.columns([1, 3])
    wt = c4.selectbox("Hình thức", li["work_type"].value_counts().index.tolist())
    remote = c4.toggle("Cho làm từ xa")
    skills = c5.multiselect("Kỹ năng", SKILL_NAMES, default=["Python", "SQL", "AWS"])
    model = c5.selectbox("Mô hình", res["Mô hình"].tolist(), index=res["Mô hình"].tolist().index(b["best"]))
    pred = predict_salary(b, role, exp, state, wt, remote, skills, model)
    same = sal[(sal["role"] == role) & (sal["experience"] == exp)]["salary_usd"]
    st.markdown(f"""<div class="card"><h4>💵 Mức lương dự đoán: <span style="color:{SERIES[0]}">{fmt_usd(pred)}</span> / năm</h4>
    Tham khảo: lương thực tế của <b>{role}</b> cấp <b>{exp}</b> trong dữ liệu có trung vị
    <b>{fmt_usd(same.median()) if len(same) else '–'}</b> ({fmt_int(len(same))} tin).</div>""", unsafe_allow_html=True)
