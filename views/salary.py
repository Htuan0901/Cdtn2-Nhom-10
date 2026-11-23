import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.pipeline import EXP_ORDER
from src.salary_model import (OVERALL_LABEL, predict_salary, salary_trend_change, salary_trend_table,
                               train_salary_models)
from src.skills import SKILL_NAMES
from src.state import get_data
from src.ui import SERIES, fmt_int, fmt_usd, hico, kpi_cards, page_header, style_fig

page_header("money", "Dự đoán mức lương", "Mô hình hồi quy dự đoán lương (USD/năm) từ vị trí, cấp bậc, bang, hình thức làm việc và kỹ năng "
            "(LinkedIn). So sánh với baseline bằng MAE, RMSE, R².")

li, hn, _ = get_data()
sal = li[li["salary_usd"].notna()]
st.caption(f"Dữ liệu: {fmt_int(len(sal))} tin CNTT có công bố lương. Chia ngẫu nhiên 60% train / 20% validation / 20% test. "
           "Chọn mô hình theo validation MAE, test chỉ để đánh giá.")

if st.button("Huấn luyện & so sánh mô hình", icon=":material/model_training:", type="primary") or st.session_state.get("salary_bundle") is None:
    with st.spinner("Đang huấn luyện 4 mô hình…"):
        st.session_state["salary_bundle"] = train_salary_models(li)
b = st.session_state["salary_bundle"]
res = b["results"]
best = res[res["Mô hình"] == b["best"]].iloc[0]
base = res[res["Mô hình"].str.startswith("Baseline")].iloc[0]
kpi_cards([
    ("Mô hình tốt nhất", b["best"]),
    ("MAE trên test", fmt_usd(best["MAE (USD)"]), f"Baseline: {fmt_usd(base['MAE (USD)'])}"),
    ("Giảm sai số so với baseline", f"{best['Giảm MAE so với baseline (%)']:.1f}%", "", "up"),
    ("R² trên test", f"{best['R²']:.3f}", "tỉ lệ biến thiên giải thích được"),
    ("Train / Val / Test", f"{fmt_int(b['n_train'])} / {fmt_int(b['n_validation'])} / {fmt_int(b['n_test'])}",
     f"mô hình cuối học lại trên train+val ({fmt_int(b['n_train'] + b['n_validation'])} tin)"),
])

t1, t_trend, t2, t3 = st.tabs([":material/leaderboard: So sánh mô hình", ":material/trending_up: Xu hướng lương",
                             ":material/tune: Yếu tố ảnh hưởng", ":material/calculate: Dự đoán thử"])
with t1:
    st.caption("Xếp hạng theo Validation MAE (USD). MAE/RMSE/R² là kết quả test giữ lại độc lập; "
               "mô hình tốt nhất không được chọn bằng test.")
    st.dataframe(res.style.format({"MAE (USD)": "{:,.0f}", "RMSE (USD)": "{:,.0f}", "R²": "{:.3f}",
                                   "Validation MAE (USD)": "{:,.0f}", "Validation R²": "{:.3f}", "Thời gian (s)": "{:.1f}",
                                   "Giảm MAE so với baseline (%)": "{:+.1f}"})
                 .highlight_min(subset=["Validation MAE (USD)"], color="#d9f2e6"), width="stretch", hide_index=True)
    st.caption("Ô tô xanh là tiêu chí chọn mô hình (Validation MAE nhỏ nhất). Các cột test chỉ để báo cáo, không tô màu "
               "để tránh hiểu nhầm là chọn theo test.")
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
    st.markdown("##### Xu hướng lương trung vị theo thời gian, phân nhóm vị trí")
    src_name = st.radio("Nguồn dữ liệu", ["Hacker News (2018–2023)", "LinkedIn (2024)"], horizontal=True,
                        key="salary_trend_src",
                        help="Hacker News trải dài 6 năm nên thể hiện được xu hướng; LinkedIn gần như chỉ có tin tháng 3–4/2024.")
    is_hn = src_name.startswith("Hacker")
    src_df = hn if is_hn else li
    src_sal = src_df[src_df["salary_usd"].notna() & (src_df["salary_usd"] > 0)]
    date_col = "month" if is_hn else "listed_date"
    freq_opts = {"Năm": "YS", "Nửa năm": "6MS", "Quý": "QS"} if is_hn else {"Tháng": "MS", "Quý": "QS"}
    c_freq, c_count, c_all = st.columns([2, 2, 1])
    period = c_freq.radio("Gộp theo", list(freq_opts), horizontal=True, key=f"salary_trend_period_{is_hn}")
    threshold = c_count.slider("Tối thiểu số tin mỗi nhóm/kỳ", 1, 30, 10 if is_hn else 5,
                               key=f"salary_trend_threshold_{is_hn}")
    show_all = c_all.toggle("Đường tổng", True, key="salary_trend_all", help="Thêm đường trung vị của tất cả vị trí")
    role_counts = src_sal["role"].value_counts()
    role_counts = role_counts[role_counts.index != "Khác (IT)"]
    roles = st.multiselect("Nhóm vị trí", role_counts.index.tolist(), default=role_counts.head(4).index.tolist(),
                           key=f"salary_trend_roles_{is_hn}")
    trend = salary_trend_table(src_sal, freq_opts[period], threshold, roles, date_col=date_col,
                               include_overall=show_all)
    if trend.empty:
        st.info("Không có nhóm nào đủ số tin với bộ lọc hiện tại. Hãy giảm ngưỡng hoặc chọn thêm vị trí.")
    else:
        if trend["Kỳ"].nunique() < 3:
            st.warning("Chỉ có dưới 3 kỳ đủ mẫu với bộ lọc hiện tại: đồ thị chỉ mô tả các kỳ quan sát được, "
                       "chưa đủ cơ sở kết luận xu hướng.")
        fig = px.line(trend, x="Kỳ", y="Lương trung vị (USD/năm)", color="Vị trí", markers=True,
                      hover_data={"Số tin": True},
                      title=f"Lương trung vị theo {period.lower()} và nhóm vị trí – {src_name} (USD/năm)")
        fig.for_each_trace(lambda t: t.update(line=dict(width=4, dash="dash", color="#1b1f29"))
                           if t.name == OVERALL_LABEL else None)
        fig.update_layout(yaxis_tickprefix="$", xaxis_title="Kỳ đăng tin")
        st.plotly_chart(style_fig(fig, 440), width="stretch")
        change = salary_trend_change(trend) if trend["Kỳ"].nunique() >= 3 else pd.DataFrame()
        fmt_p = "%Y" if freq_opts[period] == "YS" else "%m/%Y"
        if not change.empty:
            st.markdown("##### Thay đổi giữa kỳ đầu và kỳ cuối có đủ mẫu")
            st.dataframe(change.style.format({"Kỳ đầu": lambda d: d.strftime(fmt_p), "Kỳ cuối": lambda d: d.strftime(fmt_p), "Lương kỳ đầu": "{:,.0f}",
                                              "Lương kỳ cuối": "{:,.0f}", "Thay đổi (%)": "{:+.1f}"}),
                         hide_index=True, width="stretch")
            if OVERALL_LABEL in set(change["Vị trí"]):
                r = change[change["Vị trí"] == OVERALL_LABEL].iloc[0]
                up = r["Thay đổi (%)"] >= 0
                st.info(f"Trung vị lương của các tin có công bố lương {'tăng' if up else 'giảm'} từ "
                        f"**{fmt_usd(r['Lương kỳ đầu'])}** ({r['Kỳ đầu'].strftime(fmt_p)}) {'lên' if up else 'xuống'} "
                        f"**{fmt_usd(r['Lương kỳ cuối'])}** ({r['Kỳ cuối'].strftime(fmt_p)}), "
                        f"tức **{r['Thay đổi (%)']:+.1f}%** trên {fmt_int(r['Tổng số tin'])} tin.")
        with st.expander("Bảng số liệu chi tiết"):
            st.dataframe(trend.style.format({"Kỳ": "{:%Y-%m}", "Lương trung vị (USD/năm)": "{:,.0f}"}),
                         hide_index=True, width="stretch")
    date_values = pd.to_datetime(src_sal[date_col], errors="coerce").dropna()
    if not date_values.empty:
        st.caption(f"Phạm vi tin có lương: {date_values.min():%m/%Y} – {date_values.max():%m/%Y} "
                   f"({fmt_int(len(src_sal))} tin). Mỗi điểm là trung vị lương được công bố trong tin đăng, "
                   "không phải lương đại diện toàn thị trường. "
                   + ("Năm 2023 chỉ có dữ liệu đến tháng 6. Lương Hacker News là trung điểm khoảng lương dạng $120k–$160k ghi trong tin."
                      if is_hn else "LinkedIn tập trung gần hết ở 04/2024 nên không dùng để kết luận xu hướng."))

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
    st.markdown(f"""<div class="card"><h4>{hico("money")}Mức lương dự đoán: <span style="color:{SERIES[0]}">{fmt_usd(pred)}</span> / năm</h4>
    Tham khảo: lương thực tế của <b>{role}</b> cấp <b>{exp}</b> trong dữ liệu có trung vị
    <b>{fmt_usd(same.median()) if len(same) else '–'}</b> ({fmt_int(len(same))} tin).</div>""", unsafe_allow_html=True)
