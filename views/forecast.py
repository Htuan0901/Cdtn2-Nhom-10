import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import forecast as fc
from src.pipeline import explode_skills
from src.state import get_data
from src.ui import SEQ_BLUE, SERIES, fmt_int, kpi_cards, page_header, style_fig

page_header("📈", "Xu hướng & dự báo", "Phân tích xu hướng kỹ năng theo giai đoạn và dự báo nhu cầu tuyển dụng theo tháng "
            "(Hacker News 2018–2023). Mô hình được so sánh với baseline bằng MAE/RMSE/MAPE.")

_, hn, _ = get_data()


@st.cache_data(show_spinner="Đang huấn luyện và kiểm định 9 mô hình (ARIMA/SARIMA mất vài giây)…")
def _cached_backtest(series: pd.Series, h: int):
    return fc.evaluate_models(series, h)

skill_counts = explode_skills(hn)["skill"].value_counts()

tab_trend, tab_fc = st.tabs(["📊 Phân tích xu hướng theo giai đoạn", "🔮 Dự báo theo tháng"])

# ============================ XU HƯỚNG ============================
with tab_trend:
    st.markdown("So sánh **tỉ lệ % tin nhắc tới mỗi kỹ năng** giữa giai đoạn **2018–2019** (trước COVID) và **2022–2023** "
                "(sau làn sóng cắt giảm nhân sự công nghệ). Dùng tỉ lệ thay vì số tin để loại bỏ ảnh hưởng của tổng số tin.")
    rank = fc.trend_ranking(hn, min_posts=300)
    up, down = rank.head(10), rank.tail(10).iloc[::-1]
    kpi_cards([
        ("Tăng mạnh nhất", up.iloc[0]["Kỹ năng"], f"+{up.iloc[0]['Thay đổi (điểm %)']:.1f} điểm %", "up"),
        ("Tăng thứ 2", up.iloc[1]["Kỹ năng"], f"+{up.iloc[1]['Thay đổi (điểm %)']:.1f} điểm %", "up"),
        ("Giảm mạnh nhất", down.iloc[0]["Kỹ năng"], f"{down.iloc[0]['Thay đổi (điểm %)']:.1f} điểm %", "down"),
        ("Giảm thứ 2", down.iloc[1]["Kỹ năng"], f"{down.iloc[1]['Thay đổi (điểm %)']:.1f} điểm %", "down"),
    ])
    d = pd.concat([up, down])
    d["Xu hướng"] = d["Thay đổi (điểm %)"].map(lambda v: "Tăng" if v > 0 else "Giảm")
    fig = px.bar(d.sort_values("Thay đổi (điểm %)"), x="Thay đổi (điểm %)", y="Kỹ năng", orientation="h", color="Xu hướng",
                 color_discrete_map={"Tăng": "#1baf7a", "Giảm": "#e34948"},
                 title="Top 10 kỹ năng tăng/giảm (thay đổi tỉ lệ % tin, 2022–23 so với 2018–19)", labels={"Kỹ năng": ""})
    fig.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(style_fig(fig, 560), width="stretch")

    c1, c2 = st.columns([1, 2.2])
    default = [s for s in ["Python", "React", "TypeScript", "Ruby on Rails", "Rust", "Kubernetes", "LLM/GenAI"] if s in skill_counts]
    chosen = c1.multiselect("Chọn kỹ năng để so sánh", skill_counts.index.tolist(), default=default[:6])
    freq = c1.radio("Gộp theo", ["Quý", "Năm"], horizontal=True)
    if chosen:
        tbl = fc.skill_share_table(hn, chosen, "QS" if freq == "Quý" else "YS")
        long = tbl.reset_index().melt(id_vars="month", var_name="Kỹ năng", value_name="%")
        fig = px.line(long, x="month", y="%", color="Kỹ năng", markers=True,
                      title=f"Tỉ lệ % tin nhắc tới kỹ năng theo {freq.lower()}", labels={"month": ""})
        c2.plotly_chart(style_fig(fig, 420), width="stretch")
    top = skill_counts.head(20).index.tolist()
    heat = fc.skill_share_table(hn, top, "YS")
    heat.index = heat.index.year
    fig = px.imshow(heat.T.round(1), text_auto=True, color_continuous_scale=SEQ_BLUE, aspect="auto",
                    title="Bản đồ nhiệt: % tin nhắc tới 20 kỹ năng phổ biến nhất theo năm (2023: 6 tháng đầu)",
                    labels=dict(color="%", x="Năm", y=""))
    st.plotly_chart(style_fig(fig, 620), width="stretch")
    with st.expander("📋 Bảng xếp hạng xu hướng đầy đủ"):
        st.dataframe(rank, width="stretch", hide_index=True)

# ============================ DỰ BÁO ============================
with tab_fc:
    c1, c2, c3, c4 = st.columns([1.4, 1.2, 1, 1])
    target = c1.selectbox("Chuỗi cần dự báo", ["Tổng số tin tuyển dụng"] + [f"Kỹ năng: {s}" for s in skill_counts.head(40).index])
    skill = None if target.startswith("Tổng") else target.replace("Kỹ năng: ", "")
    metric = "count"
    if skill:
        metric = c2.radio("Đơn vị", ["Số tin", "% số tin"], horizontal=True)
        metric = "count" if metric == "Số tin" else "share"
    h = c3.slider("Số tháng test (h)", 3, 12, 6, help="h tháng cuối được giữ lại để đánh giá mô hình")
    future = c4.slider("Dự báo tương lai (tháng)", 3, 18, 12)

    s = fc.monthly_series(hn, skill, metric)
    unit = "số tin" if metric == "count" else "% tin"

    with st.expander("ℹ️ Các mô hình được so sánh", expanded=False):
        st.markdown("\n".join(f"- **{k}**: {v}" for k, v in fc.MODEL_INFO.items()))

    evaluation = _cached_backtest(s, h)
    res, preds = evaluation["test_table"], evaluation["test_predictions"]
    best = evaluation["best"]
    base_rmse = res.loc[res["Mô hình"] == "Naive (baseline)", "RMSE"].iloc[0]
    best_row = res[res["Mô hình"] == best].iloc[0]
    st.session_state["fc_result"] = {"target": target, "unit": unit, "h": h, "results": res, "best": best,
                                     "validation_results": evaluation["validation_table"]}

    kpi_cards([
        ("Độ dài chuỗi", f"{len(s)} tháng", f"{s.index[0]:%m/%Y} – {s.index[-1]:%m/%Y}"),
        ("Train / Val / Test", f"{len(evaluation['train'])} / {len(evaluation['validation'])} / {h}", "chia theo thời gian"),
        ("Mô hình tốt nhất", best, "chọn bằng RMSE validation"),
        ("RMSE tốt nhất", f"{best_row['RMSE']:.2f}", f"Baseline: {base_rmse:.2f}"),
        ("Cải thiện so với baseline", f"{best_row.get('Cải thiện RMSE so với baseline (%)', 0):.1f}%", "", "up"
         if best_row.get("Cải thiện RMSE so với baseline (%)", 0) > 0 else "down"),
    ])

    fig = go.Figure()
    fig.add_scatter(x=s.index, y=s.values, name="Thực tế", mode="lines+markers", line=dict(color="#1b1f29", width=2))
    for i, name in enumerate([c for c in preds.columns if c != "Thực tế"]):
        fig.add_scatter(x=preds.index, y=preds[name], name=name, mode="lines",
                        line=dict(width=3 if name == best else 1.5, dash="solid" if name == best else "dot",
                                  color=SERIES[i % len(SERIES)]))
    fig.add_vrect(x0=preds.index[0] - pd.Timedelta(days=15), x1=preds.index[-1] + pd.Timedelta(days=15),
                  fillcolor="#f3f6fb", opacity=0.6, line_width=0, annotation_text="Tập test", annotation_position="top left")
    fig.update_layout(title=f"Backtest: dự báo {h} tháng cuối – {target} ({unit})", yaxis_title=unit)
    st.plotly_chart(style_fig(fig, 430), width="stretch")

    show = res.copy()
    st.caption("Mô hình tốt nhất được chọn bằng RMSE trên validation; bảng dưới chỉ đánh giá trên test, không dùng chọn lại. "
               "Để dự báo tương lai, mô hình được huấn luyện lại trên toàn bộ chuỗi sau khi chốt đánh giá.")
    with st.expander("✅ Kết quả chọn mô hình trên validation"):
        st.dataframe(evaluation["validation_table"], width="stretch", hide_index=True)
    st.markdown("##### 📋 Bảng so sánh mô hình trên tập test")
    st.dataframe(show.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "MAPE (%)": "{:.1f}",
                                    "Cải thiện RMSE so với baseline (%)": "{:+.1f}",
                                    "Validation RMSE": "{:.2f}", "Validation MAE": "{:.2f}"})
                 .highlight_min(subset=["MAE", "RMSE", "MAPE (%)"], color="#d9f2e6"), width="stretch", hide_index=True)

    with st.expander("🔁 Đánh giá cuốn chiếu (rolling origin) – ổn định hơn 1 lần chia"):
        st.caption("Lặp lại việc dự báo 3 tháng tiếp theo tại 6 mốc thời gian khác nhau, rồi lấy trung bình chỉ số.")
        if st.button("Chạy đánh giá cuốn chiếu"):
            with st.spinner("Đang chạy…"):
                rb = fc.rolling_backtest(s, h=3, n_origins=6)
            st.caption("Mô hình không đủ điều kiện ở một mốc (vd SARIMA cần ≥ 36 tháng train) sẽ có số lần đánh giá ít hơn.")
            st.dataframe(rb.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "MAPE (%)": "{:.1f}"})
                         .highlight_min(subset=["MAE", "RMSE"], color="#d9f2e6"), width="stretch", hide_index=True)

    st.markdown("##### 🔮 Dự báo tương lai")
    model = st.selectbox("Mô hình dùng để dự báo", res["Mô hình"].tolist(), index=res["Mô hình"].tolist().index(best))
    resid = (preds["Thực tế"] - preds[model]).std() if model in preds else None
    f = fc.forecast(s, model, future, resid_std=resid)
    fig = go.Figure()
    fig.add_scatter(x=s.index, y=s.values, name="Lịch sử", mode="lines", line=dict(color="#1b1f29"))
    fig.add_scatter(x=f.index.tolist() + f.index[::-1].tolist(), y=f["Cận trên"].tolist() + f["Cận dưới"][::-1].tolist(),
                    fill="toself", fillcolor="rgba(42,120,214,0.15)", line=dict(width=0), name="Khoảng ~95%")
    fig.add_scatter(x=f.index, y=f["Dự báo"], name=f"Dự báo ({model})", mode="lines+markers", line=dict(color=SERIES[0], width=3))
    fig.update_layout(title=f"Dự báo {future} tháng tới – {target}", yaxis_title=unit)
    st.plotly_chart(style_fig(fig, 420), width="stretch")
    st.session_state["fc_future"] = f.assign(**{"Chuỗi": target, "Mô hình": model})
    chg = (f["Dự báo"].mean() / s.iloc[-6:].mean() - 1) * 100 if s.iloc[-6:].mean() else 0
    st.info(f"Theo **{model}**, trung bình {future} tháng tới là **{f['Dự báo'].mean():.1f} {unit}/tháng**, "
            f"{'tăng' if chg >= 0 else 'giảm'} **{abs(chg):.1f}%** so với trung bình 6 tháng gần nhất ({s.iloc[-6:].mean():.1f}).")
    st.caption("Lưu ý: dữ liệu kết thúc ở 06/2023 nên “tương lai” là từ 07/2023. Dự báo chuỗi ngắn (66 điểm) có độ bất định cao; "
               "khoảng dự báo xấp xỉ được ước lượng từ sai số trên tập test (mô tả, không dùng chọn mô hình).")
