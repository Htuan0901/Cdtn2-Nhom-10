"""MODULE 5 - Data Mining: tương quan, K-Means (Elbow, Silhouette), PCA, hồ sơ cụm, mẫu đáng chú ý."""
import pandas as pd
import streamlit as st

from src.analytics import state
from src.analytics.analysis import eda
from src.analytics.data import schema
from src.analytics.db.database import log_activity
from src.analytics.mining import clustering
from src.analytics.ui import charts
from src.analytics.ui import components as ui
from src.analytics.ui.theme import BLUE, NAVY, TEAL


def render():
    ss = st.session_state
    ui.page_header("Khai phá dữ liệu", "Phân cụm K-Means, chọn số cụm bằng Elbow & Silhouette, giảm chiều PCA và diễn giải đặc trưng từng cụm.", "layers")
    df = ui.require_data()
    if df is None:
        return
    nums = schema.numeric_columns(df)
    if len(nums) < 2:
        ui.callout("Cần ít nhất 2 biến số để khai phá dữ liệu. Với dữ liệu tuyển dụng hãy tạo đặc trưng (lương, kinh nghiệm) trước.", "warn")
        return
    default = [c for c in ("salary_mid", "exp_years", "salary_min", "salary_max") if c in nums] or nums[:4]
    if {"salary_mid", "salary_min", "salary_max"} <= set(default):
        default = [c for c in default if c not in ("salary_min", "salary_max")]
    feats = st.multiselect("Biến đầu vào", nums, default=default[:4], key="mine_feats",
                           help="Các biến được điền thiếu bằng trung vị và chuẩn hóa StandardScaler trước khi phân cụm.")
    if len(feats) < 2:
        ui.callout("Chọn ít nhất 2 biến.", "info")
        return

    tab1, tab2, tab3 = st.tabs(["Tương quan & mẫu đáng chú ý", "Chọn số cụm (Elbow / Silhouette)", "Kết quả phân cụm & PCA"])
    with tab1:
        corr = eda.correlation_matrix(df, feats, "spearman")
        c1, c2 = st.columns([3, 2])
        with c1:
            if not corr.empty:
                ui.plot(charts.heatmap(corr, "Tương quan Spearman giữa các biến được chọn"))
        with c2:
            ui.section("Mẫu dữ liệu đáng chú ý")
            pats = clustering.notable_patterns(df, feats)
            ui.insights(pats or ["Không có cặp biến tương quan đáng kể (|ρ| ≥ 0,3) hay tỷ lệ ngoại lệ cực đoan > 1%."])

    with tab2:
        kmax = st.slider("Khoảng k tối đa", 4, 12, 8, key="kmax")
        if ui.button("Tính Elbow & Silhouette", key="elbow", primary=True):
            with st.spinner("Đang chạy K-Means cho từng k..."):
                X, _, sampled = clustering.prepare_matrix(df, feats)
                tab = clustering.elbow_silhouette(X, 2, kmax)
            ss["elbow"] = {"tab": tab, "feats": feats, "dataset_id": ss.get("dataset_id"), "sampled": sampled}
        el = ss.get("elbow")
        if el and el["feats"] == feats and el["dataset_id"] == ss.get("dataset_id"):
            tab = el["tab"]
            sug, elb = clustering.suggest_k(tab), clustering.elbow_point(tab)
            ui.plot(charts.elbow_chart(tab, sug))
            ui.show_df(tab.round(4))
            ui.callout(f"Silhouette cao nhất tại <b>k = {sug}</b> ({tab.set_index('k').loc[sug, 'Silhouette']:.3f}); "
                       f"điểm khuỷu (Elbow) của đường inertia tại <b>k = {elb}</b>. "
                       "Khi hai tiêu chí khác nhau, nên ưu tiên k có Silhouette cao và cụm có ý nghĩa nghiệp vụ.", "info")
            if el["sampled"]:
                st.caption("Dữ liệu lớn ⇒ đã lấy mẫu ngẫu nhiên (random_state = 42) để tính toán.")

    with tab3:
        el = ss.get("elbow")
        sug = clustering.suggest_k(el["tab"]) if el and el["feats"] == feats else 3
        k = st.slider("Số cụm k", 2, 12, int(sug), key="k_final")
        if ui.button("Chạy K-Means", key="run_km", primary=True):
            with st.spinner("Đang phân cụm..."):
                res = clustering.run_kmeans(df, feats, k)
            state.put_result(ss, "mining", res)
            log_activity("mining", "K-Means", f"k={k}, features={feats}, silhouette={res['silhouette']:.3f}", ss.get("dataset_id"))
        res = state.get_result(ss, "mining")
        if res:
            ui.kpi_cards([
                {"label": "Số cụm", "value": res["k"], "icon": "layers", "accent": BLUE},
                {"label": "Silhouette", "value": f"{res['silhouette']:.3f}", "icon": "check", "accent": TEAL,
                 "hint": clustering.silhouette_label(res["silhouette"])},
                {"label": "PCA (PC1 + PC2)", "value": f"{sum(res['explained_variance']) * 100:.1f}%", "icon": "trend", "accent": NAVY,
                 "hint": "Tỷ lệ phương sai được giữ lại"},
                {"label": "Số bản ghi phân cụm", "value": f"{res['n']:,}", "icon": "rows", "accent": BLUE,
                 "hint": "Đã lấy mẫu" if res["sampled"] else "Toàn bộ"},
            ])
            c1, c2 = st.columns([3, 2])
            with c1:
                ui.plot(charts.pca_scatter(res["projection"], "Cụm", "Các cụm trên không gian PCA 2D", res["explained_variance"]))
            with c2:
                ui.section("Ý nghĩa các cụm")
                ui.insights(res["descriptions"])
            ui.section("Hồ sơ đặc trưng từng cụm (giá trị trung bình, đơn vị gốc)")
            ui.show_df(res["profile"].reset_index())
            ui.section("Hệ số tải PCA (loadings)")
            ui.show_df(res["loadings"].reset_index().rename(columns={"index": "Biến"}))
            out = df.loc[res["labels"].index].assign(cluster=res["labels"].values)
            ui.download("Tải dữ liệu kèm nhãn cụm (CSV)", out.to_csv(index=False).encode("utf-8-sig"), "clusters.csv", "text/csv")
        ui.algorithm_card("kmeans")
        ui.algorithm_card("pca")
