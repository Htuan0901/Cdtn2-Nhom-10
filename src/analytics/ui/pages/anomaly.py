"""MODULE 8 - Anomaly Detection bằng Isolation Forest."""
import streamlit as st

from src.analytics import state
from src.analytics.anomaly import detector
from src.analytics.data import schema
from src.analytics.db.database import log_activity
from src.analytics.ui import charts
from src.analytics.ui import components as ui
from src.analytics.ui.theme import BLUE, DANGER, MUTED, NAVY, TEAL


def render():
    ss = st.session_state
    ui.page_header("Phát hiện bất thường", "Isolation Forest đa biến: phát hiện, đánh dấu và xuất các bản ghi bất thường.", "alert")
    df = ui.require_data()
    if df is None:
        return
    nums = schema.numeric_columns(df)
    if not nums:
        ui.callout("Cần ít nhất 1 biến số. Với dữ liệu tuyển dụng hãy tạo đặc trưng lương/kinh nghiệm trước.", "warn")
        return
    default = [c for c in ("salary_mid", "exp_years") if c in nums] or nums[:3]
    c1, c2, c3 = st.columns([3, 1, 1])
    feats = c1.multiselect("Biến đầu vào", nums, default=default, key="an_feats")
    auto = c3.checkbox("contamination = auto", value=False, key="an_auto")
    cont = c2.slider("Contamination", 0.005, 0.20, 0.03, 0.005, key="an_cont", disabled=auto,
                     help="Tỷ lệ bất thường kỳ vọng trong dữ liệu.")
    if ui.button("Huấn luyện Isolation Forest", key="an_run", primary=True, disabled=not feats):
        try:
            with st.spinner("Đang phát hiện bất thường..."):
                res = detector.detect(df, feats, "auto" if auto else cont)
            state.put_result(ss, "anomaly", res)
            log_activity("anomaly", "Isolation Forest", f"{feats}, contamination={res['contamination']}, n={res['n_anomalies']}",
                         ss.get("dataset_id"))
        except ValueError as e:
            st.error(str(e))
    res = state.get_result(ss, "anomaly")
    if res:
        ui.kpi_cards([
            {"label": "Số điểm bất thường", "value": f"{res['n_anomalies']:,}", "icon": "alert", "accent": DANGER},
            {"label": "Tỷ lệ bất thường", "value": f"{res['rate'] * 100:.2f}%", "icon": "hash", "accent": NAVY},
            {"label": "Contamination", "value": str(res["contamination"]), "icon": "settings", "accent": BLUE},
            {"label": "Số biến", "value": len(res["features"]), "icon": "columns", "accent": TEAL},
        ])
        cmap = {"Bình thường": "#93C5FD", "Bất thường": DANGER}
        c1, c2 = st.columns(2)
        with c1:
            ui.plot(charts.pca_scatter(res["projection"], "Trạng thái", "Điểm bất thường trên không gian PCA 2D", color_map=cmap))
        with c2:
            fr = res["frame"]
            if len(res["features"]) >= 2:
                a, b = st.columns(2)
                x = a.selectbox("Trục X", res["features"], key="an_x")
                y = b.selectbox("Trục Y", [f for f in res["features"] if f != x], key="an_y")
                d = fr.assign(**{"Trạng thái": fr["is_anomaly"].map({True: "Bất thường", False: "Bình thường"})})
                fig = charts.scatter(d, x, y, "Trạng thái", f"{y} theo {x}", trend=False)
                for tr in fig.data:
                    if tr.name in cmap:
                        tr.marker.color = cmap[tr.name]
                ui.plot(fig)
            else:
                ui.plot(charts.histogram(fr, "anomaly_score", 50, "Phân phối điểm bất thường"))
        ui.section("So sánh nhóm bất thường và bình thường")
        ui.show_df(res["comparison"].reset_index().rename(columns={"index": "Biến"}))
        ui.insights(detector.describe(res))
        ui.section("Top 50 bản ghi bất thường nhất")
        top = res["frame"].sort_values("anomaly_score", ascending=False)
        ui.show_df(top[top["is_anomaly"]].head(50))
        c1, c2 = st.columns(2)
        with c1:
            ui.download("Xuất toàn bộ dữ liệu kèm nhãn (CSV)", res["frame"].to_csv(index=False).encode("utf-8-sig"),
                        "anomaly_all.csv", "text/csv", key="an_dl_all")
        with c2:
            ui.download("Chỉ xuất các điểm bất thường (CSV)", top[top["is_anomaly"]].to_csv(index=False).encode("utf-8-sig"),
                        "anomalies_only.csv", "text/csv", key="an_dl_only")
    ui.algorithm_card("isolation_forest")
