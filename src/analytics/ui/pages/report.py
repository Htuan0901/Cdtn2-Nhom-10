"""MODULE 10 - Report Generation: PDF tổng hợp + xuất CSV/PNG riêng lẻ."""
import io
import zipfile

import streamlit as st

from src.analytics import state
from src.analytics.analysis import eda
from src.analytics.data import schema
from src.analytics.db.database import log_activity, log_error
from src.analytics.reporting import pdf_report
from src.analytics.reporting import static_charts as sc
from src.analytics.ui import components as ui


def _ctx():
    ss = st.session_state
    results = {k: state.get_result(ss, k) for k in state.RESULT_KEYS}
    return {"df": ss["df"], "raw_df": ss.get("raw_df"), "meta": ss.get("meta"), "ws_clean_log": ss.get("ws_clean_log"),
            "time_col": ss.get("time_col"), "results": {k: v for k, v in results.items() if v is not None},
            "kaggle": {"fc_result": ss.get("fc_result"), "salary_bundle": ss.get("salary_bundle"), "pipeline_log": ss.get("clean_log")}}


def render(embedded: bool = False):
    ss = st.session_state
    if not embedded:
        ui.page_header("Báo cáo", "Tạo báo cáo PDF từ kết quả thực nghiệm của phiên làm việc; xuất CSV và biểu đồ PNG riêng lẻ.", "report")
    df = ui.require_data()
    if df is None:
        return
    ctx = _ctx()
    done = ctx["results"]
    ui.section("Trạng thái các phần")
    status = {"Làm sạch": bool(ss.get("ws_clean_log")), "Thị trường CNTT / kỹ năng": "skills" in done, "Data Mining": "mining" in done,
              "Machine Learning": "ml" in done, "Dự báo": "forecast" in done, "Bất thường": "anomaly" in done}
    st.markdown("".join(f'<span class="pill {"ok" if v else "no"}">{k}: {"đã có" if v else "chưa chạy"}</span>' for k, v in status.items()),
                unsafe_allow_html=True)
    st.caption("Báo cáo chỉ chứa kết quả đã thực sự chạy; phần chưa chạy sẽ được ghi rõ 'chưa thực hiện' (không tạo số liệu giả).")

    c1, c2 = st.columns([2, 1])
    with c1:
        secs = st.multiselect("Các phần đưa vào báo cáo", list(pdf_report.SECTIONS), default=list(pdf_report.SECTIONS),
                              format_func=pdf_report.SECTIONS.get, key="rp_secs")
    with c2:
        author = st.text_input("Sinh viên thực hiện / ghi chú", "", key="rp_author")
    if ui.button("Tạo báo cáo PDF", key="rp_build", primary=True, disabled=not secs):
        try:
            with st.spinner("Đang dựng biểu đồ và báo cáo PDF..."):
                ss["report_pdf"] = pdf_report.build_report(ctx, secs, author)
                ss["report_pdf_ds"] = ss.get("dataset_id")
            log_activity("report", "Tạo báo cáo PDF", f"{len(secs)} phần", ss.get("dataset_id"))
        except Exception as e:  # noqa: BLE001
            log_error("report", e)
            st.error(f"Không tạo được báo cáo: {e}")
    if ss.get("report_pdf") and ss.get("report_pdf_ds") == ss.get("dataset_id"):
        ui.callout(f"Báo cáo đã sẵn sàng ({len(ss['report_pdf']) / 1024:,.0f} KB).", "ok")
        ui.download("Tải báo cáo PDF", ss["report_pdf"], "bao_cao_phan_tich.pdf", "application/pdf", key="rp_dl")

    ui.section("Xuất dữ liệu (CSV)")
    files = {"du_lieu_hien_tai.csv": df.to_csv(index=False).encode("utf-8-sig")}
    nums = schema.numeric_columns(df)
    if nums:
        files["thong_ke_mo_ta.csv"] = eda.descriptive_stats(df, nums).to_csv(index=False).encode("utf-8-sig")
    if "skills" in done:
        files["top_ky_nang.csv"] = done["skills"]["counts"].to_csv(index=False).encode("utf-8-sig")
        if done["skills"].get("share") is not None:
            files["xu_huong_ky_nang.csv"] = done["skills"]["share"].to_csv().encode("utf-8-sig")
    if "ml" in done:
        files["so_sanh_mo_hinh.csv"] = done["ml"].comparison.to_csv(index=False).encode("utf-8-sig")
    if "forecast" in done:
        f = done["forecast"]
        files["bang_danh_gia_du_bao.csv"] = f["table"].to_csv(index=False).encode("utf-8-sig")
    if "anomaly" in done:
        a = done["anomaly"]["frame"]
        files["bat_thuong.csv"] = a[a["is_anomaly"]].to_csv(index=False).encode("utf-8-sig")
    cols = st.columns(3)
    for i, (name, data) in enumerate(files.items()):
        with cols[i % 3]:
            ui.download(name, data, name, "text/csv", key=f"csv_{name}")

    ui.section("Xuất biểu đồ (PNG)")
    if ui.button("Tạo gói biểu đồ PNG (.zip)", key="rp_png"):
        with st.spinner("Đang vẽ biểu đồ..."):
            pngs = {"thieu_du_lieu.png": sc.missing_bar(df)}
            corr = eda.correlation_matrix(df, nums[:12]) if nums else None
            pngs["tuong_quan.png"] = sc.corr_heatmap(corr) if corr is not None else None
            if nums:
                pngs["phan_phoi.png"] = sc.histogram(df, "salary_mid" if "salary_mid" in nums else nums[0])
            if "skills" in done:
                c = done["skills"]["counts"].head(20)
                pngs["top_ky_nang.png"] = sc.barh(c["Kỹ năng"], c["Tỷ lệ tin (%)"], "Top kỹ năng (% số tin)")
                if done["skills"].get("share") is not None:
                    pngs["xu_huong_ky_nang.png"] = sc.skill_trend(done["skills"]["share"], c["Kỹ năng"].head(6).tolist())
            if "ml" in done:
                r = done["ml"]
                pngs["so_sanh_mo_hinh.png"] = sc.model_comparison(r.comparison, "RMSE" if r.problem == "regression" else "F1 (macro)")
            if "forecast" in done:
                pngs["du_bao.png"] = sc.forecast_plot(done["forecast"])
            if "mining" in done:
                pngs["phan_cum_pca.png"] = sc.pca_clusters(done["mining"]["projection"], done["mining"]["explained_variance"])
            if "anomaly" in done:
                pngs["bat_thuong.png"] = sc.anomaly_scatter(done["anomaly"]["projection"])
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for name, data in pngs.items():
                    if data:
                        z.writestr(name, data)
            ss["png_zip"] = buf.getvalue()
    if ss.get("png_zip"):
        ui.download("Tải gói biểu đồ PNG", ss["png_zip"], "bieu_do.zip", "application/zip", key="png_dl")
