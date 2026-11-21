"""MODULE CHUYÊN NGÀNH - Thị trường tuyển dụng CNTT.

Bước 1: ánh xạ cột + tạo đặc trưng (lương, kinh nghiệm, tỉnh/vùng, tin CNTT, nhóm vị trí).
Bước 2: trích xuất kỹ năng từ văn bản (text mining) → top kỹ năng, xu hướng, dự báo, đồng xuất hiện, chênh lệch lương.
"""
import numpy as np
import pandas as pd
import streamlit as st

from src.analytics import state
from src.analytics.data import schema
from src.analytics.data.cleaning import CleaningStep
from src.analytics.db.database import log_activity
from src.analytics.domain import jobs, skills
from src.analytics.ui import charts
from src.analytics.ui import components as ui
from src.analytics.ui.theme import BLUE, NAVY, TEAL, WARNING


def _mapping_form(df):
    ss = st.session_state
    auto = ss.get("job_mapping") or jobs.auto_map_columns(df)
    opts = ["(Không có)"] + list(df.columns)
    with st.form("map_form"):
        st.markdown("**Ánh xạ cột** - hệ thống tự gợi ý dựa trên tên và nội dung cột, bạn có thể sửa lại:")
        cols = st.columns(4)
        mapping = {}
        for i, (fld, label) in enumerate(jobs.FIELD_LABELS.items()):
            cur = auto.get(fld)
            mapping[fld] = cols[i % 4].selectbox(label, opts, index=opts.index(cur) if cur in opts else 0, key=f"map_{fld}")
        it_only = st.checkbox("Chỉ giữ lại tin tuyển dụng ngành CNTT (khuyến nghị cho đề tài)", value=True)
        ok = st.form_submit_button("Tạo đặc trưng tuyển dụng", type="primary")
    mapping = {k: (None if v == "(Không có)" else v) for k, v in mapping.items()}
    return ok, mapping, it_only


def render():
    ss = st.session_state
    ui.page_header("Thị trường tuyển dụng (dữ liệu Việt Nam & CSV tuyển dụng)", "Chuẩn hóa tin tuyển dụng, trích xuất kỹ năng từ mô tả công việc, "
                   "phân tích lương - khu vực và dự báo xu hướng kỹ năng.", "briefcase")
    df = ui.require_data()
    if df is None:
        return
    if not jobs.is_job_dataset(df) and "salary_mid" not in df.columns:
        ui.callout("Dataset hiện tại không giống dữ liệu tin tuyển dụng (cần tối thiểu cột tiêu đề công việc + 2 trong các cột lương, "
                   "địa điểm, kinh nghiệm, mô tả, yêu cầu). Bạn vẫn có thể ánh xạ cột thủ công bên dưới.", "warn")

    built = "salary_mid" in df.columns or "is_it" in df.columns
    with st.expander("Bước 1 - Tạo đặc trưng tuyển dụng" + (" (đã thực hiện)" if built else ""), expanded=not built):
        ok, mapping, it_only = _mapping_form(df)
        if ok:
            if not mapping.get("title"):
                st.error("Cần chọn cột Tiêu đề công việc.")
            else:
                with st.spinner("Đang chuẩn hóa lương, kinh nghiệm, địa điểm và nhận diện tin CNTT..."):
                    new_df, rep = jobs.build_job_features(df, mapping, it_only)
                step = CleaningStep("Tạo đặc trưng tuyển dụng",
                                    f"Tạo {len(rep.created_columns)} cột: {', '.join(rep.created_columns)}. " + " ".join(rep.notes),
                                    rep.n_input, rep.n_output, rep.n_output * len(rep.created_columns), rep.created_columns)
                state.apply_step(ss, new_df, step)
                ss["job_mapping"] = mapping
                ss["job_report"] = rep.__dict__
                if mapping.get("time") and not ss.get("time_col"):
                    ss["time_col"] = mapping["time"]
                log_activity("jobs", "Tạo đặc trưng tuyển dụng", step.detail, ss.get("dataset_id"))
                st.rerun()
        rep = ss.get("job_report")
        if rep and built:
            st.markdown(f"Tin đầu vào: **{rep['n_input']:,}** · Tin CNTT nhận diện: **{rep['n_it']:,}** · Tin sau xử lý: **{rep['n_output']:,}** · "
                        f"Kinh nghiệm đọc được: **{rep['exp_parsed_pct']}%** · Địa phương xác định được: **{rep['province_known_pct']}%**")
            if rep.get("salary_status"):
                st.markdown("Trạng thái lương: " + ", ".join(f"`{k}`: {v:,}" for k, v in rep["salary_status"].items()))
            for n in rep.get("notes", []):
                st.caption(n)

    if not built:
        ui.callout("Hãy thực hiện Bước 1 để mở khóa các phân tích thị trường tuyển dụng.", "info")
        return

    mapping = ss.get("job_mapping") or jobs.auto_map_columns(df)
    sk = state.get_result(ss, "skills")
    with st.expander("Bước 2 - Trích xuất kỹ năng từ văn bản (text mining)" + (" (đã thực hiện)" if sk else ""), expanded=sk is None):
        text_default = [c for c in (mapping.get("title"), mapping.get("description"), mapping.get("requirements")) if c and c in df.columns]
        text_opts = [c for c in df.columns if schema.is_text_dtype(df[c])]
        text_cols = st.multiselect("Cột văn bản dùng để trích xuất", text_opts, default=[c for c in text_default if c in text_opts])
        tcol = ss.get("time_col") or mapping.get("time")
        if ui.button("Trích xuất kỹ năng", key="extract", primary=True, disabled=not text_cols):
            with st.spinner(f"Đang trích xuất kỹ năng từ {len(df):,} tin..."):
                sets = skills.extract_skills_frame(df, text_cols)
                counts = skills.skill_counts(sets)
                top = counts["Kỹ năng"].head(20).tolist()
                share = fcst = None
                if tcol and tcol in df.columns and top:
                    t = schema.to_time_index(df[tcol])
                    period = t.dt.year if t.notna().any() else df[tcol]
                    share = skills.skill_share_by_period(sets, period, top)
                    fcst = skills.forecast_skill_shares(share, top[:15])
                sal = df["salary_mid"] if "salary_mid" in df.columns else pd.Series(np.nan, index=df.index)
                res = {"counts": counts, "share": share, "forecast": fcst, "n_postings": len(df), "text_cols": text_cols,
                       "no_skill_pct": round(100 * (sets.map(len) == 0).mean(), 2), "cooc": skills.cooccurrence(sets, top[:15]),
                       "premium": skills.salary_premium(sets, sal, top), "sets": sets, "time_col": tcol}
                state.put_result(ss, "skills", res)
                log_activity("jobs", "Trích xuất kỹ năng", f"{len(df)} tin, {len(counts)} kỹ năng", ss.get("dataset_id"))
            st.rerun()
        ui.algorithm_card("skill_extraction")

    _overview(df)
    if sk is None:
        return
    tabs = st.tabs(["Top kỹ năng", "Xu hướng & dự báo kỹ năng", "Lương", "Đồng xuất hiện kỹ năng"])
    with tabs[0]:
        _top_skills(sk)
    with tabs[1]:
        _skill_trend(sk)
    with tabs[2]:
        _salary(df, sk)
    with tabs[3]:
        if sk["cooc"] is not None and not sk["cooc"].empty:
            co = sk["cooc"]
            diag = np.diag(co.values).astype(float)
            cond = (co.values / np.where(diag[:, None] == 0, np.nan, diag[:, None])) * 100   # P(cột | hàng)
            mat = pd.DataFrame(cond, index=co.index, columns=co.columns).round(1)
            ui.plot(charts.heatmap(mat, "P(kỹ năng cột | kỹ năng hàng) - %", diverging=False, fmt=".0f"))
            st.caption("Đọc theo hàng: trong các tin yêu cầu kỹ năng ở hàng, bao nhiêu % đồng thời yêu cầu kỹ năng ở cột. "
                       "Giúp phát hiện 'bộ kỹ năng' thường đi cùng nhau (vd: React ↔ JavaScript).")


def _overview(df):
    ui.section("Tổng quan thị trường")
    items = [{"label": "Số tin phân tích", "value": ui.fmt_num(len(df)), "icon": "briefcase", "accent": BLUE}]
    if "salary_mid" in df:
        s = df["salary_mid"].dropna()
        items.append({"label": "Lương trung vị", "value": f"{s.median():.1f} tr" if len(s) else "—", "icon": "money", "accent": TEAL,
                      "hint": f"IQR: {s.quantile(.25):.1f} - {s.quantile(.75):.1f} triệu/tháng" if len(s) else ""})
        items.append({"label": "Tin có công bố lương", "value": f"{100 * df['salary_mid'].notna().mean():.1f}%", "icon": "check", "accent": NAVY})
    if "exp_years" in df:
        items.append({"label": "Không yêu cầu kinh nghiệm", "value": f"{100 * (df['exp_years'] == 0).mean():.1f}%", "icon": "clock",
                      "accent": WARNING, "hint": "Cơ hội cho sinh viên mới tốt nghiệp"})
    ui.kpi_cards(items)
    c1, c2 = st.columns(2)
    if "job_group" in df:
        with c1:
            vc = df["job_group"].value_counts().head(15).rename_axis("Nhóm vị trí").reset_index(name="Số tin")
            ui.plot(charts.bar(vc, "Nhóm vị trí", "Số tin", "Cơ cấu nhóm vị trí", horizontal=True, height=460))
    if "province" in df:
        with c2:
            vc = df["province"].value_counts().head(12).rename_axis("Tỉnh/Thành").reset_index(name="Số tin")
            ui.plot(charts.bar(vc, "Tỉnh/Thành", "Số tin", "Top địa phương tuyển dụng", horizontal=True, height=460))


def _top_skills(sk):
    counts = sk["counts"]
    if counts.empty:
        ui.callout("Không trích xuất được kỹ năng nào từ các cột văn bản đã chọn.", "warn")
        return
    n = st.slider("Số kỹ năng hiển thị", 10, 40, 20, key="sk_n")
    c1, c2 = st.columns([3, 2])
    with c1:
        ui.plot(charts.bar(counts.head(n), "Kỹ năng", "Tỷ lệ tin (%)", f"Top {n} kỹ năng/công nghệ được yêu cầu (% số tin)",
                           horizontal=True, color="Nhóm", height=max(420, 22 * n + 120), text=False))
    with c2:
        cat = counts.groupby("Nhóm")["Số tin"].sum().sort_values(ascending=False)
        ui.plot(charts.pie(cat.index, cat.values, "Tỷ trọng lượt nhắc theo nhóm kỹ năng"))
        ui.callout(f"{sk['no_skill_pct']}% số tin không trích xuất được kỹ năng nào trong từ điển "
                   "(thường là vị trí hỗ trợ/không chuyên sâu hoặc mô tả quá ngắn) - đây là giới hạn của phương pháp từ điển.", "info")
    ui.show_df(counts, height=360)


def _skill_trend(sk):
    share = sk.get("share")
    if share is None:
        ui.callout("Cần cột thời gian để phân tích xu hướng kỹ năng (chọn ở Data Upload Center hoặc ánh xạ cột Thời gian).", "warn")
        return
    cols = [c for c in share.columns if c != "Số tin"]
    pick = st.multiselect("Kỹ năng", cols, default=cols[:6], key="sk_trend")
    d = share.reset_index().rename(columns={share.index.name or "index": "Kỳ"})
    d["Kỳ"] = d["Kỳ"].astype(str)
    if pick:
        ui.plot(charts.line(d, "Kỳ", pick, "Tỷ lệ tin yêu cầu kỹ năng theo kỳ (%)", height=420))
    counts_per = share["Số tin"]
    ui.callout("Dùng <b>tỷ lệ %</b> số tin trong mỗi kỳ (thay vì số lượng tuyệt đối) vì số tin thu thập mỗi năm khác nhau do nguồn dữ liệu; "
               "chuẩn hóa giúp so sánh công bằng giữa các kỳ. Số tin mỗi kỳ: " + ", ".join(f"{k}: {int(v):,}" for k, v in counts_per.items()), "info")
    fc = sk.get("forecast")
    ui.section("Dự báo tỷ lệ kỹ năng kỳ kế tiếp", "Linear Trend so với baseline Naive")
    if not fc or not fc.get("ok"):
        ui.callout(fc.get("reason") if fc else "Không đủ dữ liệu để dự báo.", "warn")
        return
    better = fc["better"]
    ui.kpi_cards([
        {"label": "MAE Linear Trend", "value": f"{fc['mae_linear']:.2f} đ%", "icon": "trend", "accent": BLUE,
         "hint": f"Backtest kỳ {fc['test_period']}"},
        {"label": "MAE Naive (baseline)", "value": f"{fc['mae_naive']:.2f} đ%", "icon": "chart", "accent": NAVY},
        {"label": "RMSE Linear / Naive", "value": f"{fc['rmse_linear']:.2f} / {fc['rmse_naive']:.2f}", "icon": "hash", "accent": TEAL},
        {"label": "Mô hình tốt hơn", "value": f"<span style='font-size:1.1rem'>{better}</span>", "icon": "check", "accent": TEAL},
    ])
    if better == "Naive":
        ui.callout("Trên kỳ kiểm định, <b>Naive tốt hơn Linear Trend</b> ⇒ xu hướng tuyến tính chưa đủ tin cậy để ngoại suy; "
                   "nguyên nhân thường do chuỗi ngắn (ít kỳ) và có điểm gãy cấu trúc (thay đổi nguồn dữ liệu giữa các năm). "
                   "Kết quả dự báo bên dưới chỉ nên dùng như tín hiệu định hướng.", "warn")
    else:
        ui.callout("Linear Trend có sai số backtest thấp hơn Naive ⇒ có bằng chứng xu hướng tuyến tính mang thông tin dự báo, "
                   "nhưng tập kiểm định chỉ gồm 1 kỳ nên cần thận trọng.", "ok")
    ui.show_df(fc["table"])
    st.caption("đ% = điểm phần trăm. Backtest: huấn luyện trên các kỳ trước, dự báo kỳ cuối, so sánh với giá trị thực tế. "
               "Dự báo kỳ tới dùng toàn bộ chuỗi, giới hạn trong [0, 100]%.")
    ui.algorithm_card("linear_trend")


def _salary(df, sk):
    if "salary_mid" not in df:
        ui.callout("Chưa có cột lương chuẩn hóa.", "info")
        return
    groups = [c for c in ("job_group", "exp_bucket", "province", "region") if c in df]
    g = st.selectbox("So sánh lương theo", groups, format_func={"job_group": "Nhóm vị trí", "exp_bucket": "Kinh nghiệm",
                                                               "province": "Tỉnh/Thành", "region": "Vùng miền"}.get, key="sal_g")
    d = df.dropna(subset=["salary_mid"])
    d = d[d["salary_mid"] <= d["salary_mid"].quantile(0.99)]      # hiển thị: bỏ 1% đuôi trên để biểu đồ dễ đọc
    if g == "exp_bucket":
        order = [b for b in jobs.EXP_ORDER if b in d[g].unique()]
        tab = d.groupby(g)["salary_mid"].agg(["count", "median", "mean"]).reindex(order).reset_index()
    else:
        tab = d.groupby(g)["salary_mid"].agg(["count", "median", "mean"]).query("count >= 20").sort_values("median", ascending=False).head(15).reset_index()
    tab.columns = [g, "Số tin", "Trung vị (triệu)", "Trung bình (triệu)"]
    c1, c2 = st.columns([3, 2])
    with c1:
        ui.plot(charts.box_by_group(d[d[g].isin(tab[g])], "salary_mid", g, 15, "Phân phối lương (triệu VND/tháng)"))
    with c2:
        ui.show_df(tab.round(2))
        st.caption("Ẩn 1% tin lương cao nhất khi vẽ để tránh ngoại lệ làm méo trục; bảng chỉ gồm nhóm ≥ 20 tin.")
    prem = sk.get("premium")
    ui.section("Chênh lệch lương theo kỹ năng", "Trung vị lương tin CÓ vs KHÔNG yêu cầu kỹ năng")
    if prem is not None and not prem.empty:
        ui.plot(charts.bar(prem, "Kỹ năng", "Chênh lệch (triệu)", "Chênh lệch trung vị lương (triệu VND/tháng)", horizontal=True,
                           height=max(380, 24 * len(prem) + 100)))
        ui.callout("Đây là <b>tương quan, không phải nhân quả</b>: kỹ năng có chênh lệch cao thường xuất hiện ở vị trí cấp cao hơn "
                   "(yếu tố gây nhiễu là kinh nghiệm/cấp bậc). Có thể kiểm chứng bằng mô hình ML có kiểm soát biến kinh nghiệm.", "warn")
        ui.show_df(prem)
