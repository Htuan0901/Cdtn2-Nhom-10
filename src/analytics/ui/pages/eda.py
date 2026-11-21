"""MODULE 4 - EDA tự động: biểu đồ thích nghi kiểu dữ liệu + diễn giải bằng ngôn ngữ tự nhiên."""
import pandas as pd
import streamlit as st

from src.analytics import state
from src.analytics.analysis import eda
from src.analytics.data import schema
from src.analytics.ui import charts
from src.analytics.ui import components as ui


def render():
    ss = st.session_state
    ui.page_header("EDA tự động", "Thống kê mô tả, phân phối, tương quan, phân tích nhóm, thời gian và khu vực.", "chart")
    df = ui.require_data()
    if df is None:
        return
    nums = schema.numeric_columns(df)
    cats = schema.categorical_columns(df, max_unique=60)
    geo = [c for c in df.columns if c in ("province", "region")] + schema.geo_candidates(df)
    geo = list(dict.fromkeys(geo))
    tcands = schema.time_candidates(df)

    tabs = st.tabs(["Thống kê mô tả", "Phân phối", "Quan hệ hai biến", "Tương quan", "Phân tích nhóm", "Theo thời gian", "Theo khu vực"])

    with tabs[0]:
        if not nums:
            ui.callout("Không có biến số để thống kê mô tả.", "info")
        else:
            stats = eda.descriptive_stats(df, nums)
            ui.show_df(stats, height=min(520, 40 + 36 * len(stats)))
            ui.section("Diễn giải")
            ui.insights(eda.describe_numeric_text(stats))
            state.put_result(ss, "eda", {"stats": stats})
        if cats:
            ui.section("Biến phân loại")
            rows = [{"Cột": c, "Số giá trị khác nhau": int(df[c].nunique()), "Giá trị phổ biến nhất (mode)": str(df[c].mode().iloc[0]) if not df[c].mode().empty else "—",
                     "Tần suất mode (%)": round(100 * df[c].value_counts(normalize=True).iloc[0], 2) if df[c].notna().any() else 0}
                    for c in cats]
            ui.show_df(pd.DataFrame(rows))

    with tabs[1]:
        col = st.selectbox("Chọn cột", nums + cats, key="eda_dist")
        if col in nums:
            c1, c2 = st.columns([3, 2])
            bins = c2.slider("Số khoảng (bins)", 10, 100, 40, key="eda_bins")
            with c1:
                ui.plot(charts.histogram(df, col, bins))
            with c2:
                ui.plot(charts.box(df, [col], f"Boxplot {col}"))
            st.markdown(eda.describe_numeric_text(eda.descriptive_stats(df, [col]))[0])
        elif col:
            ft = eda.frequency_table(df, col, top=20)
            c1, c2 = st.columns([3, 2])
            with c1:
                ui.plot(charts.bar(ft[~ft["Giá trị"].str.startswith("(Khác")], "Giá trị", "Tần số", f"Phân phối tần số {col}",
                                   horizontal=True, height=max(320, 24 * len(ft) + 100)))
            with c2:
                ui.show_df(ft)

    with tabs[2]:
        if len(nums) < 2:
            ui.callout("Cần ít nhất 2 biến số cho biểu đồ phân tán.", "info")
        else:
            c1, c2, c3 = st.columns(3)
            x = c1.selectbox("Trục X", nums, key="sc_x")
            y = c2.selectbox("Trục Y", [n for n in nums if n != x], key="sc_y")
            color = c3.selectbox("Tô màu theo", ["(Không)"] + [c for c in cats if df[c].nunique() <= 12], key="sc_c")
            ui.plot(charts.scatter(df, x, y, None if color == "(Không)" else color))
            r = df[[x, y]].corr().iloc[0, 1]
            rs = df[[x, y]].corr(method="spearman").iloc[0, 1]
            st.markdown(f"Pearson r = **{r:.3f}** ({eda.strength_label(r).lower()}), Spearman ρ = **{rs:.3f}**. "
                        "Nếu ρ lớn hơn r rõ rệt ⇒ quan hệ đơn điệu nhưng phi tuyến.")

    with tabs[3]:
        method = st.radio("Phương pháp", ["pearson", "spearman"], horizontal=True, key="corr_m",
                          format_func={"pearson": "Pearson (tuyến tính)", "spearman": "Spearman (thứ hạng, bền với ngoại lệ)"}.get)
        corr = eda.correlation_matrix(df, nums, method)
        if corr.empty:
            ui.callout("Không đủ biến số hợp lệ (cần ≥ 2 cột số có biến thiên, đã loại ID/hằng số/năm). "
                       "Hệ thống không vẽ tương quan cho cột phân loại hoặc văn bản.", "warn")
        else:
            c1, c2 = st.columns([3, 2])
            with c1:
                ui.plot(charts.heatmap(corr, f"Ma trận tương quan {method.title()}"))
            with c2:
                top = eda.top_correlations(corr, 10)
                ui.show_df(top)
                ui.insights(eda.describe_correlation_text(top, method))

    with tabs[4]:
        if not cats:
            ui.callout("Không có cột phân loại phù hợp để nhóm.", "info")
        else:
            c1, c2 = st.columns(2)
            g = c1.selectbox("Nhóm theo", cats, key="gb_g")
            v = c2.selectbox("Chỉ số", ["(Đếm số bản ghi)"] + nums, key="gb_v")
            if v == "(Đếm số bản ghi)":
                tab = eda.group_stats(df, g, None, top=20)
                ui.plot(charts.bar(tab, g, "Số bản ghi", f"Số bản ghi theo {g}", horizontal=True, height=max(320, 24 * len(tab) + 100)))
            else:
                tab = eda.group_stats(df, g, v, top=20)
                ui.plot(charts.box_by_group(df, v, g, 12))
                best = tab.sort_values("Median", ascending=False).iloc[0]
                st.markdown(f"Nhóm có trung vị **{v}** cao nhất: **{best[g]}** ({best['Median']:,.3g}, n = {int(best['Số bản ghi']):,}).")
            ui.show_df(tab)

    with tabs[5]:
        if not tcands and not ss.get("time_col"):
            ui.callout("Không phát hiện cột thời gian ⇒ không thể phân tích theo thời gian. Có thể chuyển kiểu cột ở trang Làm sạch.", "warn")
        else:
            opts = list(dict.fromkeys(([ss["time_col"]] if ss.get("time_col") else []) + tcands))
            c1, c2, c3 = st.columns(3)
            tcol = c1.selectbox("Cột thời gian", opts, key="t_col")
            v = c2.selectbox("Chỉ số", ["(Số bản ghi)"] + nums, key="t_v")
            freq_opts = ["auto", "D", "W", "MS", "QS", "YS"]
            fq = c3.selectbox("Tần suất", freq_opts, key="t_f", format_func=lambda f: "Tự động" if f == "auto" else eda.FREQ_LABEL[f])
            agg = "mean"
            if v != "(Số bản ghi)":
                agg = st.radio("Phép gộp", ["mean", "median", "sum"], horizontal=True, key="t_agg")
            ts, freq = eda.time_aggregate(df, tcol, None if v == "(Số bản ghi)" else v, None if fq == "auto" else fq, agg)
            ui.plot(charts.line(ts, "Thời gian", ts.columns[1], f"{ts.columns[1]} - {eda.FREQ_LABEL.get(freq, freq).lower()}"))
            st.markdown(eda.describe_trend_text(ts.dropna()))
            ui.show_df(ts)

    with tabs[6]:
        if not geo:
            ui.callout("Không phát hiện cột địa lý (vd: tỉnh/thành, địa điểm, khu vực). "
                       "Với dữ liệu tuyển dụng, hãy tạo đặc trưng ở module Thị trường CNTT để có cột province/region.", "info")
        else:
            c1, c2 = st.columns(2)
            gcol = c1.selectbox("Cột địa lý", geo, key="geo_c")
            v = c2.selectbox("Chỉ số", ["(Số bản ghi)"] + nums, key="geo_v")
            if v == "(Số bản ghi)":
                tab = eda.group_stats(df, gcol, None, top=20)
                ui.plot(charts.bar(tab, gcol, "Số bản ghi", f"Top khu vực theo số bản ghi ({gcol})", horizontal=True, height=560))
            else:
                tab = eda.group_stats(df, gcol, v, top=20, min_count=20).sort_values("Median", ascending=False)
                ui.plot(charts.bar(tab, gcol, "Median", f"Trung vị {v} theo {gcol} (nhóm ≥ 20 bản ghi)", horizontal=True, height=560))
            ui.show_df(tab)
