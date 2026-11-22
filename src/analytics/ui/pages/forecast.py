"""MODULE 7 - Forecasting: dự báo chuỗi thời gian có kiểm định theo thời gian và so sánh baseline."""
import pandas as pd
import streamlit as st

from src.analytics import state
from src.analytics.analysis.eda import FREQ_LABEL, infer_frequency
from src.analytics.data import schema
from src.analytics.db.database import log_activity
from src.analytics.forecasting import forecaster as fc
from src.analytics.ui import charts
from src.analytics.ui import components as ui
from src.analytics.ui.theme import BLUE, NAVY, TEAL


def render():
    ss = st.session_state
    ui.page_header("Dự báo tổng quát (Naive, MA, Holt, ARIMA/SARIMA)", "Chọn mô hình bằng validation RMSE, sau đó mới kiểm định test độc lập; so sánh với baseline Naive.", "trend")
    df = ui.require_data()
    if df is None:
        return
    tcands = list(dict.fromkeys(([ss["time_col"]] if ss.get("time_col") else []) + schema.time_candidates(df)))
    if not tcands:
        ui.callout("<b>Không thể dự báo:</b> dữ liệu không có cột thời gian (ngày/tháng/năm). Dự báo chuỗi thời gian yêu cầu các quan sát "
                   "có thứ tự thời gian. Hãy chọn/chuyển kiểu cột thời gian ở Data Upload Center hoặc trang Làm sạch.", "err")
        return
    nums = schema.numeric_columns(df)
    c1, c2, c3, c4 = st.columns(4)
    tcol = c1.selectbox("Cột thời gian", tcands, key="fc_t")
    vcol = c2.selectbox("Biến cần dự báo", ["(Số bản ghi mỗi kỳ)"] + nums, key="fc_v")
    agg = c3.selectbox("Phép gộp mỗi kỳ", ["mean", "median", "sum"], key="fc_agg", disabled=vcol.startswith("("),
                       format_func={"mean": "Trung bình", "median": "Trung vị", "sum": "Tổng"}.get)
    t = schema.to_time_index(df[tcol])
    base = infer_frequency(t.dropna())
    freqs = [f for f in ["D", "W", "MS", "QS", "YS"] if ["h", "D", "W", "MS", "QS", "YS"].index(f) >= ["h", "D", "W", "MS", "QS", "YS"].index(base)]
    freq = c4.selectbox("Tần suất dữ liệu", freqs, key="fc_f", format_func=FREQ_LABEL.get,
                        help=f"Tần suất gốc nhận diện được: {FREQ_LABEL.get(base, base)}. Chỉ cho phép gộp lên tần suất thô hơn.")
    try:
        info = fc.prepare_series(df, tcol, None if vcol.startswith("(") else vcol, agg, freq)
    except ValueError as e:
        ui.callout(f"<b>Không thể dự báo:</b> {e}", "err")
        return

    n = info.n_periods
    c1, c2, c3, c4 = st.columns(4)
    horizon = c1.number_input("Số kỳ dự báo", 1, 36, 3 if freq != "YS" else 1, key="fc_h")
    max_test = max(1, min(12, n - 3))
    test_size = c2.number_input("Số kỳ kiểm định (test)", 1, max_test, max(1, min(max_test, round(n * 0.2))), key="fc_ts")
    window = c3.number_input("Cửa sổ Moving Average", 2, 24, 3, key="fc_w")
    excl = c4.checkbox("Loại kỳ cuối chưa đầy đủ", value=info.partial_last, key="fc_excl", disabled=not info.partial_last)
    n_eff = n - (1 if excl and info.partial_last else 0)
    val_size = min(int(test_size), n_eff - int(test_size) - 2)
    elig = fc.check_eligibility(n_eff - int(test_size), freq, max(1, val_size))

    ui.kpi_cards([
        {"label": "Số kỳ trong chuỗi", "value": n_eff, "icon": "clock", "accent": BLUE, "hint": FREQ_LABEL.get(freq, freq)},
        {"label": "Kỳ thiếu dữ liệu", "value": info.n_missing_periods, "icon": "missing", "accent": NAVY, "hint": info.fill_method},
        {"label": "Train / Val / Test", "value": f"{n_eff - int(test_size) - val_size} / {val_size} / {int(test_size)}", "icon": "columns", "accent": TEAL,
         "hint": "Chia theo thứ tự thời gian"},
        {"label": "statsmodels", "value": "Có" if fc.has_statsmodels() else "Chưa cài", "icon": "cpu", "accent": BLUE, "hint": "Cần cho ARIMA/SARIMA"},
    ])
    for note in info.notes:
        ui.callout(note, "warn" if "chưa thu thập đủ" in note else "info")

    ui.section("Điều kiện áp dụng mô hình")
    pills = "".join(f'<span class="pill {"ok" if ok else "no"}">{fc.MODEL_LABELS[k]}: {"đủ điều kiện" if ok else "không đủ"}</span>'
                    for k, (ok, _) in elig.items())
    st.markdown(pills, unsafe_allow_html=True)
    for k, (ok, why) in elig.items():
        if not ok:
            st.caption(f"{fc.MODEL_LABELS[k]}: {why}")
    allowed = [k for k, (ok, _) in elig.items() if ok and k != "naive"]
    models = st.multiselect("Mô hình (Naive luôn được chạy làm baseline)", list(fc.MODEL_LABELS)[1:], default=allowed, key="fc_models",
                            format_func=fc.MODEL_LABELS.get)
    if ui.button("Chạy dự báo", key="fc_run", primary=True):
        if n_eff < 4:
            st.error(f"Không dự báo: chuỗi chỉ có {n_eff} kỳ - không đủ điều kiện để vừa huấn luyện vừa kiểm định (cần ≥ 4 kỳ).")
        else:
            with st.spinner("Đang huấn luyện và kiểm định mô hình..."):
                res = fc.run_forecast(info, models, int(horizon), int(test_size), int(window), excl)
            if res["ok"]:
                state.put_result(ss, "forecast", res)
                log_activity("forecast", "Dự báo", f"{info.series.name}, best={res['best']}", ss.get("dataset_id"))
            else:
                st.error(res["reason"])
    res = state.get_result(ss, "forecast")
    if res:
        ui.plot(charts.forecast_chart(res, list(res["results"])))
        c1, c2 = st.columns([3, 2])
        with c1:
            ui.section("Đánh giá trên tập test (mô hình chọn bằng validation)")
            st.caption(f"Đã chọn {fc.MODEL_LABELS[res['best']]} theo RMSE trên {res['validation_size']} kỳ validation; "
                       f"{res['test_size']} kỳ test không tham gia lựa chọn.")
            ui.show_df(res["table"])
            if res["skipped"]:
                st.markdown("**Mô hình không chạy:** " + "; ".join(f"{fc.MODEL_LABELS.get(k, k)} - {v}" for k, v in res["skipped"].items()))
        with c2:
            ui.section("Giải thích kết quả")
            ui.insights(res["explanation"])
        best = res["results"][res["best"]]
        out = pd.DataFrame({"Kỳ": res["future_index"], "Dự báo": best["future"], "Cận dưới 95%": best["future_lo"],
                            "Cận trên 95%": best["future_hi"]})
        ui.section(f"Giá trị dự báo - {best['label']}")
        ui.show_df(out.round({"Dự báo": 4, "Cận dưới 95%": 4, "Cận trên 95%": 4}))
        ui.download("Tải kết quả dự báo (CSV)", out.to_csv(index=False).encode("utf-8-sig"), "forecast.csv", "text/csv")
    for k in ("naive", "moving_average", "linear_trend", "holt", "arima", "sarima"):
        ui.algorithm_card(k)
