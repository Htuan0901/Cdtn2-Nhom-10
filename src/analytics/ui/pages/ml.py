"""MODULE 6 - Machine Learning: xác định bài toán, chống rò rỉ, huấn luyện nhiều mô hình, so sánh với baseline, lưu & dự đoán."""
import numpy as np
import pandas as pd
import streamlit as st

from src.analytics import state
from src.analytics.data import schema
from src.analytics.db.database import log_activity, log_training
from src.analytics.ml import trainer
from src.analytics.ui import charts
from src.analytics.ui import components as ui
from src.analytics.ui.theme import BLUE, NAVY, TEAL, WARNING


def render():
    ss = st.session_state
    ui.page_header("Machine Learning", "Huấn luyện, đánh giá và so sánh các mô hình hồi quy/phân loại với baseline, kiểm soát rò rỉ dữ liệu.", "cpu")
    df = ui.require_data()
    if df is None:
        return
    targets = schema.target_candidates(df)
    if not targets:
        ui.callout("Không có cột phù hợp làm biến mục tiêu.", "warn")
        return
    pref = ss.get("target_col") if ss.get("target_col") in targets else ("salary_mid" if "salary_mid" in targets else targets[0])
    c1, c2 = st.columns([2, 3])
    with c1:
        # khóa widget phụ thuộc tập cột ⇒ khi tạo thêm đặc trưng mới, lựa chọn mặc định được tính lại
        target = st.selectbox("Biến mục tiêu", targets, index=targets.index(pref) if pref in targets else 0,
                              key=f"ml_target_{df.shape[1]}")
        auto, why = trainer.detect_problem_type(df[target])
        st.caption(f"Tự động nhận diện: **{'Hồi quy' if auto == 'regression' else 'Phân loại'}** - {why}")
        problem = st.radio("Xác nhận loại bài toán", ["regression", "classification"], index=0 if auto == "regression" else 1,
                           format_func={"regression": "Hồi quy (Regression)", "classification": "Phân loại (Classification)"}.get,
                           horizontal=True, key="ml_problem")
    usable, excluded = trainer.usable_features(df, target)
    suspects = trainer.leakage_check(df, usable, target)
    default_feats = [f for f in usable if f not in suspects][:15]
    with c2:
        feats = st.multiselect("Biến đầu vào (features)", usable, default=default_feats, key=f"ml_feats_{target}")
        if suspects:
            ui.callout("<b>Cảnh báo rò rỉ dữ liệu</b> - các cột sau đã được bỏ chọn mặc định vì có thể chứa thông tin của biến mục tiêu: "
                       + "; ".join(f"<code>{k}</code> ({v})" for k, v in suspects.items()), "warn")
        if excluded:
            with st.expander(f"{len(excluded)} cột bị loại khỏi danh sách đặc trưng và lý do"):
                ui.show_df(pd.DataFrame({"Cột": list(excluded), "Lý do": list(excluded.values())}))

    catalog = list(trainer.model_catalog(problem, False))
    c3, c4, c5 = st.columns([3, 1, 1])
    models = c3.multiselect("Mô hình", catalog, default=catalog, key=f"ml_models_{problem}")
    test_size = c4.slider("Tỷ lệ tập test", 0.1, 0.4, 0.2, 0.05, key="ml_ts")
    imb = c5.selectbox("Mất cân bằng lớp", ["auto", "balanced", "none"], key="ml_imb", disabled=problem == "regression",
                       format_func={"auto": "Tự động", "balanced": "Luôn cân bằng", "none": "Không xử lý"}.get)
    if problem == "regression" and not trainer.HAS_XGB:
        st.caption("XGBoost chưa được cài - bỏ qua XGBoost Regressor (pip install xgboost để bật).")

    if ui.button("Huấn luyện & đánh giá", key="ml_train", primary=True, disabled=not (feats and models)):
        bar = st.progress(0.0, text="Chuẩn bị dữ liệu...")
        try:
            res = trainer.train_models(df, target, feats, problem, models, test_size, imb,
                                       progress=lambda p, t: bar.progress(min(p, 1.0), text=t))
            state.put_result(ss, "ml", res)
            for name in models:
                if name in res.details:
                    log_training(ss.get("dataset_id"), problem, target, name, res.details[name]["metrics"], name == res.best_name)
            log_activity("ml", "Huấn luyện mô hình", f"target={target}, best={res.best_name}", ss.get("dataset_id"))
            bar.empty()
        except ValueError as e:
            bar.empty()
            st.error(str(e))
    res = state.get_result(ss, "ml")
    if not res:
        ui.algorithm_card("linear_regression" if problem == "regression" else "logistic_regression")
        return
    _results(res)


def _results(res):
    ss = st.session_state
    ui.section("Kết quả so sánh mô hình", f"Train: {res.n_train:,} · Test: {res.n_test:,} · Biến mục tiêu: {res.target}")
    comp = res.comparison
    ui.show_df(comp.style.apply(lambda r: ["background-color:#E6F6F4; font-weight:600" if r["Mô hình"] == res.best_name else "" for _ in r], axis=1))
    ui.callout(f"<b>Mô hình tốt nhất: {res.best_name}</b>. {res.best_reason}", "ok")
    for n in res.notes:
        ui.callout(n, "info")
    metric = "RMSE" if res.problem == "regression" else "F1 (macro)"
    plot_df = comp.dropna(subset=[metric])
    c1, c2 = st.columns(2)
    with c1:
        ui.plot(charts.bar(plot_df, "Mô hình", metric, f"{metric} trên tập test (gồm baseline)", horizontal=True))
    with c2:
        second = "R2" if res.problem == "regression" else "Accuracy"
        ui.plot(charts.bar(plot_df, "Mô hình", second, f"{second} trên tập test", horizontal=True))
    if "CV mean" in comp:
        st.caption(f"CV mean/std: {trainer.CV_FOLDS}-fold cross-validation trên tập train (tối đa 8.000 dòng) - "
                   f"{'RMSE' if res.problem == 'regression' else 'F1-macro'}; độ lệch chuẩn nhỏ ⇒ mô hình ổn định.")

    ui.section("Chi tiết mô hình")
    names = [n for n in res.details]
    pick = st.selectbox("Mô hình", names, index=names.index(res.best_name), key="ml_detail")
    det = res.details[pick]
    m = det["metrics"]
    if res.problem == "regression":
        ui.kpi_cards([{"label": k, "value": f"{m[k]:,.4g}", "icon": "hash", "accent": BLUE if k != "R2" else TEAL,
                       "hint": f"Train: {det['train_metrics'][k]:,.4g}"} for k in ("MAE", "MSE", "RMSE", "R2")])
        c1, c2 = st.columns(2)
        with c1:
            ui.plot(charts.actual_vs_pred(det["residual_sample"]))
        with c2:
            _importance(det)
    else:
        ui.kpi_cards([{"label": k, "value": "—" if pd.isna(m[k]) else f"{m[k]:.3f}", "icon": "check", "accent": TEAL}
                      for k in ("Accuracy", "Precision", "Recall", "F1 (macro)", "ROC-AUC")])
        c1, c2 = st.columns(2)
        with c1:
            ui.plot(charts.confusion(det["confusion"]))
        with c2:
            if det.get("y_score") is not None and len(res.classes) == 2:
                ui.plot(charts.roc_curve_fig(det["y_true"], det["y_score"], res.classes[1]))
                st.caption(f"Lớp dương (positive class): {res.classes[1]}")
            else:
                _importance(det)
        if det.get("y_score") is not None and len(res.classes) == 2:
            _importance(det)
    ui.algorithm_card({"Linear Regression": "linear_regression", "Logistic Regression": "logistic_regression"}.get(
        pick, "random_forest" if "Random" in pick else ("decision_tree" if "Tree" in pick else "gradient_boosting")))

    ui.section("Lưu mô hình & dự đoán")
    c1, c2 = st.columns([1, 2])
    with c1:
        if ui.button(f"Lưu mô hình {pick}", key="ml_save", primary=True):
            path = trainer.save_model(res, ss.get("dataset_id") or "dataset", pick)
            log_activity("ml", "Lưu mô hình", path, ss.get("dataset_id"))
            st.success(f"Đã lưu: {path}")
    with c2:
        with st.form("predict_form"):
            st.markdown(f"**Dự đoán thử với {pick}**")
            df = ss["df"]
            vals = {}
            cols = st.columns(3)
            for i, f in enumerate(res.features[:12]):
                s = df[f]
                if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                    vals[f] = cols[i % 3].number_input(f, value=float(np.nan_to_num(s.median())), key=f"pred_{f}")
                else:
                    opts = s.dropna().astype(str).value_counts().head(30).index.tolist() or [""]
                    vals[f] = cols[i % 3].selectbox(f, opts, key=f"pred_{f}")
            for f in res.features[12:]:
                vals[f] = np.nan
            go = st.form_submit_button("Dự đoán", type="primary")
        if go:
            pred, proba = trainer.predict_one(res.pipelines[pick], res.features, vals)
            if res.problem == "regression":
                st.success(f"Giá trị dự đoán {res.target}: **{pred:,.3f}**")
            else:
                st.success(f"Lớp dự đoán: **{pred}**")
                if proba:
                    ui.show_df(pd.DataFrame({"Lớp": list(proba), "Xác suất": list(proba.values())}))


def _importance(det):
    imp = det.get("importance")
    if imp is not None and not imp.empty:
        ui.plot(charts.bar(imp.head(15), "Đặc trưng", "Mức quan trọng", "Mức độ quan trọng của đặc trưng (chuẩn hóa)",
                           horizontal=True, height=460, text=False))
    else:
        st.caption("Mô hình không cung cấp độ quan trọng đặc trưng.")
