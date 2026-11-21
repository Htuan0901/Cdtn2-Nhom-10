"""MODULE 3 - Data Cleaning: mọi thao tác đều do người dùng chọn, xem trước tác động và được ghi nhật ký."""
import pandas as pd
import streamlit as st

from src.analytics import state
from src.analytics.data import cleaning as cl
from src.analytics.data import schema
from src.analytics.db.database import log_activity
from src.analytics.ui import charts
from src.analytics.ui import components as ui
from src.analytics.ui.theme import BLUE, DANGER, TEAL, WARNING


def _apply(new_df, step):
    ss = st.session_state
    state.apply_step(ss, new_df, step)
    log_activity("cleaning", step.action, step.detail, ss.get("dataset_id"))
    ss["clean_flash"] = f"{step.action}: {step.detail} (dòng: {step.rows_before:,} → {step.rows_after:,})"
    st.rerun()


def render():
    ss = st.session_state
    ui.page_header("Làm sạch tương tác", "Kiểm tra chất lượng, xử lý thiếu, trùng lặp, ngoại lệ, chuẩn hóa và chuyển đổi kiểu dữ liệu.", "clean")
    df = ui.require_data()
    if df is None:
        return
    if ss.get("clean_flash"):
        ui.callout("Đã áp dụng - " + ss.pop("clean_flash"), "ok")

    q = cl.quality_summary(df)
    ui.kpi_cards([
        {"label": "Số dòng", "value": ui.fmt_num(q["rows"]), "icon": "rows", "accent": BLUE},
        {"label": "Ô thiếu", "value": ui.fmt_num(q["missing_cells"]), "icon": "missing", "accent": WARNING, "hint": f"{q['missing_pct']}%"},
        {"label": "Dòng trùng lặp", "value": ui.fmt_num(q["duplicates"]), "icon": "copy", "accent": DANGER, "hint": "Bỏ qua cột ID"},
        {"label": "Bước đã áp dụng", "value": len(ss.get("ws_clean_log") or []), "icon": "check", "accent": TEAL},
    ])

    tabs = st.tabs(["Giá trị thiếu", "Trùng lặp", "Ngoại lệ", "Kiểu dữ liệu & ngày tháng", "Chuẩn hóa",
                    "Giá trị không hợp lệ", "Trước / Sau & Nhật ký"])
    all_cols = list(df.columns)
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]

    # ------------------------------------------------------------ Missing
    with tabs[0]:
        mt = cl.missing_table(df)
        has_miss = mt[mt["Số ô thiếu"] > 0]
        c1, c2 = st.columns([3, 2])
        with c1:
            ui.section("Thống kê giá trị thiếu theo cột")
            if has_miss.empty:
                ui.callout("Dữ liệu không có giá trị thiếu.", "ok")
            else:
                ui.plot(charts.bar(has_miss.head(20), "Cột", "Tỷ lệ (%)", "Tỷ lệ thiếu (%) theo cột", horizontal=True,
                                   height=max(300, 26 * min(len(has_miss), 20) + 100)))
        with c2:
            ui.section("Xử lý")
            if not has_miss.empty:
                cols = st.multiselect("Cột cần xử lý", has_miss["Cột"].tolist(), default=has_miss["Cột"].tolist()[:1], key="mv_cols")
                methods = {"median": "Median (trung vị)", "mean": "Mean (trung bình)", "mode": "Mode (giá trị phổ biến nhất)",
                           "constant": "Hằng số tự nhập", "ffill": "Forward fill (dữ liệu chuỗi thời gian)", "drop_rows": "Xóa các dòng bị thiếu"}
                if cols:
                    sug = {c: cl.suggest_impute_method(df[c]) for c in cols}
                    st.caption("Gợi ý: " + "; ".join(f"{c} → {v}" for c, v in sug.items()) +
                               " (số lệch mạnh dùng median, số gần đối xứng dùng mean, phân loại dùng mode).")
                m = st.selectbox("Phương pháp", list(methods), format_func=methods.get, key="mv_method")
                const = st.text_input("Giá trị hằng số", "0", key="mv_const") if m == "constant" else None
                if cols:
                    affected = int(df[cols].isna().any(axis=1).sum()) if m == "drop_rows" else int(df[cols].isna().sum().sum())
                    unit = "dòng sẽ bị xóa" if m == "drop_rows" else "ô sẽ được điền"
                    ui.callout(f"Tác động dự kiến: <b>{affected:,}</b> {unit}.", "warn" if m == "drop_rows" else "info")
                    if ui.button("Áp dụng xử lý thiếu", key="mv_apply", primary=True):
                        _apply(*cl.impute(df, cols, m, const))
        ui.show_df(mt, height=300)

    # ------------------------------------------------------------ Duplicates
    with tabs[1]:
        ids = schema.id_columns(df)
        ignore = st.checkbox(f"Bỏ qua cột ID khi so sánh ({', '.join(ids) if ids else 'không có cột ID'})", value=True, key="dup_ign")
        n_dup = cl.count_duplicates(df, ignore)
        c1, c2 = st.columns([3, 2])
        with c1:
            ui.callout(f"Phát hiện <b>{n_dup:,}</b> dòng trùng lặp ({100 * n_dup / max(len(df), 1):.2f}%).", "warn" if n_dup else "ok")
            if n_dup:
                subset = cl.duplicate_subset(df, ignore)
                dmask = df.duplicated(subset=subset, keep=False)
                st.caption("Ví dụ các nhóm bản ghi trùng (sắp xếp để thấy các bản trùng nằm cạnh nhau):")
                sample = df[dmask].head(400)
                ui.show_df(sample.sort_values(by=list(sample.columns[1:3]) if sample.shape[1] > 2 else list(sample.columns)).head(20))
        with c2:
            keep = st.radio("Giữ lại bản ghi", ["first", "last"], format_func={"first": "Bản xuất hiện đầu tiên", "last": "Bản cuối cùng"}.get,
                            key="dup_keep", horizontal=True)
            if n_dup and ui.button(f"Xóa {n_dup:,} dòng trùng lặp", key="dup_apply", primary=True):
                _apply(*cl.drop_duplicates(df, ignore, keep))

    # ------------------------------------------------------------ Outliers
    with tabs[2]:
        if not nums:
            ui.callout("Không có cột số để phát hiện ngoại lệ.", "info")
        else:
            c1, c2, c3 = st.columns(3)
            cols = c1.multiselect("Cột số", nums, default=nums[:min(3, len(nums))], key="out_cols")
            method = c2.radio("Phương pháp", ["iqr", "zscore"], format_func={"iqr": "IQR", "zscore": "Z-score"}.get, horizontal=True, key="out_m")
            k = c3.slider("Hệ số k (IQR)", 1.0, 3.0, 1.5, 0.1, key="out_k") if method == "iqr" else 1.5
            z = c3.slider("Ngưỡng |z|", 2.0, 4.0, 3.0, 0.1, key="out_z") if method == "zscore" else 3.0
            if cols:
                rep = cl.detect_outliers(df, cols, method, k, z)
                ui.show_df(rep)
                ui.plot(charts.box(df, cols, "Boxplot các cột được chọn", normalize=len(cols) > 1))
                action = st.radio("Hành động", ["keep", "cap", "remove"], horizontal=True, key="out_act",
                                  format_func={"keep": "Giữ lại", "cap": "Giới hạn về cận (capping)", "remove": "Loại bỏ dòng"}.get)
                total = int(rep["Số ngoại lệ"].sum())
                if action == "remove":
                    lo_hi = [cl.outlier_bounds(df[c], method, k, z) for c in cols]
                    mask = pd.Series(False, index=df.index)
                    for c, (lo, hi) in zip(cols, lo_hi):
                        x = pd.to_numeric(df[c], errors="coerce")
                        mask |= ((x < lo) | (x > hi)).fillna(False)
                    ui.callout(f"Tác động: <b>{int(mask.sum()):,}</b> dòng sẽ bị xóa ({100 * mask.mean():.2f}% dữ liệu).", "warn")
                elif action == "cap":
                    ui.callout(f"Tác động: <b>{total:,}</b> giá trị sẽ được đưa về cận dưới/cận trên; số dòng giữ nguyên.", "info")
                ui.algorithm_card("iqr_zscore")
                if ui.button("Áp dụng xử lý ngoại lệ", key="out_apply", primary=True):
                    _apply(*cl.handle_outliers(df, cols, method, action, k, z))

    # ------------------------------------------------------------ Types & dates
    with tabs[3]:
        c1, c2 = st.columns(2)
        with c1:
            ui.section("Chuyển đổi kiểu dữ liệu")
            col = st.selectbox("Cột", all_cols, key="ct_col")
            st.caption(f"Kiểu hiện tại: `{df[col].dtype}` · Ví dụ: {', '.join(map(str, df[col].dropna().astype(str).head(3).tolist()))}")
            tgt = st.selectbox("Kiểu đích", ["numeric", "integer", "datetime", "category", "string"], key="ct_tgt",
                               format_func={"numeric": "Số thực", "integer": "Số nguyên", "datetime": "Thời gian",
                                            "category": "Phân loại", "string": "Chuỗi"}.get)
            if ui.button("Chuyển kiểu", key="ct_apply", primary=True):
                _apply(*cl.convert_type(df, col, tgt))
        with c2:
            ui.section("Xử lý dữ liệu ngày tháng")
            cands = [c for c in all_cols if schema.infer_role(df[c], c) in ("datetime", "year")] or all_cols
            dcol = st.selectbox("Cột ngày tháng", cands, key="dt_col")
            parts = st.multiselect("Tách thành phần", ["year", "month", "quarter", "day", "dayofweek"], default=["year", "month"], key="dt_parts")
            if parts and ui.button("Tách thành phần thời gian", key="dt_apply", primary=True):
                _apply(*cl.extract_datetime_parts(df, dcol, tuple(parts)))
        ui.section("Xóa cột không cần thiết")
        drop = st.multiselect("Cột cần xóa", all_cols, key="drop_cols")
        if drop and ui.button(f"Xóa {len(drop)} cột", key="drop_apply"):
            _apply(*cl.drop_columns(df, drop))

    # ------------------------------------------------------------ Scaling
    with tabs[4]:
        ui.callout("Lưu ý khoa học: module Machine Learning tự chuẩn hóa <b>bên trong pipeline</b> (chỉ học tham số trên tập train) để tránh "
                   "rò rỉ dữ liệu. Chuẩn hóa ở đây phục vụ phân tích/khám phá; mặc định tạo cột mới, không ghi đè cột gốc.", "info")
        if nums:
            cols = st.multiselect("Cột số", nums, key="sc_cols")
            m = st.radio("Phương pháp", ["standard", "minmax"], horizontal=True, key="sc_m",
                         format_func={"standard": "StandardScaler (z = (x − μ)/σ)", "minmax": "MinMaxScaler (về [0, 1])"}.get)
            overwrite = st.checkbox("Ghi đè cột gốc", value=False, key="sc_ow")
            if cols and ui.button("Áp dụng chuẩn hóa", key="sc_apply", primary=True):
                _apply(*cl.scale_columns(df, cols, m, suffix=not overwrite))
        else:
            ui.callout("Không có cột số.", "info")

    # ------------------------------------------------------------ Invalid values
    with tabs[5]:
        rep = cl.invalid_value_report(df)
        if rep.empty:
            ui.callout("Không phát hiện giá trị không hợp lệ (chuỗi giữ chỗ, vô hạn, số âm, khoảng trắng thừa).", "ok")
        else:
            ui.show_df(rep)
            c1, c2 = st.columns(2)
            ph = c1.checkbox("Chuỗi giữ chỗ (N/A, '-', 'Đang cập nhật'...) → NaN", value=True, key="inv_ph")
            ws = c1.checkbox("Loại bỏ khoảng trắng thừa", value=True, key="inv_ws")
            inf = c2.checkbox("Giá trị vô hạn → NaN", value=True, key="inv_inf")
            neg = c2.multiselect("Cột không được âm (số âm → NaN)", nums, key="inv_neg")
            if ui.button("Sửa giá trị không hợp lệ", key="inv_apply", primary=True):
                _apply(*cl.fix_invalid_values(df, rep["Cột"].tolist(), ph, ws, inf, neg))

    # ------------------------------------------------------------ Before/after & log
    with tabs[6]:
        raw = ss["raw_df"]
        ui.section("So sánh trước và sau khi làm sạch", "Dữ liệu gốc lúc tải lên → dữ liệu hiện tại")
        ui.show_df(cl.compare_quality(raw, df))
        ui.section("Nhật ký thao tác")
        log = ss.get("ws_clean_log") or []
        if log:
            lg = pd.DataFrame(log)[["ts", "action", "detail", "rows_before", "rows_after", "cells_changed"]]
            lg.columns = ["Thời điểm", "Thao tác", "Chi tiết", "Dòng trước", "Dòng sau", "Ô thay đổi"]
            ui.show_df(lg)
        else:
            st.caption("Chưa có thao tác nào.")
        c1, c2, c3 = st.columns(3)
        with c1:
            if ui.button("Hoàn tác bước gần nhất", key="undo", disabled=not ss.get("undo_stack")):
                state.undo(ss)
                st.rerun()
        with c2:
            if ui.button("Khôi phục dữ liệu gốc", key="reset", disabled=not log):
                state.reset_to_raw(ss)
                log_activity("cleaning", "Khôi phục dữ liệu gốc", "", ss.get("dataset_id"))
                st.rerun()
        with c3:
            ui.download("Tải dữ liệu sau xử lý (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                        f"cleaned_{ss['meta']['name'].rsplit('.', 1)[0]}.csv", "text/csv", key="dl_clean")
