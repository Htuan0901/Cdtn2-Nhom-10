"""Bộ công cụ nội bộ của Data Analysis Agent.

Mọi con số trả lời người dùng đều được TÍNH bằng Python tại đây (pandas/numpy),
Gemini chỉ nhận kết quả đã tính để diễn giải - không tự tính toán, không bịa số liệu.
Mỗi hàm trả về dict có thể tuần tự hóa JSON; nếu dữ liệu không đủ ⇒ trả {"available": False, "reason": ...}.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.analytics.analysis import eda
from src.analytics.data import schema
from src.analytics.data.cleaning import count_duplicates

ALIASES = {
    "job_group": ("nhom vi tri", "vi tri cong viec", "nhom nghe", "linh vuc cntt", "job group", "vi tri"),
    "exp_bucket": ("nhom kinh nghiem", "muc kinh nghiem"),
    "salary_mid": ("luong", "muc luong", "salary", "thu nhap", "luong trung binh"),
    "exp_years": ("kinh nghiem", "so nam kinh nghiem", "experience"),
    "province": ("tinh", "thanh pho", "tinh thanh", "dia phuong", "khu vuc", "province", "city"),
    "region": ("vung mien", "mien", "vung", "region"),
}
# Khi nhóm theo một biến số liên tục, dùng phiên bản đã phân nhóm nếu có
BUCKETED = {"exp_years": "exp_bucket"}


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else round(v, 4)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, pd.DataFrame):
        return to_jsonable(obj.head(30).to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return to_jsonable(obj.head(30).to_dict())
    return obj


class DataTools:
    """state_getter() trả về dict: {"df", "meta", "time_col", "results": {key: value}, "ws_clean_log"}."""

    def __init__(self, state_getter: Callable[[], dict]):
        self._get = state_getter

    # ------------------------------------------------------------ tiện ích
    @property
    def df(self) -> pd.DataFrame | None:
        return self._get().get("df")

    def _result(self, key: str):
        return (self._get().get("results") or {}).get(key)

    def resolve_column(self, text: str | None, numeric_only: bool = False) -> str | None:
        """Ánh xạ tên cột người dùng nói (có/không dấu, alias) → tên cột thật trong dữ liệu."""
        df = self.df
        if df is None or not text:
            return None
        q = schema.norm_name(text)
        cols = [c for c in df.columns if not numeric_only or pd.api.types.is_numeric_dtype(df[c])]
        for c in cols:
            if schema.norm_name(c) == q:
                return c
        # alias của cột đặc trưng tuyển dụng
        for col, al in ALIASES.items():
            if col in cols and q in al:
                return col
        for col, al in ALIASES.items():
            if col in cols and any(a in q for a in al):
                return col
        # khớp một phần (ưu tiên tên dài nhất)
        best = None
        for c in cols:
            n = schema.norm_name(c)
            if len(n) >= 2 and (n in q or q in n):
                if best is None or len(n) > len(schema.norm_name(best)):
                    best = c
        return best

    def _need_df(self):
        if self.df is None:
            return {"available": False, "reason": "Chưa có dataset nào được tải lên hệ thống."}
        return None

    # ------------------------------------------------------------ công cụ
    def get_dataset_summary(self) -> dict:
        if (e := self._need_df()):
            return e
        df, st = self.df, self._get()
        meta = st.get("meta") or {}
        prof = schema.profile_columns(df)
        return to_jsonable({
            "available": True, "ten_file": meta.get("name"), "so_ban_ghi": len(df), "so_thuoc_tinh": df.shape[1],
            "so_o_thieu": int(df.isna().sum().sum()), "ty_le_hop_le_pct": 100 * (1 - df.isna().sum().sum() / max(df.size, 1)),
            "so_dong_trung_lap_bo_qua_id": count_duplicates(df), "so_bien_so": len(schema.numeric_columns(df)),
            "cot_thoi_gian": st.get("time_col"), "so_buoc_lam_sach_da_ap_dung": len(st.get("ws_clean_log") or []),
            "danh_sach_cot": [{"ten": p.name, "vai_tro": p.to_dict()["role_label"]} for p in prof][:60],
        })

    def get_column_statistics(self, column: str) -> dict:
        if (e := self._need_df()):
            return e
        col = self.resolve_column(column)
        if col is None:
            return {"available": False, "reason": f"Không tìm thấy cột phù hợp với '{column}' trong dữ liệu.",
                    "cac_cot_hien_co": list(self.df.columns)[:60]}
        s = self.df[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            x = s.dropna()
            if x.empty:
                return {"available": False, "reason": f"Cột {col} không có giá trị hợp lệ."}
            mode = x.mode()
            return to_jsonable({"available": True, "cot": col, "kieu": "số", "so_gia_tri": int(x.size), "so_thieu": int(s.isna().sum()),
                                "mean": x.mean(), "median": x.median(), "mode": mode.iloc[0] if not mode.empty else None,
                                "std": x.std(), "variance": x.var(), "min": x.min(), "q1": x.quantile(.25), "q3": x.quantile(.75),
                                "max": x.max(), "skewness": x.skew()})
        vc = s.astype("object").value_counts()
        return to_jsonable({"available": True, "cot": col, "kieu": "phân loại/văn bản", "so_gia_tri_khac_nhau": int(vc.size),
                            "so_thieu": int(s.isna().sum()),
                            "top_gia_tri": [{"gia_tri": str(k)[:80], "so_luong": int(v), "ty_le_pct": 100 * v / max(vc.sum(), 1)}
                                            for k, v in vc.head(10).items()]})

    def calculate_correlation(self, column: str = "", method: str = "pearson") -> dict:
        if (e := self._need_df()):
            return e
        corr = eda.correlation_matrix(self.df, method=method)
        if corr.empty:
            return {"available": False, "reason": "Cần ít nhất 2 biến số có biến thiên để tính tương quan."}
        col = self.resolve_column(column, numeric_only=True) if column else None
        if col and col in corr.columns:
            s = corr[col].drop(col).dropna().sort_values(key=np.abs, ascending=False)
            return to_jsonable({"available": True, "phuong_phap": method, "bien": col,
                                "tuong_quan_voi_bien_khac": [{"bien": k, "r": v, "muc_do": eda.strength_label(v)} for k, v in s.head(10).items()]})
        top = eda.top_correlations(corr, 10)
        return to_jsonable({"available": True, "phuong_phap": method, "so_bien": corr.shape[0],
                            "cac_cap_tuong_quan_manh_nhat": top.to_dict(orient="records"),
                            "luu_y": "Tương quan không phải nhân quả; chỉ tính trên các cột số hợp lệ (đã loại ID, hằng số)."})

    def get_missing_values(self) -> dict:
        if (e := self._need_df()):
            return e
        df = self.df
        miss = df.isna().sum()
        miss = miss[miss > 0].sort_values(ascending=False)
        return to_jsonable({"available": True, "tong_o_thieu": int(miss.sum()), "tong_o": int(df.size),
                            "ty_le_thieu_pct": 100 * miss.sum() / max(df.size, 1), "so_cot_co_thieu": int(miss.size),
                            "chi_tiet": [{"cot": k, "so_thieu": int(v), "ty_le_pct": 100 * v / len(df)} for k, v in miss.head(20).items()]})

    def get_group_statistics(self, group_column: str, value_column: str = "") -> dict:
        if (e := self._need_df()):
            return e
        g = self.resolve_column(group_column)
        if g is None:
            return {"available": False, "reason": f"Không tìm thấy cột nhóm '{group_column}'."}
        g = BUCKETED.get(g, g) if BUCKETED.get(g) in self.df.columns else g
        v = self.resolve_column(value_column, numeric_only=True) if value_column else None
        tab = eda.group_stats(self.df, g, v, top=15, min_count=5 if v else 1)
        if v:
            tab = tab.sort_values("Median", ascending=False)
        return to_jsonable({"available": True, "cot_nhom": g, "cot_gia_tri": v, "so_nhom": int(self.df[g].nunique()),
                            "bang": tab.to_dict(orient="records")})

    def analyze_time_trend(self, value_column: str = "", granularity: str = "") -> dict:
        if (e := self._need_df()):
            return e
        st = self._get()
        tcol = st.get("time_col") or next(iter(schema.time_candidates(self.df)), None)
        if not tcol:
            return {"available": False, "reason": "Dữ liệu không có cột thời gian nên không thể phân tích xu hướng theo thời gian."}
        t = schema.to_time_index(self.df[tcol])
        base_freq = eda.infer_frequency(t.dropna())
        want = {"thang": "MS", "month": "MS", "quy": "QS", "nam": "YS", "year": "YS", "ngay": "D", "tuan": "W"}.get(
            schema.norm_name(granularity), None)
        order = ["h", "D", "W", "MS", "QS", "YS"]
        if want and order.index(want) < order.index(base_freq):
            return {"available": False, "reason": f"Cột thời gian '{tcol}' chỉ có độ chi tiết '{eda.FREQ_LABEL[base_freq]}', "
                                                  f"không đủ chi tiết để phân tích '{eda.FREQ_LABEL[want]}'.",
                    "goi_y": "Có thể hỏi xu hướng theo đúng độ chi tiết hiện có."}
        v = self.resolve_column(value_column, numeric_only=True) if value_column else None
        series, freq = eda.time_aggregate(self.df, tcol, v, freq=want or base_freq)
        col = series.columns[1]
        series = series.dropna()
        if series.empty:
            return {"available": False, "reason": "Không có dữ liệu hợp lệ theo thời gian."}
        peak = series.loc[series[col].idxmax()]
        low = series.loc[series[col].idxmin()]
        fmt = (lambda ts: str(ts.year)) if freq == "YS" else (lambda ts: ts.strftime("%m/%Y") if freq in ("MS", "QS") else ts.strftime("%d/%m/%Y"))
        return to_jsonable({"available": True, "cot_thoi_gian": tcol, "do_chi_tiet": eda.FREQ_LABEL.get(freq, freq),
                            "chi_so": col if v else "Số bản ghi", "phep_gop": "trung bình" if v else "đếm",
                            "chuoi": [{"ky": fmt(r[0]), "gia_tri": r[1]} for r in series.itertuples(index=False)][-36:],
                            "ky_cao_nhat": {"ky": fmt(peak.iloc[0]), "gia_tri": peak.iloc[1]},
                            "ky_thap_nhat": {"ky": fmt(low.iloc[0]), "gia_tri": low.iloc[1]},
                            "nhan_xet": eda.describe_trend_text(series)})

    def get_anomalies(self) -> dict:
        r = self._result("anomaly")
        if not r:
            return {"available": False, "reason": "Chưa chạy phát hiện bất thường trên dataset hiện tại (module Anomaly Detection)."}
        top = r["frame"].sort_values("anomaly_score", ascending=False).head(5)
        cols = [c for c in r["features"]][:6] + ["anomaly_score"]
        return to_jsonable({"available": True, "so_bat_thuong": r["n_anomalies"], "ty_le_pct": 100 * r["rate"],
                            "contamination": r["contamination"], "bien_su_dung": r["features"],
                            "so_sanh_nhom": r["comparison"].reset_index().rename(columns={"index": "bien"}).to_dict(orient="records"),
                            "top_5_bat_thuong": top[cols].to_dict(orient="records")})

    def get_model_metrics(self) -> dict:
        r = self._result("ml")
        if not r:
            return {"available": False, "reason": "Chưa huấn luyện mô hình nào trên dataset hiện tại (module Machine Learning)."}
        return to_jsonable({"available": True, "bai_toan": "hồi quy" if r.problem == "regression" else "phân loại",
                            "bien_muc_tieu": r.target, "so_dac_trung": len(r.features), "n_train": r.n_train, "n_test": r.n_test,
                            "bang_so_sanh": r.comparison.to_dict(orient="records"), "mo_hinh_tot_nhat": r.best_name,
                            "ly_do_chon": r.best_reason, "ghi_chu": r.notes,
                            "dac_trung_quan_trong": r.details[r.best_name]["importance"].head(8).to_dict(orient="records")
                            if not r.details[r.best_name]["importance"].empty else []})

    def get_forecast_results(self) -> dict:
        r = self._result("forecast")
        if not r:
            return {"available": False, "reason": "Chưa chạy dự báo trên dataset hiện tại (module Forecasting)."}
        best = r["results"][r["best"]]
        fut = [{"ky": ts.strftime("%Y-%m-%d"), "du_bao": p, "can_duoi_95": lo, "can_tren_95": hi}
               for ts, p, lo, hi in zip(r["future_index"], best["future"], best["future_lo"], best["future_hi"])]
        return to_jsonable({"available": True, "bien": r["series"].name, "so_ky": len(r["series"]), "so_ky_test": r["test_size"],
                            "bang_danh_gia": r["table"].to_dict(orient="records"), "mo_hinh_tot_nhat": best["label"],
                            "du_bao_tuong_lai": fut, "mo_hinh_bi_tu_choi": r["skipped"], "dien_giai": r["explanation"]})

    def get_cluster_results(self) -> dict:
        r = self._result("mining")
        if not r:
            return {"available": False, "reason": "Chưa chạy phân cụm K-Means trên dataset hiện tại (module Data Mining)."}
        return to_jsonable({"available": True, "k": r["k"], "silhouette": r["silhouette"], "bien": r["features"],
                            "pca_phuong_sai_giai_thich": r["explained_variance"],
                            "ho_so_cum": r["profile"].reset_index().to_dict(orient="records"), "mo_ta": r["descriptions"]})

    def get_cleaning_log(self) -> dict:
        log = self._get().get("ws_clean_log") or []
        return to_jsonable({"available": True, "so_buoc": len(log), "cac_buoc": log[-15:]})

    # ------------------------------------------------------------ miền tuyển dụng
    def get_top_skills(self, n: int = 15) -> dict:
        r = self._result("skills")
        if not r:
            return {"available": False, "reason": "Chưa trích xuất kỹ năng (module Thị trường tuyển dụng CNTT)."}
        return to_jsonable({"available": True, "so_tin_phan_tich": r["n_postings"], "cot_van_ban": r["text_cols"],
                            "top_ky_nang": r["counts"].head(int(n)).to_dict(orient="records"),
                            "ty_le_tin_khong_trich_xuat_duoc_pct": r["no_skill_pct"]})

    def get_skill_trend(self, skill: str = "") -> dict:
        r = self._result("skills")
        if not r or r.get("share") is None:
            return {"available": False, "reason": "Chưa có bảng xu hướng kỹ năng theo thời gian (cần trích xuất kỹ năng và có cột thời gian)."}
        share = r["share"]
        cols = [c for c in share.columns if c != "Số tin"]
        target = None
        if skill:
            q = schema.norm_name(skill)
            target = next((c for c in cols if schema.norm_name(c) == q), None) or next((c for c in cols if schema.norm_name(c) in q or q in schema.norm_name(c)), None)
            if target is None:
                return {"available": False, "reason": f"Kỹ năng '{skill}' không nằm trong nhóm kỹ năng đang theo dõi xu hướng.",
                        "ky_nang_dang_theo_doi": cols}
        out = {"available": True, "don_vi": "% số tin trong kỳ có nhắc tới kỹ năng", "so_tin_moi_ky": share["Số tin"].to_dict()}
        if target:
            out["ky_nang"] = target
            out["ty_le_theo_ky"] = share[target].to_dict()
        else:
            first, last = share.index[0], share.index[-1]
            delta = (share.loc[last, cols] - share.loc[first, cols]).sort_values(ascending=False)
            out["tang_manh_nhat"] = delta.head(5).round(2).to_dict()
            out["giam_manh_nhat"] = delta.tail(5).round(2).to_dict()
            out["giai_doan"] = f"{first} → {last}"
        fc = r.get("forecast")
        if fc and fc.get("ok"):
            tab = fc["table"]
            if target:
                tab = tab[tab["Kỹ năng"] == target]
            out["du_bao_ky_toi"] = tab.head(15).to_dict(orient="records")
            out["backtest"] = {"ky_test": fc["test_period"], "MAE_linear": fc["mae_linear"], "MAE_naive": fc["mae_naive"],
                               "mo_hinh_tot_hon": fc["better"]}
        return to_jsonable(out)

    def get_salary_statistics(self, group_column: str = "") -> dict:
        if (e := self._need_df()):
            return e
        if "salary_mid" not in self.df.columns:
            return {"available": False, "reason": "Chưa chuẩn hóa lương (hãy chạy 'Tạo đặc trưng tuyển dụng' ở module Thị trường CNTT)."}
        x = self.df["salary_mid"].dropna()
        out = {"available": True, "don_vi": "triệu VND/tháng (điểm giữa khoảng lương)", "so_tin_co_luong": int(x.size),
               "mean": x.mean(), "median": x.median(), "q1": x.quantile(.25), "q3": x.quantile(.75), "min": x.min(), "max": x.max()}
        if "salary_status" in self.df:
            out["trang_thai_luong"] = self.df["salary_status"].value_counts().to_dict()
        g = self.resolve_column(group_column) if group_column else None
        if g and BUCKETED.get(g) in self.df.columns:
            g = BUCKETED[g]
        if g == "salary_mid":
            g = None
        if g:
            out["theo_nhom"] = eda.group_stats(self.df, g, "salary_mid", top=15, min_count=10).sort_values("Median", ascending=False).to_dict(orient="records")
            out["cot_nhom"] = g
        return to_jsonable(out)

    # ------------------------------------------------------------ đăng ký công cụ
    def registry(self) -> dict[str, Callable]:
        return {
            "get_dataset_summary": self.get_dataset_summary, "get_column_statistics": self.get_column_statistics,
            "calculate_correlation": self.calculate_correlation, "get_missing_values": self.get_missing_values,
            "get_group_statistics": self.get_group_statistics, "analyze_time_trend": self.analyze_time_trend,
            "get_anomalies": self.get_anomalies, "get_model_metrics": self.get_model_metrics,
            "get_forecast_results": self.get_forecast_results, "get_cluster_results": self.get_cluster_results,
            "get_cleaning_log": self.get_cleaning_log, "get_top_skills": self.get_top_skills,
            "get_skill_trend": self.get_skill_trend, "get_salary_statistics": self.get_salary_statistics,
        }

    def call(self, name: str, **kwargs) -> dict:
        fn = self.registry().get(name)
        if fn is None:
            return {"available": False, "reason": f"Công cụ {name} không tồn tại."}
        try:
            return fn(**kwargs)
        except Exception as exc:
            return {"available": False, "reason": f"Lỗi khi tính toán ({name}): {exc}"}

    def gemini_functions(self) -> list[Callable]:
        """Các hàm Python có chữ ký kiểu đơn giản + docstring để Gemini tự gọi (automatic function calling)."""
        tools = self

        def get_dataset_summary() -> dict:
            """Tổng quan dataset hiện tại: số bản ghi, số cột, số ô thiếu, trùng lặp, danh sách cột."""
            return tools.get_dataset_summary()

        def get_column_statistics(column: str) -> dict:
            """Thống kê mô tả của một cột (mean, median, mode, std, min, max hoặc tần suất nếu là cột phân loại)."""
            return tools.get_column_statistics(column)

        def calculate_correlation(column: str = "", method: str = "pearson") -> dict:
            """Tương quan giữa các biến số. Truyền column để xem tương quan của một biến cụ thể; method: pearson|spearman."""
            return tools.calculate_correlation(column, method)

        def get_missing_values() -> dict:
            """Số lượng và tỷ lệ giá trị thiếu theo từng cột."""
            return tools.get_missing_values()

        def get_group_statistics(group_column: str, value_column: str = "") -> dict:
            """Thống kê theo nhóm: đếm số bản ghi mỗi nhóm, hoặc mean/median của value_column theo group_column."""
            return tools.get_group_statistics(group_column, value_column)

        def analyze_time_trend(value_column: str = "", granularity: str = "") -> dict:
            """Xu hướng theo thời gian của một biến số (hoặc số bản ghi nếu bỏ trống). granularity: ngay|tuan|thang|quy|nam."""
            return tools.analyze_time_trend(value_column, granularity)

        def get_anomalies() -> dict:
            """Kết quả phát hiện bất thường Isolation Forest đã chạy."""
            return tools.get_anomalies()

        def get_model_metrics() -> dict:
            """Bảng so sánh mô hình Machine Learning đã huấn luyện và mô hình tốt nhất."""
            return tools.get_model_metrics()

        def get_forecast_results() -> dict:
            """Kết quả dự báo chuỗi thời gian đã chạy (sai số, mô hình tốt nhất, giá trị dự báo)."""
            return tools.get_forecast_results()

        def get_cluster_results() -> dict:
            """Kết quả phân cụm K-Means và PCA đã chạy."""
            return tools.get_cluster_results()

        def get_top_skills(n: int = 15) -> dict:
            """Các kỹ năng/công nghệ được yêu cầu nhiều nhất trong tin tuyển dụng."""
            return tools.get_top_skills(n)

        def get_skill_trend(skill: str = "") -> dict:
            """Xu hướng tỷ lệ tin yêu cầu một kỹ năng theo thời gian và dự báo kỳ tới; bỏ trống để xem kỹ năng tăng/giảm mạnh nhất."""
            return tools.get_skill_trend(skill)

        def get_salary_statistics(group_column: str = "") -> dict:
            """Thống kê lương (triệu VND/tháng), có thể theo nhóm như province, job_group, exp_bucket."""
            return tools.get_salary_statistics(group_column)

        return [get_dataset_summary, get_column_statistics, calculate_correlation, get_missing_values,
                get_group_statistics, analyze_time_trend, get_anomalies, get_model_metrics, get_forecast_results,
                get_cluster_results, get_top_skills, get_skill_trend, get_salary_statistics]
