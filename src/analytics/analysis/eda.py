"""Phân tích khám phá dữ liệu (EDA) tự động + diễn giải bằng ngôn ngữ tự nhiên.

Mọi con số trong phần diễn giải đều được tính trực tiếp từ dữ liệu (không có số liệu giả).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.data import schema


# ------------------------------------------------------------------ Thống kê mô tả
def descriptive_stats(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    cols = columns or schema.numeric_columns(df)
    rows = []
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").astype(float).dropna()
        if x.empty:
            continue
        mode = x.mode()
        rows.append({
            "Biến": c, "N": int(x.size), "Mean": x.mean(), "Median": x.median(),
            "Mode": mode.iloc[0] if not mode.empty else np.nan, "Variance": x.var(), "Std": x.std(),
            "Min": x.min(), "Q1 (25%)": x.quantile(.25), "Q3 (75%)": x.quantile(.75), "Max": x.max(),
            "IQR": x.quantile(.75) - x.quantile(.25), "Skewness": x.skew(), "Kurtosis": x.kurt(),
            "CV (%)": 100 * x.std() / x.mean() if x.mean() != 0 else np.nan,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        num = out.columns.drop(["Biến", "N"])
        out[num] = out[num].astype(float).round(4)
    return out


def frequency_table(df: pd.DataFrame, column: str, top: int = 20) -> pd.DataFrame:
    vc = df[column].astype("object").fillna("(Thiếu)").value_counts()
    total = vc.sum()
    head = vc.head(top)
    out = pd.DataFrame({"Giá trị": head.index.astype(str), "Tần số": head.values,
                        "Tần suất (%)": (100 * head.values / total).round(2)})
    out["Tích lũy (%)"] = out["Tần suất (%)"].cumsum().round(2)
    if len(vc) > top:
        rest = vc.iloc[top:].sum()
        out.loc[len(out)] = [f"(Khác: {len(vc) - top} giá trị)", rest, round(100 * rest / total, 2), 100.0]
    return out


# ------------------------------------------------------------------ Tương quan
def correlation_matrix(df: pd.DataFrame, columns: list[str] | None = None, method: str = "pearson") -> pd.DataFrame:
    """Chỉ dùng cột số có phương sai > 0 và không phải ID/năm (tránh tương quan giả)."""
    cols = columns or schema.numeric_columns(df)
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > 1]
    if len(cols) < 2:
        return pd.DataFrame()
    return df[cols].corr(method=method, min_periods=10)


def top_correlations(corr: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if corr.empty:
        return pd.DataFrame(columns=["Biến 1", "Biến 2", "Hệ số", "Mức độ"])
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pairs = corr.where(mask).stack().reset_index()
    pairs.columns = ["Biến 1", "Biến 2", "Hệ số"]
    pairs["|r|"] = pairs["Hệ số"].abs()
    pairs = pairs.sort_values("|r|", ascending=False).head(n)
    pairs["Mức độ"] = pairs["|r|"].map(strength_label)
    pairs["Hệ số"] = pairs["Hệ số"].round(4)
    return pairs.drop(columns="|r|").reset_index(drop=True)


def strength_label(r: float) -> str:
    r = abs(r)
    if r >= 0.8:
        return "Rất mạnh"
    if r >= 0.6:
        return "Mạnh"
    if r >= 0.4:
        return "Trung bình"
    if r >= 0.2:
        return "Yếu"
    return "Rất yếu"


# ------------------------------------------------------------------ Nhóm & thời gian
def group_stats(df: pd.DataFrame, group_col: str, value_col: str | None = None, top: int = 20,
                min_count: int = 1) -> pd.DataFrame:
    g = df.groupby(df[group_col].astype("object").fillna("(Thiếu)"), observed=True)
    if value_col is None:
        out = g.size().rename("Số bản ghi").reset_index().sort_values("Số bản ghi", ascending=False)
        out["Tỷ lệ (%)"] = (100 * out["Số bản ghi"] / out["Số bản ghi"].sum()).round(2)
        return out.head(top).reset_index(drop=True)
    x = pd.to_numeric(df[value_col], errors="coerce").astype(float)
    agg = x.groupby(df[group_col].astype("object").fillna("(Thiếu)")).agg(["count", "mean", "median", "std", "min", "max"])
    agg = agg[agg["count"] >= min_count].sort_values("count", ascending=False).head(top)
    agg.columns = ["Số bản ghi", "Mean", "Median", "Std", "Min", "Max"]
    return agg.round(3).reset_index().rename(columns={group_col: group_col})


def time_aggregate(df: pd.DataFrame, time_col: str, value_col: str | None = None, freq: str | None = None,
                   agg: str = "mean") -> tuple[pd.DataFrame, str]:
    t = schema.to_time_index(df[time_col])
    freq = freq or infer_frequency(t)
    frame = pd.DataFrame({"t": t})
    if value_col:
        frame["v"] = pd.to_numeric(df[value_col], errors="coerce").astype(float).values
    frame = frame.dropna(subset=["t"]).set_index("t").sort_index()
    if value_col:
        res = frame["v"].resample(freq).agg(agg)
    else:
        res = frame.assign(v=1)["v"].resample(freq).sum()
    out = res.reset_index()
    out.columns = ["Thời gian", value_col or "Số bản ghi"]
    return out, freq


def infer_frequency(t: pd.Series) -> str:
    t = pd.Series(t).dropna().drop_duplicates().sort_values()
    if len(t) < 3:
        return "YS"
    if (t.dt.month == 1).all() and (t.dt.day == 1).all() and t.dt.year.nunique() == len(t):
        return "YS"
    med = t.diff().dt.total_seconds().median() / 86400
    if med <= 0.05:
        return "h"
    if med <= 1.5:
        return "D"
    if med <= 8:
        return "W"
    if med <= 40:
        return "MS"
    if med <= 120:
        return "QS"
    return "YS"


FREQ_LABEL = {"h": "Theo giờ", "D": "Theo ngày", "W": "Theo tuần", "MS": "Theo tháng", "QS": "Theo quý", "YS": "Theo năm"}


# ------------------------------------------------------------------ Diễn giải tự động
def describe_numeric_text(stats: pd.DataFrame) -> list[str]:
    msgs = []
    for _, r in stats.iterrows():
        shape = "gần đối xứng"
        if r["Skewness"] > 1:
            shape = "lệch phải mạnh (đuôi dài về phía giá trị lớn)"
        elif r["Skewness"] > 0.5:
            shape = "lệch phải nhẹ"
        elif r["Skewness"] < -1:
            shape = "lệch trái mạnh"
        elif r["Skewness"] < -0.5:
            shape = "lệch trái nhẹ"
        center = "Median" if abs(r["Skewness"]) > 1 else "Mean"
        msgs.append(
            f"**{r['Biến']}**: trung bình {r['Mean']:,.3g}, trung vị {r['Median']:,.3g}, độ lệch chuẩn {r['Std']:,.3g}; "
            f"phân phối {shape}. Nên dùng {center} làm giá trị đại diện."
        )
    return msgs


def describe_correlation_text(top: pd.DataFrame, method: str = "pearson") -> list[str]:
    if top.empty:
        return ["Không đủ biến số hợp lệ (cần ít nhất 2 cột số có biến thiên) để phân tích tương quan."]
    msgs = []
    for _, r in top.head(5).iterrows():
        direction = "cùng chiều" if r["Hệ số"] > 0 else "ngược chiều"
        msgs.append(f"{r['Biến 1']} và {r['Biến 2']}: r = {r['Hệ số']:.3f} ({r['Mức độ'].lower()}, {direction}).")
    msgs.append(f"Hệ số {method.title()} chỉ đo quan hệ {'tuyến tính' if method == 'pearson' else 'đơn điệu'}; "
                "tương quan không đồng nghĩa với quan hệ nhân quả.")
    return msgs


def describe_trend_text(series: pd.DataFrame) -> str:
    if series.shape[0] < 2:
        return "Không đủ số kỳ để nhận xét xu hướng."
    col = series.columns[1]
    y = series[col].astype(float).to_numpy()
    valid = ~np.isnan(y)
    if valid.sum() < 2:
        return "Không đủ số kỳ có dữ liệu để nhận xét xu hướng."
    x = np.arange(len(y))[valid]
    slope = np.polyfit(x, y[valid], 1)[0]
    first, last = y[valid][0], y[valid][-1]
    change = (last - first) / abs(first) * 100 if first else np.nan
    peak_idx = int(np.nanargmax(y))
    peak_t = series.iloc[peak_idx, 0]
    peak_label = peak_t.strftime("%Y-%m-%d") if hasattr(peak_t, "strftime") else str(peak_t)
    trend = "tăng" if slope > 0 else "giảm"
    txt = (f"{col} có xu hướng {trend} (độ dốc tuyến tính {slope:,.3g}/kỳ). Giá trị kỳ đầu {first:,.3g}, kỳ cuối {last:,.3g}"
           + (f" (thay đổi {change:+.1f}%)" if not np.isnan(change) else "") + f". Đỉnh cao nhất tại {peak_label} ({np.nanmax(y):,.3g}).")
    return txt


def key_findings(df: pd.DataFrame, max_items: int = 8) -> list[str]:
    """Các phát hiện quan trọng — tất cả tính từ dữ liệu hiện tại."""
    out = []
    n = len(df)
    if n == 0:
        return out
    miss = df.isna().mean().sort_values(ascending=False)
    if miss.iloc[0] > 0:
        out.append(f"Cột thiếu nhiều nhất là **{miss.index[0]}** ({miss.iloc[0] * 100:.1f}% giá trị bị thiếu).")
    else:
        out.append("Không có giá trị thiếu trong dữ liệu hiện tại.")
    ids = schema.id_columns(df)
    dup = int(df.duplicated(subset=[c for c in df.columns if c not in ids] or None).sum())
    if dup:
        out.append(f"Có **{dup:,}** dòng trùng lặp ({100 * dup / n:.2f}%) khi bỏ qua cột ID{' ' + ', '.join(ids) if ids else ''}.")
    nums = schema.numeric_columns(df)
    if nums:
        stats = descriptive_stats(df, nums)
        if not stats.empty:
            sk = stats.loc[stats["Skewness"].abs().idxmax()]
            if abs(sk["Skewness"]) > 1:
                out.append(f"Biến **{sk['Biến']}** lệch mạnh (skewness = {sk['Skewness']:.2f}); nên dùng trung vị hoặc biến đổi log.")
        corr = correlation_matrix(df, nums)
        top = top_correlations(corr, 1)
        if not top.empty:
            r = top.iloc[0]
            out.append(f"Cặp biến tương quan mạnh nhất: **{r['Biến 1']}** – **{r['Biến 2']}** (r = {r['Hệ số']:.3f}, {r['Mức độ'].lower()}).")
    for c in schema.categorical_columns(df, max_unique=40)[:3]:
        vc = df[c].value_counts(normalize=True)
        if not vc.empty and vc.index[0] is not None:
            out.append(f"Ở cột **{c}**, giá trị phổ biến nhất là “{vc.index[0]}” ({vc.iloc[0] * 100:.1f}%).")
    return out[:max_items]
