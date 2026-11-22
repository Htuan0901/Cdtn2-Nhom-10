"""Các thao tác làm sạch dữ liệu.

Nguyên tắc:
- Mỗi hàm KHÔNG sửa DataFrame gốc, luôn trả về (df_mới, CleaningStep) để ghi nhật ký.
- Không hàm nào tự động xóa dữ liệu: người dùng phải gọi tường minh với tham số đã chọn.
- Mỗi bước đều có thống kê tác động (số dòng / ô bị thay đổi) để hiển thị trước & sau.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime

import numpy as np
import pandas as pd

from src.analytics.data.schema import id_columns, parse_dates, strip_accents

PLACEHOLDER_TOKENS = {
    "", "na", "n/a", "nan", "null", "none", "-", "--", "?", "#n/a", "missing",
    "dang cap nhat", "chua cap nhat", "khong xac dinh", "chua xac dinh", "khong ro",
}


@dataclass
class CleaningStep:
    action: str
    detail: str
    rows_before: int
    rows_after: int
    cells_changed: int = 0
    columns: list[str] = field(default_factory=list)
    ts: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ Tổng quan chất lượng
def quality_summary(df: pd.DataFrame, exclude_ids: bool = True) -> dict:
    total_cells = int(df.size) or 1
    missing_cells = int(df.isna().sum().sum())
    subset = [c for c in df.columns if c not in id_columns(df)] if exclude_ids else None
    dup = int(df.duplicated(subset=subset or None).sum()) if len(df) else 0
    return {
        "rows": int(len(df)), "cols": int(df.shape[1]),
        "missing_cells": missing_cells,
        "missing_pct": round(100 * missing_cells / total_cells, 2),
        "complete_rows": int(df.dropna().shape[0]),
        "valid_pct": round(100 * (1 - missing_cells / total_cells), 2),
        "duplicates": dup,
        "numeric_cols": int(df.select_dtypes(include="number").shape[1]),
        "memory_mb": round(df.memory_usage(deep=False).sum() / 1e6, 2),
    }


def compare_quality(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    b, a = quality_summary(before), quality_summary(after)
    labels = {
        "rows": "Số dòng", "cols": "Số cột", "missing_cells": "Ô bị thiếu", "missing_pct": "Tỷ lệ thiếu (%)",
        "complete_rows": "Dòng đầy đủ", "valid_pct": "Tỷ lệ hợp lệ (%)", "duplicates": "Dòng trùng lặp",
        "numeric_cols": "Số cột dạng số", "memory_mb": "Bộ nhớ (MB)",
    }
    rows = [{"Chỉ số": labels[k], "Trước": b[k], "Sau": a[k], "Thay đổi": round(a[k] - b[k], 2)} for k in labels]
    return pd.DataFrame(rows)


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    n = max(len(df), 1)
    miss = df.isna().sum()
    out = pd.DataFrame({"Cột": miss.index, "Số ô thiếu": miss.values,
                        "Tỷ lệ (%)": (100 * miss.values / n).round(2),
                        "Kiểu": [str(df[c].dtype) for c in miss.index]})
    return out.sort_values("Số ô thiếu", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ Trùng lặp
def duplicate_subset(df: pd.DataFrame, ignore_id: bool = True) -> list[str] | None:
    if not ignore_id:
        return None
    ids = id_columns(df)
    subset = [c for c in df.columns if c not in ids]
    return subset or None


def count_duplicates(df: pd.DataFrame, ignore_id: bool = True) -> int:
    return int(df.duplicated(subset=duplicate_subset(df, ignore_id)).sum())


def drop_duplicates(df: pd.DataFrame, ignore_id: bool = True, keep: str = "first") -> tuple[pd.DataFrame, CleaningStep]:
    subset = duplicate_subset(df, ignore_id)
    out = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    note = " (bỏ qua cột ID khi so sánh)" if ignore_id and subset is not None and len(subset) < df.shape[1] else ""
    step = CleaningStep("Xóa trùng lặp", f"Giữ bản ghi '{keep}'{note}; xóa {len(df) - len(out)} dòng.",
                        len(df), len(out), (len(df) - len(out)) * df.shape[1])
    return out, step


# ------------------------------------------------------------------ Giá trị thiếu
def impute(df: pd.DataFrame, columns: list[str], method: str, constant=None) -> tuple[pd.DataFrame, CleaningStep]:
    """method: mean | median | mode | constant | drop_rows | ffill."""
    out = df.copy()
    changed = 0
    details = []
    if method == "drop_rows":
        before = len(out)
        out = out.dropna(subset=columns).reset_index(drop=True)
        return out, CleaningStep("Xử lý thiếu: xóa dòng", f"Xóa {before - len(out)} dòng thiếu ở {len(columns)} cột.",
                                 before, len(out), (before - len(out)) * df.shape[1], columns)
    for col in columns:
        s = out[col]
        n_miss = int(s.isna().sum())
        if n_miss == 0:
            continue
        if method in ("mean", "median"):
            if not pd.api.types.is_numeric_dtype(s):
                details.append(f"{col}: bỏ qua (không phải cột số)")
                continue
            value = s.mean() if method == "mean" else s.median()
        elif method == "mode":
            m = s.mode(dropna=True)
            if m.empty:
                continue
            value = m.iloc[0]
        elif method == "constant":
            value = constant
            if pd.api.types.is_numeric_dtype(s):
                try:
                    value = float(constant)
                except (TypeError, ValueError):
                    details.append(f"{col}: hằng số không phải số, bỏ qua")
                    continue
        elif method == "ffill":
            out[col] = s.ffill()
            changed += n_miss - int(out[col].isna().sum())
            details.append(f"{col}: forward-fill")
            continue
        else:
            raise ValueError(f"Phương pháp không hợp lệ: {method}")
        out[col] = s.fillna(value)
        changed += n_miss
        shown = f"{value:.4g}" if isinstance(value, (int, float, np.floating)) else str(value)[:30]
        details.append(f"{col}: {n_miss} ô ← {shown}")
    label = {"mean": "Mean", "median": "Median", "mode": "Mode", "constant": "Hằng số", "ffill": "Forward fill"}[method]
    return out, CleaningStep(f"Xử lý thiếu: {label}", "; ".join(details) or "Không có ô thiếu.",
                             len(df), len(out), changed, columns)


def suggest_impute_method(s: pd.Series) -> str:
    """Gợi ý có cơ sở: số lệch mạnh → median, số gần đối xứng → mean, phân loại → mode."""
    if not pd.api.types.is_numeric_dtype(s):
        return "mode"
    skew = s.dropna().skew() if s.dropna().size > 2 else 0
    return "median" if abs(skew) > 1 else "mean"


# ------------------------------------------------------------------ Ngoại lệ
def outlier_bounds(s: pd.Series, method: str = "iqr", k: float = 1.5, z: float = 3.0) -> tuple[float, float]:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return (np.nan, np.nan)
    if method == "iqr":
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        return float(q1 - k * iqr), float(q3 + k * iqr)
    mu, sd = x.mean(), x.std(ddof=0)
    if sd == 0:
        return float(mu), float(mu)
    return float(mu - z * sd), float(mu + z * sd)


def detect_outliers(df: pd.DataFrame, columns: list[str], method: str = "iqr", k: float = 1.5, z: float = 3.0) -> pd.DataFrame:
    rows = []
    for col in columns:
        lo, hi = outlier_bounds(df[col], method, k, z)
        x = pd.to_numeric(df[col], errors="coerce")
        mask = (x < lo) | (x > hi)
        rows.append({"Cột": col, "Cận dưới": round(lo, 4), "Cận trên": round(hi, 4),
                     "Số ngoại lệ": int(mask.sum()),
                     "Tỷ lệ (%)": round(100 * mask.sum() / max(x.notna().sum(), 1), 2)})
    return pd.DataFrame(rows)


def handle_outliers(df: pd.DataFrame, columns: list[str], method: str = "iqr", action: str = "cap",
                    k: float = 1.5, z: float = 3.0) -> tuple[pd.DataFrame, CleaningStep]:
    """action: keep | remove | cap (winsorize về cận)."""
    out = df.copy()
    if action == "keep":
        return out, CleaningStep("Ngoại lệ: giữ nguyên", "Người dùng chọn giữ lại toàn bộ ngoại lệ.", len(df), len(df), 0, columns)
    mask_any = pd.Series(False, index=out.index)
    changed = 0
    for col in columns:
        lo, hi = outlier_bounds(out[col], method, k, z)
        x = pd.to_numeric(out[col], errors="coerce")
        mask = (x < lo) | (x > hi)
        if action == "cap":
            out[col] = x.clip(lower=lo, upper=hi).where(x.notna(), np.nan)
            changed += int(mask.sum())
        else:
            mask_any |= mask.fillna(False)
    if action == "remove":
        changed = int(mask_any.sum()) * df.shape[1]
        out = out.loc[~mask_any].reset_index(drop=True)
    m = "IQR (k=%.1f)" % k if method == "iqr" else "Z-score (|z|>%.1f)" % z
    verb = "giới hạn (capping)" if action == "cap" else "loại bỏ dòng"
    return out, CleaningStep(f"Ngoại lệ: {verb}", f"Phương pháp {m} trên {len(columns)} cột.", len(df), len(out), changed, columns)


# ------------------------------------------------------------------ Chuẩn hóa
def scale_columns(df: pd.DataFrame, columns: list[str], method: str = "standard", suffix: bool = True) -> tuple[pd.DataFrame, CleaningStep]:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    out = df.copy()
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    cols = [c for c in columns if pd.api.types.is_numeric_dtype(out[c])]
    if not cols:
        return out, CleaningStep("Chuẩn hóa", "Không có cột số hợp lệ.", len(df), len(df))
    values = scaler.fit_transform(out[cols].astype(float))  # sklearn bỏ qua NaN khi fit, giữ NaN khi transform
    tag = "_std" if method == "standard" else "_minmax"
    for i, c in enumerate(cols):
        out[f"{c}{tag}" if suffix else c] = values[:, i]
    name = "StandardScaler" if method == "standard" else "MinMaxScaler"
    where = f"tạo cột mới hậu tố '{tag}'" if suffix else "ghi đè cột gốc"
    return out, CleaningStep(f"Chuẩn hóa: {name}", f"{len(cols)} cột, {where}.", len(df), len(out),
                             int(out[cols].notna().sum().sum()), cols)


# ------------------------------------------------------------------ Kiểu dữ liệu
def convert_type(df: pd.DataFrame, column: str, target: str) -> tuple[pd.DataFrame, CleaningStep]:
    """target: numeric | datetime | category | string | integer."""
    out = df.copy()
    s = out[column]
    before_na = int(s.isna().sum())
    if target in ("numeric", "integer"):
        cleaned = s
        if not pd.api.types.is_numeric_dtype(s):
            cleaned = (s.astype(str).str.strip().str.replace(r"\s", "", regex=True)
                       .str.replace(",", ".", regex=False).replace({"nan": None, "None": None}))
        out[column] = pd.to_numeric(cleaned, errors="coerce")
        if target == "integer":
            out[column] = out[column].round().astype("Int64")
    elif target == "datetime":
        out[column] = parse_dates(s)
    elif target == "category":
        out[column] = s.astype("category")
    elif target == "string":
        out[column] = s.astype("string")
    else:
        raise ValueError("Kiểu đích không hợp lệ")
    coerced = int(out[column].isna().sum()) - before_na
    return out, CleaningStep(f"Chuyển kiểu → {target}", f"Cột '{column}': {max(coerced, 0)} giá trị không chuyển được đã thành NaN.",
                             len(df), len(out), int(s.notna().sum()), [column])


def extract_datetime_parts(df: pd.DataFrame, column: str, parts=("year", "month", "quarter", "dayofweek")) -> tuple[pd.DataFrame, CleaningStep]:
    out = df.copy()
    dt = parse_dates(out[column])
    out[column] = dt
    for p in parts:
        out[f"{column}_{p}"] = getattr(dt.dt, p)
    return out, CleaningStep("Xử lý ngày tháng", f"Tách {', '.join(parts)} từ '{column}'; {int(dt.isna().sum())} giá trị không hợp lệ.",
                             len(df), len(out), int(dt.notna().sum()) * len(parts), [column])


# ------------------------------------------------------------------ Giá trị không hợp lệ
def invalid_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Đếm: chuỗi giữ chỗ (N/A, '-', 'Đang cập nhật'...), giá trị vô hạn, số âm, khoảng trắng thừa."""
    rows = []
    for col in df.columns:
        s = df[col]
        placeholders = negatives = infinite = whitespace = 0
        if pd.api.types.is_numeric_dtype(s):
            x = pd.to_numeric(s, errors="coerce")
            infinite = int(np.isinf(x.astype(float)).sum())
            negatives = int((x < 0).sum())
        else:
            st = s.dropna().astype(str)
            if not st.empty:
                normalized = st.str.strip().str.lower().map(strip_accents)
                placeholders = int(normalized.isin(PLACEHOLDER_TOKENS).sum())
                whitespace = int((st != st.str.strip()).sum())
        if placeholders or negatives or infinite or whitespace:
            rows.append({"Cột": col, "Chuỗi giữ chỗ (N/A, -, ...)": placeholders, "Giá trị âm": negatives,
                         "Vô hạn (inf)": infinite, "Khoảng trắng thừa": whitespace})
    return pd.DataFrame(rows)


def fix_invalid_values(df: pd.DataFrame, columns: list[str] | None = None, placeholders_to_nan: bool = True,
                       strip_whitespace: bool = True, inf_to_nan: bool = True,
                       negative_to_nan_cols: list[str] | None = None) -> tuple[pd.DataFrame, CleaningStep]:
    out = df.copy()
    cols = columns or list(out.columns)
    changed = 0
    for col in cols:
        s = out[col]
        if pd.api.types.is_numeric_dtype(s):
            if inf_to_nan:
                mask = np.isinf(s.astype(float))
                changed += int(mask.sum())
                out[col] = s.mask(mask)
            if negative_to_nan_cols and col in negative_to_nan_cols:
                mask = out[col] < 0
                changed += int(mask.sum())
                out[col] = out[col].mask(mask)
        else:
            st = s.astype("object")
            notna = st.notna()
            if strip_whitespace:
                stripped = st.where(~notna, st.astype(str).str.strip())
                changed += int((stripped != st)[notna].sum())
                st = stripped
            if placeholders_to_nan:
                norm = st.where(~notna, st.astype(str).str.strip().str.lower().map(strip_accents))
                mask = norm.isin(PLACEHOLDER_TOKENS) & notna
                changed += int(mask.sum())
                st = st.mask(mask)
            out[col] = st
    return out, CleaningStep("Sửa giá trị không hợp lệ", f"{changed} ô đã được chuẩn hóa/chuyển NaN trên {len(cols)} cột.",
                             len(df), len(out), changed, cols)


def drop_columns(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, CleaningStep]:
    out = df.drop(columns=columns)
    return out, CleaningStep("Xóa cột", ", ".join(columns), len(df), len(out), len(df) * len(columns), columns)
