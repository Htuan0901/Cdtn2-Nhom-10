"""Nhận diện schema linh hoạt: suy luận vai trò từng cột dựa trên KIỂU + NỘI DUNG + gợi ý tên cột.

Không có tên cột nào bị cố định. Tên cột chỉ được dùng như "gợi ý" (hint),
quyết định cuối cùng dựa trên phân tích giá trị thực tế.
"""
from __future__ import annotations

import re
import unicodedata
import warnings
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

ROLE_LABELS = {
    "id": "Định danh (ID)", "numeric": "Số (liên tục)", "discrete": "Số (rời rạc)",
    "categorical": "Phân loại", "text": "Văn bản tự do", "datetime": "Thời gian",
    "year": "Năm", "boolean": "Nhị phân", "constant": "Hằng số", "geo": "Địa lý",
}

TIME_HINTS = ("date", "time", "ngay", "thoi gian", "timestamp", "datetime", "month", "thang", "created", "posted")
YEAR_HINTS = ("year", "nam", "yr")
GEO_HINTS = ("location", "dia diem", "province", "city", "tinh", "thanh pho", "region", "khu vuc",
             "district", "quan", "country", "quoc gia", "address", "dia chi", "station", "tram", "area")
ID_HINTS = ("id", "ma", "code", "uuid", "index", "stt")


def is_text_dtype(s: pd.Series) -> bool:
    """Cột chuỗi: object (pandas 2) hoặc kiểu 'str'/'string' (pandas 3)."""
    return (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and not pd.api.types.is_bool_dtype(s)


def strip_accents(text: str) -> str:
    text = str(text).replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def norm_name(name: str) -> str:
    return re.sub(r"[_\-\.]+", " ", strip_accents(str(name)).lower()).strip()


def _has_hint(name: str, hints) -> bool:
    n = norm_name(name)
    tokens = set(n.split())
    return any((h in tokens) if " " not in h and len(h) <= 4 else (h in n) for h in hints)


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    role: str
    n_missing: int
    missing_pct: float
    n_unique: int
    sample: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["role_label"] = ROLE_LABELS.get(self.role, self.role)
        return d


_ISO_RE = r"^\s*\d{4}[-/]\d{1,2}([-/]\d{1,2})?([ T].*)?$"


def parse_dates(s: pd.Series) -> pd.Series:
    """Chuyển chuỗi → datetime: định dạng ISO (YYYY-MM-DD) đọc theo năm-tháng-ngày, còn lại ưu tiên ngày/tháng/năm (VN)."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    txt = s.astype("string")
    iso = txt.str.match(_ISO_RE).fillna(False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
            if iso.any():
                out[iso] = pd.to_datetime(txt[iso], errors="coerce", format="mixed")
            if (~iso).any():
                out[~iso] = pd.to_datetime(txt[~iso], errors="coerce", dayfirst=True, format="mixed")
        except (TypeError, ValueError):   # vd: chuỗi có múi giờ lẫn lộn
            out = pd.to_datetime(txt, errors="coerce", utc=True).dt.tz_localize(None)
    return out


def _is_datetime_like(s: pd.Series, name: str) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
        return False
    sample = s.dropna().astype(str).head(300)
    if sample.empty:
        return False
    # loại bỏ các chuỗi số thuần (vd: "12") để không nhận nhầm thành ngày
    if sample.str.fullmatch(r"\s*-?\d+(\.\d+)?\s*").mean() > 0.5:
        return False
    if sample.str.len().median() > 40:
        return False
    parsed = parse_dates(sample)
    ratio = parsed.notna().mean()
    return ratio >= (0.8 if _has_hint(name, TIME_HINTS) else 0.95)


def infer_role(s: pd.Series, name: str) -> str:
    n = len(s)
    non_null = s.dropna()
    nunique = non_null.nunique()
    if n == 0 or nunique <= 1:
        return "constant"
    if pd.api.types.is_bool_dtype(s) or (nunique == 2 and set(map(str, non_null.unique())) <= {"0", "1", "True", "False", "true", "false", "0.0", "1.0"}):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        is_int = pd.api.types.is_integer_dtype(s) or bool(np.all(np.mod(non_null, 1) == 0))
        if is_int and non_null.between(1900, 2100).all() and (_has_hint(name, YEAR_HINTS) or nunique <= 60):
            if _has_hint(name, YEAR_HINTS):
                return "year"
        if is_int and nunique == len(non_null) and _has_hint(name, ID_HINTS):
            return "id"
        if is_int and nunique == len(non_null) and len(non_null) > 50:
            diffs = np.diff(np.sort(non_null.to_numpy()))
            if diffs.size and np.all(diffs == 1):
                return "id"
        if is_int and nunique <= 15:
            return "discrete"
        return "numeric"
    # object / string
    if _is_datetime_like(s, name):
        return "datetime"
    as_str = non_null.astype(str)
    avg_len = as_str.str.len().mean()
    if _has_hint(name, GEO_HINTS) and avg_len < 120:
        return "geo"
    if nunique == len(non_null) and len(non_null) > 50 and _has_hint(name, ID_HINTS):
        return "id"
    if avg_len > 60:
        return "text"
    ratio = nunique / max(len(non_null), 1)
    if nunique <= 50 or ratio < 0.05:
        return "categorical"
    return "text" if avg_len > 30 else "categorical"


_CACHE: dict = {}


def _cache_key(df: pd.DataFrame):
    return (id(df), df.shape, tuple(map(str, df.columns)), tuple(map(str, df.dtypes)))


def profile_columns(df: pd.DataFrame) -> list[ColumnProfile]:
    key = _cache_key(df)
    if key in _CACHE:
        return _CACHE[key]
    out = _profile_columns(df)
    if len(_CACHE) > 16:
        _CACHE.clear()
    _CACHE[key] = out
    return out


def _profile_columns(df: pd.DataFrame) -> list[ColumnProfile]:
    out = []
    n = max(len(df), 1)
    for col in df.columns:
        s = df[col]
        miss = int(s.isna().sum())
        sample_vals = s.dropna().astype(str).head(3).tolist()
        sample = ", ".join(v[:30] + ("…" if len(v) > 30 else "") for v in sample_vals)
        out.append(ColumnProfile(col, str(s.dtype), infer_role(s, col), miss, round(100 * miss / n, 2),
                                 int(s.nunique(dropna=True)), sample))
    return out


def profile_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = [p.to_dict() for p in profile_columns(df)]
    return pd.DataFrame(rows)[["name", "dtype", "role_label", "n_missing", "missing_pct", "n_unique", "sample", "role"]]


def roles(df: pd.DataFrame) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for p in profile_columns(df):
        groups.setdefault(p.role, []).append(p.name)
    return groups


def numeric_columns(df: pd.DataFrame, include_discrete: bool = True) -> list[str]:
    """Các cột số có ý nghĩa phân tích: loại cột ID, hằng số, năm."""
    allowed = {"numeric", "discrete", "boolean"} if include_discrete else {"numeric"}
    return [p.name for p in profile_columns(df)
            if p.role in allowed and pd.api.types.is_numeric_dtype(df[p.name])]


def categorical_columns(df: pd.DataFrame, max_unique: int = 60) -> list[str]:
    return [p.name for p in profile_columns(df)
            if p.role in {"categorical", "geo", "boolean", "discrete", "year"} and p.n_unique <= max_unique]


def time_candidates(df: pd.DataFrame) -> list[str]:
    profs = profile_columns(df)
    cands = [p.name for p in profs if p.role in {"datetime", "year"}]
    return cands


def geo_candidates(df: pd.DataFrame) -> list[str]:
    return [p.name for p in profile_columns(df) if p.role == "geo" or (p.role == "categorical" and _has_hint(p.name, GEO_HINTS))]


def id_columns(df: pd.DataFrame) -> list[str]:
    return [p.name for p in profile_columns(df) if p.role == "id"]


def target_candidates(df: pd.DataFrame, max_classes: int = 30) -> list[str]:
    """Biến mục tiêu hợp lệ: cột số, hoặc cột phân loại có 2..max_classes lớp."""
    out = []
    for p in profile_columns(df):
        if p.n_unique < 2:
            continue
        if p.role in {"numeric", "discrete"} or (p.role in {"categorical", "boolean", "geo"} and p.n_unique <= max_classes):
            out.append(p.name)
    return out


def to_time_index(s: pd.Series) -> pd.Series:
    """Chuẩn hóa một cột thời gian (datetime hoặc năm) về datetime64."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s, errors="coerce")
        if vals.dropna().between(1900, 2100).all():
            return pd.to_datetime(vals.astype("Int64").astype(str).replace("<NA>", None), format="%Y", errors="coerce")
    return parse_dates(s)
