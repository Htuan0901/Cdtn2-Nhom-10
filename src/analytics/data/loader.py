"""Đọc & kiểm tra file dữ liệu tải lên (CSV, và Parquet nếu có pyarrow).

- Tự dò encoding (UTF-8, UTF-8 BOM, CP1258, ...).
- Tự dò dấu phân cách (, ; tab |).
- Phát hiện file rỗng / không đọc được / không có cột.
- Không giả định bất kỳ tên cột cố định nào.
"""
from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass, field

import pandas as pd

from src.analytics.config import CSV_DELIMITERS, CSV_ENCODINGS


class DataLoadError(Exception):
    """Lỗi có thông điệp thân thiện để hiển thị trực tiếp cho người dùng."""


@dataclass
class LoadResult:
    df: pd.DataFrame
    name: str
    size_bytes: int
    encoding: str
    delimiter: str
    dataset_id: str
    warnings: list[str] = field(default_factory=list)

    @property
    def meta(self) -> dict:
        return {
            "dataset_id": self.dataset_id, "name": self.name, "size_bytes": self.size_bytes,
            "encoding": self.encoding, "delimiter": self.delimiter,
            "n_rows": int(self.df.shape[0]), "n_cols": int(self.df.shape[1]),
        }


def human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:,.0f} {unit}" if unit == "B" else f"{size:,.2f} {unit}"
        size /= 1024
    return f"{size:.2f} GB"


def fingerprint(raw: bytes, name: str) -> str:
    h = hashlib.sha1()
    h.update(name.encode("utf-8", "ignore"))
    h.update(str(len(raw)).encode())
    h.update(raw[:1_000_000])
    h.update(raw[-1_000_000:])
    return h.hexdigest()[:16]


def _decode(raw: bytes) -> tuple[str, str]:
    for enc in CSV_ENCODINGS:
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise DataLoadError("Không giải mã được file với các bảng mã hỗ trợ (UTF-8, UTF-8 BOM, CP1258, Latin-1).")


def detect_delimiter(sample: str) -> str:
    """Dò dấu phân cách: thử csv.Sniffer, nếu thất bại thì đếm tần suất trên các dòng đầu."""
    lines = [ln for ln in sample.splitlines()[:50] if ln.strip()]
    if not lines:
        return ","
    try:
        dialect = csv.Sniffer().sniff("\n".join(lines[:20]), delimiters="".join(CSV_DELIMITERS))
        if dialect.delimiter in CSV_DELIMITERS:
            return dialect.delimiter
    except csv.Error:
        pass
    best, best_score = ",", -1.0
    for d in CSV_DELIMITERS:
        counts = [ln.count(d) for ln in lines[:20]]
        if not counts or max(counts) == 0:
            continue
        # ưu tiên ký tự xuất hiện nhiều và đều đặn giữa các dòng
        mean = sum(counts) / len(counts)
        consistency = sum(1 for c in counts if c == counts[0]) / len(counts)
        score = mean * consistency
        if score > best_score:
            best, best_score = d, score
    return best


def _dedupe_columns(cols) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for i, c in enumerate(cols):
        name = str(c).strip() or f"column_{i + 1}"
        if name.lower().startswith("unnamed:"):
            name = f"column_{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        out.append(name)
    return out


def read_csv_bytes(raw: bytes, name: str = "data.csv", max_rows: int | None = None) -> LoadResult:
    if raw is None or len(raw) == 0:
        raise DataLoadError("File rỗng (0 byte).")
    if not raw.strip():
        raise DataLoadError("File chỉ chứa khoảng trắng, không có dữ liệu.")
    if b"\x00" in raw[:4096] and not raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        raise DataLoadError("File có vẻ là file nhị phân, không phải CSV văn bản.")
    text, enc = _decode(raw)
    delim = detect_delimiter(text[:200_000])
    warnings: list[str] = []
    try:
        df = pd.read_csv(io.StringIO(text), sep=delim, nrows=max_rows, low_memory=False,
                         skipinitialspace=True, on_bad_lines="warn")
    except pd.errors.EmptyDataError:
        raise DataLoadError("File không có dòng tiêu đề hoặc dữ liệu.")
    except pd.errors.ParserError as exc:
        try:
            df = pd.read_csv(io.StringIO(text), sep=delim, nrows=max_rows, engine="python", on_bad_lines="skip")
            warnings.append(f"Một số dòng sai cấu trúc đã bị bỏ qua khi đọc ({exc.__class__.__name__}).")
        except Exception as exc2:
            raise DataLoadError(f"Không đọc được dữ liệu: {exc2}") from exc2
    return _finalize(df, raw, name, enc, delim, warnings)


def read_parquet_bytes(raw: bytes, name: str, max_rows: int | None = None) -> LoadResult:
    if not raw:
        raise DataLoadError("File rỗng (0 byte).")
    try:
        df = pd.read_parquet(io.BytesIO(raw))
    except ImportError as exc:
        raise DataLoadError("Đọc Parquet cần thư viện pyarrow (pip install pyarrow).") from exc
    except Exception as exc:
        raise DataLoadError(f"Không đọc được file Parquet: {exc}") from exc
    if max_rows:
        df = df.head(max_rows)
    return _finalize(df, raw, name, "binary", "-", [])


def _finalize(df: pd.DataFrame, raw: bytes, name: str, enc: str, delim: str, warnings: list[str]) -> LoadResult:
    if df is None or df.shape[1] == 0:
        raise DataLoadError("Không tìm thấy cột dữ liệu nào.")
    df.columns = _dedupe_columns(df.columns)
    if df.shape[1] == 1 and df.shape[0] > 0 and delim == ",":
        warnings.append("Chỉ đọc được 1 cột - hãy kiểm tra lại dấu phân cách của file.")
    df = df.dropna(how="all")
    if df.shape[0] == 0:
        raise DataLoadError("File chỉ có tiêu đề, không có dòng dữ liệu nào.")
    df = df.reset_index(drop=True)
    return LoadResult(df=df, name=name, size_bytes=len(raw), encoding=enc, delimiter=delim,
                      dataset_id=fingerprint(raw, name), warnings=warnings)


def load_file(raw: bytes, name: str, max_rows: int | None = None) -> LoadResult:
    lower = name.lower()
    if lower.endswith(".parquet"):
        return read_parquet_bytes(raw, name, max_rows)
    if lower.endswith((".csv", ".txt", ".tsv")):
        return read_csv_bytes(raw, name, max_rows)
    raise DataLoadError("Định dạng không hỗ trợ. Hãy tải file .csv (hoặc .parquet).")


def delimiter_label(d: str) -> str:
    return {",": "Dấu phẩy (,)", ";": "Chấm phẩy (;)", "\t": "Tab (\\t)", "|": "Gạch đứng (|)", "-": "Không áp dụng"}.get(d, repr(d))
