"""Lớp truy cập SQLite: lưu lịch sử hoạt động, dataset, lịch sử huấn luyện và nhật ký lỗi.

Mỗi hàm mở kết nối riêng (an toàn với cơ chế rerun đa luồng của Streamlit).
"""
from __future__ import annotations

import json
import sqlite3
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.analytics.config import DB_PATH
from src.analytics.logger import get_logger, redact

log = get_logger("db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    name TEXT, n_rows INTEGER, n_cols INTEGER, size_bytes INTEGER,
    encoding TEXT, delimiter TEXT, uploaded_at TEXT
);
CREATE TABLE IF NOT EXISTS activity_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, dataset_id TEXT, module TEXT, action TEXT, detail TEXT
);
CREATE TABLE IF NOT EXISTS training_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, dataset_id TEXT, problem_type TEXT, target TEXT, model_name TEXT,
    metrics TEXT, is_best INTEGER, model_path TEXT
);
CREATE TABLE IF NOT EXISTS error_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, module TEXT, message TEXT, trace TEXT
);
"""


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def connect(db_path: Path | str | None = None):
    path = Path(db_path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10)
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def register_dataset(meta: dict, db_path=None) -> None:
    try:
        with connect(db_path) as c:
            c.execute(
                "INSERT OR REPLACE INTO datasets VALUES (?,?,?,?,?,?,?,?)",
                (meta["dataset_id"], meta.get("name"), meta.get("n_rows"), meta.get("n_cols"),
                 meta.get("size_bytes"), meta.get("encoding"), meta.get("delimiter"), _now()),
            )
    except Exception as exc:  # DB lỗi không được làm sập ứng dụng
        log.warning("register_dataset failed: %s", exc)


def log_activity(module: str, action: str, detail: str = "", dataset_id: str | None = None, db_path=None) -> None:
    try:
        with connect(db_path) as c:
            c.execute(
                "INSERT INTO activity_log (ts, dataset_id, module, action, detail) VALUES (?,?,?,?,?)",
                (_now(), dataset_id, module, action, redact(str(detail))[:2000]),
            )
        log.info("[%s] %s - %s", module, action, redact(str(detail))[:300])
    except Exception as exc:
        log.warning("log_activity failed: %s", exc)


def log_error(module: str, exc: BaseException | str, db_path=None) -> None:
    msg = redact(str(exc))
    tb = redact(traceback.format_exc()) if isinstance(exc, BaseException) else ""
    log.error("[%s] %s", module, msg)
    try:
        with connect(db_path) as c:
            c.execute("INSERT INTO error_log (ts, module, message, trace) VALUES (?,?,?,?)",
                      (_now(), module, msg[:2000], tb[:8000]))
    except Exception as e2:
        log.warning("log_error failed: %s", e2)


def log_training(dataset_id: str, problem_type: str, target: str, model_name: str,
                 metrics: dict, is_best: bool, model_path: str | None = None, db_path=None) -> None:
    try:
        with connect(db_path) as c:
            c.execute(
                "INSERT INTO training_history (ts, dataset_id, problem_type, target, model_name, metrics, is_best, model_path)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (_now(), dataset_id, problem_type, target, model_name,
                 json.dumps(metrics, ensure_ascii=False, default=float), int(is_best), model_path),
            )
    except Exception as exc:
        log.warning("log_training failed: %s", exc)


def read_table(table: str, limit: int = 200, dataset_id: str | None = None, db_path=None) -> pd.DataFrame:
    if table not in {"datasets", "activity_log", "training_history", "error_log"}:
        raise ValueError("Bảng không hợp lệ")
    order = "uploaded_at" if table == "datasets" else "id"
    with connect(db_path) as c:
        if dataset_id and table in {"activity_log", "training_history"}:
            q = f"SELECT * FROM {table} WHERE dataset_id=? ORDER BY {order} DESC LIMIT ?"
            return pd.read_sql_query(q, c, params=(dataset_id, limit))
        return pd.read_sql_query(f"SELECT * FROM {table} ORDER BY {order} DESC LIMIT ?", c, params=(limit,))


def count_trained_models(dataset_id: str | None, db_path=None) -> int:
    if not dataset_id:
        return 0
    try:
        with connect(db_path) as c:
            row = c.execute("SELECT COUNT(*) FROM training_history WHERE dataset_id=?", (dataset_id,)).fetchone()
            return int(row[0])
    except Exception:
        return 0


def health_check(db_path=None) -> bool:
    try:
        with connect(db_path) as c:
            c.execute("SELECT 1")
        return True
    except Exception:
        return False
