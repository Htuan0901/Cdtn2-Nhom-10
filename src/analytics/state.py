"""Quản lý trạng thái phiên làm việc (dataset hiện tại + kết quả các module).

Hoạt động trên mọi đối tượng giống dict (st.session_state hoặc dict thường khi test).
Mỗi kết quả phân tích được gắn dataset_id: khi người dùng tải dataset mới, toàn bộ
kết quả cũ bị xóa để không bao giờ dùng nhầm kết quả của dataset trước.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, MutableMapping

import pandas as pd

RESULT_KEYS = ("eda", "mining", "ml", "forecast", "anomaly", "jobs", "skills")
MAX_UNDO = 5


def init_state(ss: MutableMapping) -> None:
    defaults = {
        "raw_df": None, "df": None, "meta": None, "dataset_id": None,
        "time_col": None, "target_col": None, "ws_clean_log": [], "undo_stack": [],
        "results": {}, "chat_history": [], "updated_at": None, "job_mapping": None, "job_report": None,
    }
    for k, v in defaults.items():
        if k not in ss:
            ss[k] = v


def set_dataset(ss: MutableMapping, df: pd.DataFrame, meta: dict) -> None:
    """Thay thế dataset: reset toàn bộ kết quả, nhật ký làm sạch và ngữ cảnh chat."""
    ss["raw_df"] = df
    ss["df"] = df.copy()
    ss["meta"] = dict(meta)
    ss["dataset_id"] = meta["dataset_id"]
    ss["time_col"] = None
    ss["target_col"] = None
    ss["ws_clean_log"] = []
    ss["undo_stack"] = []
    ss["results"] = {}
    ss["job_mapping"] = None
    ss["job_report"] = None
    ss["chat_history"] = []
    ss["updated_at"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")


def has_data(ss: MutableMapping) -> bool:
    return ss.get("df") is not None


def current_df(ss: MutableMapping) -> pd.DataFrame | None:
    return ss.get("df")


def apply_step(ss: MutableMapping, new_df: pd.DataFrame, step) -> None:
    """Áp dụng một bước làm sạch: lưu bản cũ để Undo, ghi log, vô hiệu kết quả phân tích phụ thuộc."""
    stack = ss.get("undo_stack") or []
    stack.append((ss["df"], len(ss.get("ws_clean_log") or [])))
    ss["undo_stack"] = stack[-MAX_UNDO:]
    ss["df"] = new_df
    log = ss.get("ws_clean_log") or []
    log.append(step.to_dict() if hasattr(step, "to_dict") else dict(step))
    ss["ws_clean_log"] = log
    invalidate_results(ss)
    ss["updated_at"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    _validate_selected_columns(ss)


def undo(ss: MutableMapping) -> bool:
    stack = ss.get("undo_stack") or []
    if not stack:
        return False
    prev_df, log_len = stack.pop()
    ss["df"] = prev_df
    ss["ws_clean_log"] = (ss.get("ws_clean_log") or [])[:log_len]
    ss["undo_stack"] = stack
    invalidate_results(ss)
    _validate_selected_columns(ss)
    return True


def reset_to_raw(ss: MutableMapping) -> None:
    if ss.get("raw_df") is not None:
        ss["df"] = ss["raw_df"].copy()
        ss["ws_clean_log"] = []
        ss["undo_stack"] = []
        ss["results"] = {}
        ss["job_mapping"] = None
        ss["job_report"] = None
        _validate_selected_columns(ss)


def invalidate_results(ss: MutableMapping, keep: tuple = ()) -> None:
    res = ss.get("results") or {}
    ss["results"] = {k: v for k, v in res.items() if k in keep}


def _validate_selected_columns(ss: MutableMapping) -> None:
    df = ss.get("df")
    for key in ("time_col", "target_col"):
        if ss.get(key) and (df is None or ss[key] not in df.columns):
            ss[key] = None


def put_result(ss: MutableMapping, key: str, value: Any) -> None:
    res = ss.get("results") or {}
    res[key] = {"dataset_id": ss.get("dataset_id"), "value": value,
                "ts": datetime.now().strftime("%d/%m/%Y %H:%M:%S")}
    ss["results"] = res


def get_result(ss: MutableMapping, key: str) -> Any:
    """Chỉ trả về kết quả nếu thuộc đúng dataset hiện tại."""
    item = (ss.get("results") or {}).get(key)
    if not item or item.get("dataset_id") != ss.get("dataset_id"):
        return None
    return item["value"]
