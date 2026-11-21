"""Bộ định tuyến câu hỏi (intent router) của Data Analysis Agent.

1. Nhận diện câu hỏi thuộc nhóm: dữ liệu (data) / kiến thức chung (general) / thông tin thời sự (web).
2. Với câu hỏi dữ liệu: chọn các công cụ Python cần gọi + tham số (cột, nhóm, độ chi tiết thời gian).
Quy tắc minh bạch, chạy được cả khi KHÔNG có Gemini (chế độ offline).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from src.analytics.data import schema
from src.analytics.domain.skills import SKILLS

DATA_REFS = ("dataset", "bo du lieu", "du lieu nay", "du lieu hien tai", "du lieu cua toi", "du lieu da tai", "trong du lieu",
             "file", "data cua toi", "trong data", "ban ghi", "so dong", "cot", "trong file", "tap du lieu", "tin tuyen dung",
             "tin dang", "bieu do", "ket qua", "cua toi", "hien co")
# Dấu hiệu câu hỏi định lượng ⇒ cần tính trên dữ liệu (kể cả khi có cụm "là gì")
DATA_CUES = r"bao nhieu|nhieu nhat|it nhat|cao nhat|thap nhat|\btop\s*\d*|trung binh|trung vi|ty le|so luong|thong ke|(?<!the )(?<!nhu the )\bnao\b"
EXPLAIN_RESULT = r"(giai thich|nhan xet|danh gia|dien giai).*(ket qua|bieu do|du bao|mo hinh|cum|bat thuong|tuong quan)"
GENERAL_PATTERNS = (r"\bla gi\b", r"hoat dong (nhu )?the nao", r"nguyen ly", r"khac (gi|nhau|nhu the nao)", r"so sanh .* va ",
                    r"giai thich (thuat toan|khai niem|cach)", r"dinh nghia", r"what is", r"how does", r"uu nhuoc diem",
                    r"tai sao", r"vi sao", r"nhung phuong phap", r"cach (hoc|tro thanh)", r"anh huong (the nao|nhu the nao)")
WEB_PATTERNS = (r"moi nhat", r"hien nay", r"hien tai", r"tin tuc", r"thoi su", r"gan day", r"nam nay", r"hom nay",
                r"cap nhat", r"\b20(2[5-9]|3\d)\b", r"latest", r"current", r"news", r"thi truong .* (hien|bay gio)")

INTENTS = [
    ("get_missing_values", r"thieu|missing|null|trong\b|khuyet"),
    ("calculate_correlation", r"tuong quan|correlation|lien quan|moi quan he|anh huong lan nhau|heatmap"),
    ("get_anomalies", r"bat thuong|anomal|dot bien|ngoai le|outlier|isolation"),
    ("get_model_metrics", r"mo hinh|model|machine learning|\bml\b|accuracy|rmse|r2|r\^2|f1|du doan|tot nhat.*mo hinh|huan luyen"),
    ("get_forecast_results", r"du bao|forecast|arima|tuong lai|ky toi|nam toi|nam sau"),
    ("get_cluster_results", r"cum|cluster|k-?means|phan nhom|pca|silhouette"),
    ("get_cleaning_log", r"lam sach|cleaning|da xu ly|nhat ky"),
    ("get_top_skills", r"ky nang|skill|cong nghe|ngon ngu lap trinh|framework|cong cu|tech stack|hot nhat|pho bien nhat"),
    ("get_salary_statistics", r"luong|salary|thu nhap|muc luong"),
    ("analyze_time_trend", r"theo (thoi gian|nam|thang|quy|ngay|tuan)|xu huong|tang|giam|bien dong|nam nao|thang nao|quy nao|qua cac nam"),
    ("get_dataset_summary", r"bao nhieu (ban ghi|dong|tin|cot|thuoc tinh)|tong quan|kich thuoc|so luong (ban ghi|tin|dong)|mo ta (dataset|du lieu)|trung lap|duplicate"),
    ("get_column_statistics", r"trung binh|mean|median|trung vi|lon nhat|nho nhat|cao nhat|thap nhat|max|min|do lech|phuong sai|thong ke|phan bo|phan phoi|mode"),
    ("get_group_statistics", r"theo (tinh|thanh pho|khu vuc|mien|vung|nhom|vi tri|cap bac|hoc van|kinh nghiem|nganh|cong ty|loai)|moi (tinh|nhom)|tung (tinh|nhom)|group"),
]


GROUP_RE = (r"(?:theo|moi|tung|giua cac) (nhom vi tri|vi tri cong viec|nhom nghe|nhom kinh nghiem|cap bac|kinh nghiem|hoc van|"
            r"thanh pho|tinh thanh|tinh|khu vuc|vung mien|mien|vung|nganh nghe|nganh|cong ty|hinh thuc|loai|nhom)")


@dataclass
class Route:
    kind: str                         # data | general | web | no_data
    calls: list[tuple[str, dict]] = field(default_factory=list)
    reason: str = ""


def _norm(q: str) -> str:
    return re.sub(r"\s+", " ", schema.strip_accents(q).lower()).strip()


def mentioned_columns(question: str, columns: list[str]) -> list[str]:
    q = " " + re.sub(r"[_\-\.]+", " ", _norm(question)) + " "
    hits = []
    for c in sorted(columns, key=lambda x: -len(str(x))):
        n = schema.norm_name(c)
        if len(n) >= 2 and re.search(r"(?<![a-z0-9])" + re.escape(n) + r"(?![a-z0-9])", q):
            if not any(n in schema.norm_name(h) for h in hits):
                hits.append(c)
    return hits


def mentioned_skill(question: str) -> str | None:
    q = question.lower()
    for canon in sorted(SKILLS, key=len, reverse=True):
        if len(canon) <= 2:
            if re.search(r"(?<![\w+#.])" + re.escape(canon) + r"(?![\w+#])", question):
                return canon
            continue
        if re.search(r"(?<![\w.])" + re.escape(canon.lower()) + r"(?![\w+#])", q):
            return canon
    return None


def _granularity(qn: str) -> str:
    for key, g in (("thang", "thang"), ("quy", "quy"), ("tuan", "tuan"), ("ngay", "ngay"), ("nam", "nam")):
        if re.search(rf"\b{key}\b", qn):
            return g
    return ""


def route(question: str, df: pd.DataFrame | None, web_enabled: bool = True) -> Route:
    qn = _norm(question)
    cols = list(df.columns) if df is not None else []
    col_hits = mentioned_columns(question, cols) if cols else []
    skill = mentioned_skill(question)
    intents = [name for name, pat in INTENTS if re.search(pat, qn)]
    has_data_ref = any(re.search(rf"\b{re.escape(r)}\b", qn) for r in DATA_REFS)
    is_general = any(re.search(p, qn) for p in GENERAL_PATTERNS)
    is_web = any(re.search(p, qn) for p in WEB_PATTERNS)
    data_cue = bool(re.search(DATA_CUES, qn))
    explain = bool(re.search(EXPLAIN_RESULT, qn))

    # 1) Câu hỏi khái niệm ("ARIMA là gì", "K-Means hoạt động thế nào") ⇒ kiến thức chung,
    #    trừ khi nhắc rõ tới dataset/kết quả hoặc hỏi định lượng ("Top 10 kỹ năng ... là gì?")
    if is_general and not (has_data_ref or data_cue or explain):
        if is_web and web_enabled:
            return Route("web", reason="Câu hỏi kiến thức cần thông tin cập nhật")
        return Route("general", reason="Câu hỏi kiến thức chung")
    # 2) Không có ý định dữ liệu nào
    if not intents and not has_data_ref and not col_hits and not explain:
        if is_web and web_enabled:
            return Route("web", reason="Câu hỏi về thông tin thời sự / mới nhất")
        return Route("general", reason="Không liên quan tới dataset")
    # 3) Hỏi về thực tế hiện tại ngoài dataset (vd "thị trường IT hiện nay xu hướng gì mới nhất?")
    if is_web and web_enabled and not has_data_ref and not col_hits and not data_cue:
        return Route("web", reason="Câu hỏi về thực tế hiện tại, không phải dataset")
    if df is None:
        return Route("no_data", reason="Câu hỏi về dữ liệu nhưng chưa có dataset")

    calls: list[tuple[str, dict]] = []
    numeric = [c for c in col_hits if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in col_hits if c not in numeric]
    gran = _granularity(qn)

    for name in intents:
        if name == "calculate_correlation":
            calls.append((name, {"column": numeric[0] if numeric else ""}))
        elif name == "analyze_time_trend":
            if "get_top_skills" in intents or skill:
                continue
            value = numeric[0] if numeric else ("salary_mid" if "get_salary_statistics" in intents and "salary_mid" in df else "")
            calls.append((name, {"value_column": value, "granularity": gran}))
        elif name == "get_column_statistics":
            targets = numeric or categorical
            if not targets and "get_salary_statistics" in intents:
                continue
            for c in targets[:2]:
                calls.append((name, {"column": c}))
        elif name == "get_group_statistics":
            m = re.search(GROUP_RE, qn)
            group = categorical[0] if categorical else (m.group(1) if m else "")
            if "get_salary_statistics" in intents:
                continue
            value = numeric[0] if numeric else ""
            if group:
                calls.append((name, {"group_column": group, "value_column": value}))
        elif name == "get_salary_statistics":
            m = re.search(GROUP_RE, qn)
            group = categorical[0] if categorical else (m.group(1) if m else "")
            calls.append((name, {"group_column": group}))
        elif name == "get_top_skills":
            if skill or re.search(r"xu huong|tang|giam|theo nam|du bao", qn):
                calls.append(("get_skill_trend", {"skill": skill or ""}))
            if not skill:
                calls.append((name, {}))
        elif name == "get_forecast_results" and (skill or "get_top_skills" in intents):
            calls.append(("get_skill_trend", {"skill": skill or ""}))
            calls.append((name, {}))
        else:
            calls.append((name, {}))
    if skill and not any(c[0] == "get_skill_trend" for c in calls):
        calls.append(("get_skill_trend", {"skill": skill}))
    if not calls:
        if numeric or categorical:
            calls.append(("get_column_statistics", {"column": (numeric or categorical)[0]}))
        else:
            calls.append(("get_dataset_summary", {}))
    # bỏ trùng lặp, giữ thứ tự
    seen, uniq = set(), []
    for n, kw in calls:
        key = (n, tuple(sorted(kw.items())))
        if key not in seen:
            seen.add(key)
            uniq.append((n, kw))
    return Route("data", uniq[:5], reason="Câu hỏi liên quan dataset")
