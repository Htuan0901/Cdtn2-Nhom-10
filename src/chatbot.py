"""Trợ lý AI (Google Gemini) trả lời câu hỏi về dữ liệu tuyển dụng CNTT và câu hỏi tổng quát.

Cách hoạt động:
1. Tóm tắt dữ liệu (quy mô, kỹ năng hot, lương, xu hướng, kết quả mô hình) được đưa vào system prompt.
2. Gemini được cấp một bộ "công cụ" (function calling) chạy pandas trên dữ liệu thật để tính toán chính xác:
   đếm tin theo kỹ năng, lương theo nhóm, xu hướng theo tháng, tìm tin theo từ khóa…
3. Với câu hỏi ngoài dữ liệu, Gemini trả lời bằng kiến thức chung (tùy chọn bật Google Search).
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

_STATE: dict = {"li": None, "hn": None}
MAX_ROWS_OUT = 25


def set_data(li: pd.DataFrame | None, hn: pd.DataFrame | None, **extra):
    _STATE.update(li=li, hn=hn, **extra)


def _ds(name: str) -> pd.DataFrame:
    key = "hn" if str(name).lower().startswith(("hn", "hacker")) else "li"
    df = _STATE.get(key)
    if df is None:
        raise ValueError("Chưa có dữ liệu được tải vào ứng dụng.")
    return df


def _jsonable(obj):
    if isinstance(obj, pd.DataFrame):
        obj = obj.head(MAX_ROWS_OUT)
        return json.loads(obj.to_json(orient="records", force_ascii=False, date_format="iso"))
    if isinstance(obj, pd.Series):
        return json.loads(obj.head(MAX_ROWS_OUT).to_json(force_ascii=False, date_format="iso"))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _has_skill(df: pd.DataFrame, skill: str) -> pd.Series:
    sk = skill.strip().lower()
    return df["skills"].str.lower().str.split(";").map(lambda xs: sk in xs)


# ============================ CÔNG CỤ CHO GEMINI ============================

def thong_tin_du_lieu() -> dict:
    """Thông tin tổng quan 2 bộ dữ liệu: 'linkedin' (tin CNTT tại Mỹ 2024, có lương) và 'hn' (Hacker News 2018–2023 theo tháng):
    số dòng, các cột, khoảng thời gian."""
    li, hn = _ds("li"), _ds("hn")
    return {
        "linkedin": {"so_dong": len(li), "cot": list(li.columns), "so_tin_co_luong": int(li["salary_usd"].notna().sum())},
        "hn": {"so_dong": len(hn), "cot": list(hn.columns), "tu_thang": f"{hn['month'].min():%Y-%m}", "den_thang": f"{hn['month'].max():%Y-%m}"},
        "ghi_chu": "Cột skills chứa danh sách kỹ năng ngăn cách bằng dấu ';'. salary_usd là lương USD/năm.",
    }


def top_ky_nang(dataset: str = "linkedin", top_n: int = 15, role: str = "", year: int = 0) -> dict:
    """Top kỹ năng được yêu cầu nhiều nhất. dataset: 'linkedin' hoặc 'hn'. role: lọc theo nhóm vị trí (vd 'Data Engineer').
    year: chỉ áp dụng cho 'hn' (0 = tất cả năm)."""
    df = _ds(dataset)
    if role:
        df = df[df["role"].str.lower() == role.lower()]
    if year and "month" in df:
        df = df[df["month"].dt.year == int(year)]
    e = df["skills"].str.split(";").explode()
    vc = e[e.notna() & (e != "")].value_counts().head(max(1, min(int(top_n), 40)))
    return {"so_tin": len(df), "ket_qua": [{"ky_nang": k, "so_tin": int(v), "ti_le_%": round(v / max(len(df), 1) * 100, 2)} for k, v in vc.items()]}


def luong_theo_nhom(group_by: str = "role", skill: str = "", min_count: int = 20) -> dict:
    """Thống kê lương (USD/năm, dữ liệu LinkedIn) theo nhóm. group_by: 'role' (vị trí), 'experience' (cấp bậc),
    'state' (bang), 'work_type', 'remote'. skill: chỉ tính các tin yêu cầu kỹ năng này (tùy chọn)."""
    df = _ds("li").dropna(subset=["salary_usd"])
    if group_by not in {"role", "experience", "state", "work_type", "remote"}:
        raise ValueError("group_by phải là: role, experience, state, work_type, remote")
    if skill:
        df = df[_has_skill(df, skill)]
    g = df.groupby(group_by)["salary_usd"].agg(so_tin="size", trung_vi="median", trung_binh="mean",
                                                p25=lambda s: s.quantile(.25), p75=lambda s: s.quantile(.75))
    g = g[g["so_tin"] >= int(min_count)].sort_values("trung_vi", ascending=False).round(0).reset_index()
    return {"nhom_theo": group_by, "ky_nang": skill or None, "ket_qua": _jsonable(g)}


def luong_theo_ky_nang(skills: str) -> dict:
    """So sánh lương trung vị (LinkedIn) của các tin có từng kỹ năng. skills: danh sách cách nhau bởi dấu phẩy, vd 'Python, Java, Go'."""
    df = _ds("li").dropna(subset=["salary_usd"])
    out = []
    for sk in [s.strip() for s in skills.split(",") if s.strip()]:
        sub = df[_has_skill(df, sk)]["salary_usd"]
        out.append({"ky_nang": sk, "so_tin_co_luong": int(len(sub)),
                    "luong_trung_vi": round(float(sub.median()), 0) if len(sub) else None})
    return {"ket_qua": out, "luong_trung_vi_toan_bo": round(float(df["salary_usd"].median()), 0)}


def xu_huong_ky_nang(skill: str = "", freq: str = "Y") -> dict:
    """Xu hướng theo thời gian trên Hacker News 2018–2023. skill rỗng = tổng số tin; có skill = số tin và % tin nhắc tới skill.
    freq: 'M' (tháng), 'Q' (quý), 'Y' (năm)."""
    hn = _ds("hn")
    f = {"M": "MS", "Q": "QS", "Y": "YS"}.get(freq.upper(), "YS")
    tot = hn.set_index("month").resample(f).size()
    if not skill:
        r = tot
        r.index = r.index.strftime("%Y-%m" if f != "YS" else "%Y")
        return {"chi_so": "so_tin", "ket_qua": {k: int(v) for k, v in r.items()}}
    cnt = hn[_has_skill(hn, skill)].set_index("month").resample(f).size().reindex(tot.index, fill_value=0)
    pct = (cnt / tot * 100).round(2)
    idx = tot.index.strftime("%Y-%m" if f != "YS" else "%Y")
    return {"ky_nang": skill, "ket_qua": [{"ky": i, "so_tin": int(c), "ti_le_%": float(p)} for i, c, p in zip(idx, cnt, pct)],
            "ghi_chu": "Năm 2023 chỉ có dữ liệu tới tháng 6."}


def ky_nang_tang_giam(top_n: int = 10) -> dict:
    """Các kỹ năng tăng/giảm mạnh nhất về tỉ lệ % tin trên Hacker News giữa 2018–2019 và 2022–2023."""
    from .forecast import trend_ranking
    r = trend_ranking(_ds("hn"), min_posts=300)
    n = max(1, min(int(top_n), 20))
    return {"tang_manh": _jsonable(r.head(n)), "giam_manh": _jsonable(r.tail(n).iloc[::-1])}


def tim_tin(keyword: str, dataset: str = "linkedin", limit: int = 5) -> dict:
    """Tìm tin tuyển dụng có chứa từ khóa (tiếng Anh, trong tiêu đề hoặc mô tả). Trả về số kết quả và vài tin mẫu."""
    df = _ds(dataset)
    pat = rf"\b{re.escape(keyword.lower().strip())}\b"
    text = df["desc_excerpt"].str.lower()
    if "title" in df:
        text = df["title"].str.lower() + " " + text
    mask = text.str.contains(pat, regex=True, na=False)
    sub = df[mask]
    cols = [c for c in ["title", "position", "company_name", "role", "experience", "location", "country", "month", "salary_usd"]
            if c in df.columns]
    samp = sub.sample(min(len(sub), max(1, min(int(limit), 10))), random_state=0)[cols] if len(sub) else sub[cols]
    out = {"tu_khoa": keyword, "so_ket_qua": int(mask.sum()), "ti_le_%": round(float(mask.mean() * 100), 3), "vi_du": _jsonable(samp)}
    if "salary_usd" in sub and sub["salary_usd"].notna().any():
        out["luong_trung_vi"] = round(float(sub["salary_usd"].median()), 0)
    return out


def dem_theo_cot(column: str, dataset: str = "linkedin", top_n: int = 15) -> dict:
    """Đếm số tin theo giá trị của một cột (vd role, experience, state, city, work_type, country, seniority, company_name)."""
    df = _ds(dataset)
    if column not in df.columns:
        raise ValueError(f"Không có cột '{column}'. Các cột: {list(df.columns)}")
    vc = df[column].value_counts().head(max(1, min(int(top_n), 50)))
    return {"cot": column, "ket_qua": [{"gia_tri": _jsonable(k), "so_tin": int(v), "ti_le_%": round(v / len(df) * 100, 2)} for k, v in vc.items()]}


def ket_qua_mo_hinh() -> dict:
    """Kết quả các mô hình đã chạy trong phiên: dự báo chuỗi thời gian (so với baseline) và mô hình dự đoán lương."""
    out = {}
    fc = _STATE.get("fc_result")
    if fc:
        out["du_bao"] = {"chuoi": fc["target"], "don_vi": fc["unit"], "so_thang_test": fc["h"], "tot_nhat": fc["best"],
                         "bang": _jsonable(fc["results"].round(3))}
    sb = _STATE.get("salary_bundle")
    if sb:
        out["luong"] = {"tot_nhat": sb["best"], "bang": _jsonable(sb["results"].round(3))}
    return out or {"thong_bao": "Người dùng chưa chạy mô hình nào trong phiên này."}


DATA_TOOLS = [thong_tin_du_lieu, top_ky_nang, luong_theo_nhom, luong_theo_ky_nang, xu_huong_ky_nang, ky_nang_tang_giam,
              tim_tin, dem_theo_cot, ket_qua_mo_hinh]


# ============================ NGỮ CẢNH DỮ LIỆU ============================

def build_data_context(li: pd.DataFrame, hn: pd.DataFrame) -> str:
    e = li["skills"].str.split(";").explode()
    top = e[e.notna() & (e != "")].value_counts().head(10)
    sal = li["salary_usd"].dropna()
    lines = ["## BỐI CẢNH DỮ LIỆU (đã làm sạch)",
             f"- LinkedIn (tin CNTT tại Mỹ, đăng chủ yếu 3–4/2024): {len(li):,} tin, {li['company_name'].nunique():,} công ty; "
             f"{len(sal):,} tin có lương, trung vị {sal.median():,.0f} USD/năm.",
             "- Top 10 kỹ năng LinkedIn: " + ", ".join(f"{k} ({v / len(li) * 100:.1f}%)" for k, v in top.items()),
             "- Nhóm vị trí: " + ", ".join(f"{k} ({v})" for k, v in li["role"].value_counts().head(8).items()),
             f"- Hacker News “Who is hiring”: {len(hn):,} tin từ {hn['month'].min():%m/%Y} đến {hn['month'].max():%m/%Y} "
             f"({hn['month'].nunique()} tháng); số tin/tháng cao nhất {hn.groupby('month').size().max()}, "
             f"tháng cuối {hn.groupby('month').size().iloc[-1]}.",
             "- Cột LinkedIn: " + ", ".join(li.columns), "- Cột HN: " + ", ".join(hn.columns)]
    fc = _STATE.get("fc_result")
    if fc:
        r = fc["results"].iloc[0]
        lines.append(f"- Mô hình dự báo tốt nhất cho '{fc['target']}': {r['Mô hình']} (RMSE={r['RMSE']:.2f}, MAE={r['MAE']:.2f}).")
    sb = _STATE.get("salary_bundle")
    if sb:
        r = sb["results"][sb["results"]["Mô hình"] == sb["best"]].iloc[0]
        lines.append(f"- Mô hình lương tốt nhất: {sb['best']} (MAE={r['MAE (USD)']:,.0f} USD, R²={r['R²']:.3f}).")
    return "\n".join(lines)


SYSTEM_PROMPT = """Bạn là **ITJobTrend AI** – trợ lý phân tích dữ liệu thông minh, thân thiện, giống Gemini, nằm trong ứng dụng đồ án
"Phân tích & dự báo xu hướng tuyển dụng ngành CNTT từ dữ liệu tin đăng việc làm".

Nguyên tắc:
1. Luôn trả lời bằng **tiếng Việt** (trừ khi người dùng hỏi bằng ngôn ngữ khác), rõ ràng, có cấu trúc (tiêu đề nhỏ, gạch đầu dòng, bảng Markdown khi phù hợp).
2. Với câu hỏi VỀ DỮ LIỆU: hãy GỌI CÁC CÔNG CỤ được cung cấp để tính số liệu chính xác thay vì đoán. Có thể gọi nhiều công cụ liên tiếp.
   Sau khi có số liệu, giải thích ý nghĩa (insight) và gợi ý cho sinh viên CNTT nếu hữu ích.
3. Với câu hỏi NGOÀI DỮ LIỆU (kiến thức chung, lập trình, học máy, viết báo cáo, định hướng nghề nghiệp…): trả lời bằng kiến thức của bạn.
4. Không bịa số liệu. Nếu công cụ báo lỗi, thử lại với tham số khác hoặc giải thích cho người dùng.
5. Nhớ giới hạn dữ liệu: LinkedIn là thị trường Mỹ, ảnh chụp 3–4/2024; Hacker News là cộng đồng startup/công nghệ, 01/2018–06/2023.
   Lương tính bằng USD/năm. Tên kỹ năng chuẩn: Python, JavaScript, TypeScript, React, Node.js, AWS, Azure, GCP, Docker, Kubernetes,
   SQL, PostgreSQL, Machine Learning, LLM/GenAI, Go, Rust, Java, C#, .NET, Ruby on Rails…
"""


# ============================ GỌI GEMINI ============================

class GeminiError(Exception):
    pass


def ask_gemini(api_key: str, model: str, history: list[dict], user_msg: str, data_context: str | None,
               use_tools: bool = True, use_search: bool = False, temperature: float = 0.7) -> str:
    """history: list các dict {role: 'user'|'assistant', content: str} (không gồm user_msg)."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:  # pragma: no cover
        raise GeminiError("Chưa cài thư viện google-genai. Chạy: pip install google-genai") from e

    client = genai.Client(api_key=api_key)
    contents = []
    for m in history[-20:]:
        role = "user" if m["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=m["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_msg)]))

    sys_prompt = SYSTEM_PROMPT
    if data_context:
        sys_prompt += "\n\n" + data_context
    else:
        sys_prompt += "\n\n(Hiện chưa có dữ liệu nào được tải lên, hoặc người dùng đã tắt kết nối dữ liệu.)"

    tools = None
    afc = None
    if use_search:
        tools = [types.Tool(google_search=types.GoogleSearch())]
    elif use_tools and data_context:
        tools = DATA_TOOLS
        afc = types.AutomaticFunctionCallingConfig(maximum_remote_calls=10)

    cfg = types.GenerateContentConfig(system_instruction=sys_prompt, temperature=temperature, tools=tools,
                                      automatic_function_calling=afc)
    try:
        resp = client.models.generate_content(model=model, contents=contents, config=cfg)
    except Exception as e:
        msg = str(e)
        if "API_KEY_INVALID" in msg or "API key not valid" in msg:
            raise GeminiError("API key không hợp lệ. Hãy kiểm tra lại GEMINI_API_KEY trong file .env.") from e
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            raise GeminiError("Đã vượt giới hạn số lần gọi (quota) của API key. Vui lòng đợi một lát rồi thử lại.") from e
        if "404" in msg or "NOT_FOUND" in msg:
            raise GeminiError(f"Không tìm thấy model '{model}'. Hãy chọn model khác ở thanh bên hoặc sửa GEMINI_MODEL trong .env.") from e
        raise GeminiError(f"Lỗi khi gọi Gemini: {msg[:400]}") from e
    text = (resp.text or "").strip() if resp is not None else ""
    if not text:
        text = "_(Gemini không trả về nội dung. Bạn thử diễn đạt lại câu hỏi nhé.)_"
    return text
