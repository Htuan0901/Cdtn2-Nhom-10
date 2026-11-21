"""Pipeline thu thập → làm sạch → chuẩn hóa → trích xuất đặc trưng cho 2 nguồn dữ liệu Kaggle.

Nguồn 1 – LinkedIn Job Postings 2023–2024 (arshkon/linkedin-job-postings): ~124 nghìn tin, có lương, kỹ năng, địa điểm.
Nguồn 2 – Hacker News "Who is hiring" 2018–2023 (kidrulit): ~47 nghìn tin IT theo tháng → chuỗi thời gian.

Mỗi bước làm sạch ghi lại số dòng trước/sau vào nhật ký (log) để báo cáo minh bạch.
"""
from __future__ import annotations

import json
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR
from .skills import GENERIC_SKILLS, IT_TITLE_PATTERN, classify_role, extract_skills, seniority

LI_FILE = PROCESSED_DIR / "linkedin_it.csv.gz"
HN_FILE = PROCESSED_DIR / "hn_jobs.csv.gz"
LOG_FILE = PROCESSED_DIR / "clean_log.json"

EXP_MAP = {
    "Internship": "Thực tập", "Entry level": "Mới đi làm (Entry)", "Associate": "Associate",
    "Mid-Senior level": "Mid-Senior", "Director": "Director", "Executive": "Executive",
}
EXP_ORDER = ["Thực tập", "Mới đi làm (Entry)", "Associate", "Mid-Senior", "Director", "Executive", "Không rõ"]
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], start=1)}
COUNTRY_ALIASES = {"usa": "United States", "us": "United States", "united states of america": "United States",
                   "u.s.": "United States", "uk": "United Kingdom", "england": "United Kingdom", "deutschland": "Germany"}


class Logger:
    def __init__(self, source: str):
        self.source, self.rows = source, []

    def add(self, step: str, before: int, after: int, note: str = ""):
        self.rows.append({"Nguồn": self.source, "Bước": step, "Trước": int(before), "Sau": int(after),
                          "Bị loại": int(before - after), "Ghi chú": note})


# ============================ ĐỌC FILE GỐC ============================

def _open_member(src, name: str):
    """Mở file `name` trong thư mục hoặc file zip `src` (tìm theo đuôi đường dẫn)."""
    src = Path(src) if not hasattr(src, "read") else src
    if hasattr(src, "read") or str(src).lower().endswith(".zip"):
        z = zipfile.ZipFile(src)
        for n in z.namelist():
            if n.replace("\\", "/").endswith(name):
                return z.open(n)
        raise FileNotFoundError(f"Không thấy {name} trong file zip")
    hits = sorted(Path(src).rglob(Path(name).name))
    hits = [h for h in hits if str(h).replace("\\", "/").endswith(name)]
    if not hits:
        raise FileNotFoundError(f"Không thấy {name} trong thư mục {src}")
    return open(hits[0], "rb")


# ============================ LINKEDIN ============================

LI_COLS = ["job_id", "company_name", "title", "description", "location", "formatted_experience_level",
           "formatted_work_type", "remote_allowed", "normalized_salary", "currency", "pay_period", "listed_time",
           "original_listed_time", "views", "applies", "skills_desc"]


def clean_linkedin(src, progress=None) -> tuple[pd.DataFrame, list[dict]]:
    log = Logger("LinkedIn")
    t0 = time.time()
    with _open_member(src, "postings.csv") as f:
        df = pd.read_csv(f, usecols=lambda c: c in LI_COLS)
    with _open_member(src, "jobs/job_skills.csv") as f:
        js = pd.read_csv(f)
    if progress:
        progress(0.25, "Đã đọc postings.csv")
    n0 = len(df)
    log.add("Dữ liệu gốc (postings.csv)", n0, n0, f"{df.shape[1]} cột được dùng")

    n = len(df)
    df = df.drop_duplicates("job_id")
    log.add("Xóa trùng job_id", n, len(df))

    n = len(df)
    df = df.dropna(subset=["title", "description"])
    df = df[df["description"].str.len() >= 50]
    log.add("Bỏ tin thiếu tiêu đề/mô tả", n, len(df), "mô tả < 50 ký tự coi như thiếu")

    n = len(df)
    df = df.drop_duplicates(["title", "company_name", "description"])
    log.add("Xóa tin đăng lặp lại", n, len(df), "trùng tiêu đề + công ty + mô tả")

    # ---- Lọc tin ngành CNTT ----
    # Giữ tin nếu: (a) tiêu đề chứa từ khóa CNTT, HOẶC (b) LinkedIn gắn nhãn chức năng IT VÀ mô tả yêu cầu ≥ 3 kỹ năng
    # công nghệ cụ thể (nhãn IT của LinkedIn khá nhiễu: có cả tin bán hàng, điều phối… nên cần thêm điều kiện b).
    n = len(df)
    it_ids = set(js.loc[js["skill_abr"] == "IT", "job_id"])
    by_title = df["title"].str.contains(IT_TITLE_PATTERN, na=False)
    by_func = df["job_id"].isin(it_ids)
    df = df[by_title | by_func].copy()
    by_title, by_func = by_title[df.index], by_func[df.index]
    text = df["title"] + " \n " + df["description"] + " \n " + df["skills_desc"].fillna("")
    df["skills"] = text.map(lambda t: ";".join(extract_skills(t)))
    n_tech = df["skills"].map(lambda s: len([x for x in s.split(";") if x and x not in GENERIC_SKILLS]))
    keep = by_title | (by_func & (n_tech >= 3))
    df = df[keep].copy()
    log.add("Lọc tin ngành CNTT", n, len(df),
            f"{int(by_title.sum()):,} khớp tiêu đề; {int((keep & ~by_title).sum()):,} thêm từ nhãn IT + ≥3 kỹ năng công nghệ")
    if progress:
        progress(0.6, "Đã lọc tin CNTT và trích xuất kỹ năng")

    # ---- Chuẩn hóa lương về USD/năm ----
    n_sal = df["normalized_salary"].notna().sum()
    df["salary_usd"] = np.where(df["currency"].fillna("USD").eq("USD"), df["normalized_salary"], np.nan)
    bad = (df["salary_usd"] < 15_000) | (df["salary_usd"] > 600_000)
    df.loc[bad, "salary_usd"] = np.nan
    log.add("Chuẩn hóa lương (USD/năm)", len(df), len(df),
            f"{n_sal:,} tin có lương; loại {int(bad.sum()):,} giá trị ngoại lai (<15k hoặc >600k USD)")

    # ---- Chuẩn hóa địa điểm, kinh nghiệm, hình thức ----
    loc = df["location"].fillna("")
    df["state"] = loc.str.extract(r",\s*([A-Z]{2})$")[0]
    df.loc[df["state"].isna() & loc.str.contains("United States"), "state"] = "Toàn quốc"
    df["state"] = df["state"].fillna("Khác")
    df["city"] = loc.str.extract(r"^([^,]+),\s*[A-Z]{2}$")[0]
    df["experience"] = df["formatted_experience_level"].map(EXP_MAP).fillna("Không rõ")
    df["work_type"] = df["formatted_work_type"].fillna("Không rõ")
    df["remote"] = df["remote_allowed"].fillna(0).astype(int).astype(bool)
    df["listed_date"] = pd.to_datetime(df["original_listed_time"].fillna(df["listed_time"]), unit="ms").dt.normalize()
    log.add("Chuẩn hóa địa điểm/kinh nghiệm/ngày đăng", len(df), len(df), "tách bang (state), map cấp bậc sang tiếng Việt")

    # ---- Kỹ năng (đã trích xuất ở bước lọc) + phân loại vị trí ----
    df["n_skills"] = df["skills"].map(lambda s: 0 if not s else s.count(";") + 1)
    df["role"] = df["title"].map(classify_role)
    log.add("Trích xuất kỹ năng & phân loại vị trí", len(df), len(df),
            f"TB {df['n_skills'].mean():.1f} kỹ năng/tin; {(df['n_skills'] > 0).mean() * 100:.0f}% tin có ≥1 kỹ năng")
    if progress:
        progress(0.7, "Đã trích xuất kỹ năng LinkedIn")

    df["desc_excerpt"] = df["description"].str.replace(r"\s+", " ", regex=True).str.slice(0, 700)
    keep = ["job_id", "title", "company_name", "role", "experience", "work_type", "remote", "location", "state", "city",
            "salary_usd", "pay_period", "listed_date", "views", "applies", "skills", "n_skills", "desc_excerpt"]
    df = df[keep].reset_index(drop=True)
    log.add("Kết quả LinkedIn (CNTT)", n0, len(df), f"{time.time() - t0:.0f} giây")
    return df, log.rows


# ============================ HACKER NEWS ============================

def _first_item(s: str) -> str | None:
    if not isinstance(s, str):
        return None
    items = re.findall(r'"([^"]+)"|([^{},]+)', s)
    vals = [a or b for a, b in items if (a or b).strip()]
    return vals[0].strip() if vals else None


def _norm_country(c: str | None) -> str:
    if not isinstance(c, str) or not c.strip():
        return "Không rõ"
    k = c.strip().lower()
    return COUNTRY_ALIASES.get(k, c.strip().title() if c.islower() else c.strip())


_SAL_RE = re.compile(
    r"\$\s?(\d{2,3})(?:[,.](\d{3}))?\s?(k|K)?\s?(?:-|–|—|to)\s?\$?\s?(\d{2,3})(?:[,.](\d{3}))?\s?(k|K)?")


def parse_salary(text: str) -> float | None:
    """Tìm khoảng lương dạng $120k-$160k hoặc $120,000 - $160,000 trong tin HN. Trả về trung điểm (USD/năm)."""
    if not isinstance(text, str):
        return None
    m = _SAL_RE.search(text)
    if not m:
        return None
    lo = float(m.group(1)) * 1000 if (m.group(3) or m.group(2)) else None
    hi = float(m.group(4)) * 1000 if (m.group(6) or m.group(5)) else None
    if lo is None or hi is None:
        return None
    mid = (lo + hi) / 2
    return mid if 30_000 <= mid <= 500_000 and hi >= lo else None


def clean_hn(src, progress=None) -> tuple[pd.DataFrame, list[dict]]:
    log = Logger("Hacker News")
    t0 = time.time()
    with _open_member(src, "jobs_2018_2023.csv") as f:
        jobs = pd.read_csv(f)
    parsed = None
    try:
        with _open_member(src, "parsed_jobs_2018_2023.csv") as f:
            parsed = pd.read_csv(f, usecols=["id", "company_name", "location_country", "is_remote", "position",
                                             "language_stack", "subindustry"])
    except FileNotFoundError:
        pass
    n0 = len(jobs)
    log.add("Dữ liệu gốc (jobs_2018_2023.csv)", n0, n0)

    n = len(jobs)
    jobs = jobs.dropna(subset=["job_description"])
    jobs = jobs[jobs["job_description"].str.len() >= 40]
    log.add("Bỏ tin rỗng/quá ngắn", n, len(jobs), "< 40 ký tự")

    n = len(jobs)
    jobs = jobs.drop_duplicates("id").drop_duplicates(["hn_hiring_year", "job_description"])
    log.add("Xóa tin trùng", n, len(jobs), "trùng id hoặc trùng nội dung trong cùng năm")

    n = len(jobs)
    jobs["month"] = pd.to_datetime(dict(year=jobs["hn_hiring_year"],
                                        month=jobs["hn_hiring_month"].str.lower().map(MONTHS), day=1), errors="coerce")
    jobs = jobs.dropna(subset=["month"])
    log.add("Chuẩn hóa tháng đăng (thread)", n, len(jobs), "ghép hn_hiring_month + hn_hiring_year")

    if parsed is not None:
        jobs = jobs.merge(parsed.drop_duplicates("id"), on="id", how="left")
        jobs["country"] = jobs["location_country"].map(_first_item).map(_norm_country)
        jobs["remote"] = jobs["is_remote"].astype(str).str.lower().eq("true")
        jobs["position"] = jobs["position"].map(_first_item)
        stack = jobs["language_stack"].fillna("").str.replace(r'[{}"]', " ", regex=True)
        log.add("Ghép thông tin đã tách (parsed_jobs)", len(jobs), len(jobs), "quốc gia, remote, vị trí, tech stack")
    else:
        jobs["country"], jobs["remote"], jobs["position"], stack = "Không rõ", False, None, ""
        jobs["company_name"], jobs["subindustry"] = None, None
    if progress:
        progress(0.85, "Đang trích xuất kỹ năng HN")

    text = jobs["job_description"] + " \n " + stack
    jobs["skills"] = text.map(lambda t: ";".join(extract_skills(t)))
    jobs["n_skills"] = jobs["skills"].map(lambda s: 0 if not s else s.count(";") + 1)
    first_line = jobs["job_description"].str.split("|").str[:3].str.join(" ")
    title_src = jobs["position"].fillna(first_line)
    jobs["role"] = title_src.map(classify_role)
    jobs["seniority"] = title_src.map(seniority)
    jobs["salary_usd"] = jobs["job_description"].map(parse_salary)
    jobs["remote"] = jobs["remote"] | jobs["job_description"].str.contains(r"\bremote\b", case=False)
    log.add("Trích xuất kỹ năng, vị trí, lương", len(jobs), len(jobs),
            f"TB {jobs['n_skills'].mean():.1f} kỹ năng/tin; {jobs['salary_usd'].notna().mean() * 100:.0f}% tin ghi lương")

    jobs["desc_excerpt"] = jobs["job_description"].str.replace(r"\s+", " ", regex=True).str.slice(0, 600)
    keep = ["id", "month", "company_name", "position", "role", "seniority", "country", "remote", "subindustry",
            "salary_usd", "skills", "n_skills", "desc_excerpt"]
    jobs = jobs[keep].sort_values("month").reset_index(drop=True)
    log.add("Kết quả Hacker News", n0, len(jobs), f"{time.time() - t0:.0f} giây")
    return jobs, log.rows


# ============================ LƯU / NẠP ============================

def save_processed(li: pd.DataFrame, hn: pd.DataFrame, log_rows: list[dict]):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    li.to_csv(LI_FILE, index=False, compression="gzip")
    hn.to_csv(HN_FILE, index=False, compression="gzip")
    LOG_FILE.write_text(json.dumps(log_rows, ensure_ascii=False, indent=1), encoding="utf-8")


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    if not (LI_FILE.exists() and HN_FILE.exists()):
        return None
    li = pd.read_csv(LI_FILE, parse_dates=["listed_date"], dtype={"skills": str})
    hn = pd.read_csv(HN_FILE, parse_dates=["month"], dtype={"skills": str})
    li["skills"] = li["skills"].fillna("")
    hn["skills"] = hn["skills"].fillna("")
    log = pd.DataFrame(json.loads(LOG_FILE.read_text(encoding="utf-8"))) if LOG_FILE.exists() else pd.DataFrame()
    return li, hn, log


def explode_skills(df: pd.DataFrame, extra_cols: list[str] | None = None) -> pd.DataFrame:
    """Mỗi (tin, kỹ năng) một dòng – tiện cho thống kê."""
    cols = ["skills"] + (extra_cols or [])
    e = df[cols].assign(skill=df["skills"].str.split(";")).explode("skill")
    return e[e["skill"].notna() & (e["skill"] != "")].drop(columns="skills")
