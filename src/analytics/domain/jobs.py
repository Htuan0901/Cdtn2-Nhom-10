"""Miền nghiệp vụ: dữ liệu tin đăng tuyển dụng (job postings).

Chức năng:
1. Tự ánh xạ cột (mapping) dựa trên tên + nội dung, người dùng có thể sửa lại.
2. Chuẩn hóa lương về "triệu VND / tháng" (min, max, mid) từ chuỗi tự do.
3. Chuẩn hóa yêu cầu kinh nghiệm về số năm.
4. Chuẩn hóa địa điểm về tỉnh/thành và vùng miền.
5. Nhận diện tin tuyển dụng ngành CNTT và nhóm vị trí (Backend, Data/AI, QA...).

Mọi quy tắc đều minh bạch (rule-based) để có thể giải thích trước hội đồng.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.analytics.config import (HOURS_PER_MONTH, SALARY_MAX_PLAUSIBLE, SALARY_MIN_PLAUSIBLE, USD_TO_VND,
                        WEEKS_PER_MONTH, WORKDAYS_PER_MONTH)
from src.analytics.data.schema import is_text_dtype, norm_name, strip_accents

# =============================================================== 1. Ánh xạ cột
FIELD_HINTS = {
    "title": ("job title", "title", "tieu de", "chuc danh", "ten cong viec", "job name", "position name"),
    "company": ("company", "cong ty", "employer", "doanh nghiep"),
    "salary": ("salary", "luong", "muc luong", "wage", "income"),
    "location": ("location", "dia diem", "noi lam viec", "city", "tinh", "province", "address", "dia chi", "khu vuc"),
    "experience": ("experience", "kinh nghiem", "exp", "years of experience"),
    "education": ("education", "hoc van", "bang cap", "trinh do", "degree"),
    "industry": ("industry", "nganh", "linh vuc", "category", "nganh nghe"),
    "level": ("job position", "cap bac", "level", "position", "vi tri", "seniority"),
    "job_type": ("job type", "hinh thuc", "loai cong viec", "employment type", "work type"),
    "description": ("description", "mo ta", "job description", "jd", "noi dung"),
    "requirements": ("requirement", "yeu cau", "skills", "ky nang", "qualification"),
    "time": ("year", "nam", "date", "ngay dang", "posted", "created", "time", "thoi gian"),
}
FIELD_LABELS = {
    "title": "Tiêu đề công việc", "company": "Công ty", "salary": "Mức lương", "location": "Địa điểm",
    "experience": "Kinh nghiệm", "education": "Học vấn", "industry": "Ngành nghề", "level": "Cấp bậc",
    "job_type": "Hình thức", "description": "Mô tả công việc", "requirements": "Yêu cầu", "time": "Thời gian",
}


def auto_map_columns(df: pd.DataFrame) -> dict[str, str | None]:
    """Chấm điểm từng cột với từng trường; mỗi cột chỉ gán cho một trường."""
    scores = []
    for col in df.columns:
        n = norm_name(col)
        for fld, hints in FIELD_HINTS.items():
            best = 0
            for h in hints:
                if n == h:
                    best = max(best, 3)
                elif h in n:
                    best = max(best, 2 if len(h) > 3 else 1)
            if best:
                scores.append((best, fld, col))
    mapping: dict[str, str | None] = {f: None for f in FIELD_HINTS}
    used: set[str] = set()
    for score, fld, col in sorted(scores, key=lambda t: -t[0]):
        if mapping[fld] is None and col not in used:
            mapping[fld] = col
            used.add(col)
    # Bổ sung theo nội dung: cột chứa nhiều "triệu"/"VND"/"usd" → lương
    if mapping["salary"] is None:
        for col in [c for c in df.columns if is_text_dtype(df[c])]:
            if col in used:
                continue
            sample = df[col].dropna().astype(str).head(200).str.lower()
            if sample.str.contains(r"tri[eệ]u|vnd|usd|\$").mean() > 0.4:
                mapping["salary"] = col
                used.add(col)
                break
    return mapping


def is_job_dataset(df: pd.DataFrame) -> bool:
    m = auto_map_columns(df)
    return bool(m.get("title")) and sum(1 for k in ("salary", "location", "experience", "description", "requirements") if m.get(k)) >= 2


# =============================================================== 2. Lương
_NEGOTIABLE = ("thoa thuan", "thuong luong", "canh tranh", "negotiable", "competitive", "dang cap nhat",
               "chua cap nhat", "upon", "agreement", "deal")
_NUM = re.compile(r"\d+(?:[.,]\d+)*")


@dataclass
class Salary:
    min: float = math.nan
    max: float = math.nan
    mid: float = math.nan
    status: str = "missing"   # ok | negotiable | missing | invalid
    unit: str = ""


def _to_number(token: str, unit: str) -> float:
    """'10.000.000' → 1e7 ; '9.5' (triệu) → 9.5 ; '1,5' → 1.5."""
    t = token
    groups = re.split(r"[.,]", t)
    if len(groups) > 1 and all(len(g) == 3 for g in groups[1:]):
        return float("".join(groups))             # dấu phân cách hàng nghìn
    return float(t.replace(",", "."))             # dấu thập phân


def parse_salary(text) -> Salary:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return Salary()
    raw = str(text).strip()
    s = strip_accents(raw).lower()
    if not s:
        return Salary()
    nums = _NUM.findall(s)
    if not nums:
        return Salary(status="negotiable" if any(k in s for k in _NEGOTIABLE) else "missing")

    if "usd" in s or "$" in s:
        unit, factor = "USD", USD_TO_VND / 1e6
    elif re.search(r"\btrieu\b|\btr\b|million", s):
        unit, factor = "trieu", 1.0
    elif re.search(r"nghin|ngan|\d\s*k\b", s):
        unit, factor = "nghin", 1e-3
    else:
        unit, factor = "VND", 1e-6

    period = 1.0
    if re.search(r"tuan|week", s):
        period = WEEKS_PER_MONTH
    elif re.search(r"/\s*ngay|\bngay\b|\bday\b", s):
        period = WORKDAYS_PER_MONTH
    elif re.search(r"gio|hour|/h\b", s):
        period = HOURS_PER_MONTH
    elif re.search(r"/\s*nam\b|year|annual", s):
        period = 1 / 12

    try:
        raw_vals = [_to_number(tok, unit) for tok in nums[:2]]
    except ValueError:
        return Salary(status="invalid", unit=unit)
    # chuỗi không ghi đơn vị và mọi số đều nhỏ (vd "15 - 20") → hiểu là triệu
    if unit == "VND" and max(raw_vals) < 1000:
        factor = 1.0
    values = [v * factor * period for v in raw_vals]
    values = [v for v in values if v > 0]
    if not values:
        return Salary(status="invalid", unit=unit)

    lo = hi = math.nan
    if len(values) >= 2:
        lo, hi = values[0], values[1]
        if hi < lo * 0.1:        # vd "99.999.999 - 9 VND": cận trên lỗi nhập liệu
            hi = math.nan
        elif hi < lo:
            lo, hi = hi, lo
    else:
        v = values[0]
        if re.search(r"tren|tu |from|\bover\b|\bmin\b|\+", s):
            lo = v
        elif re.search(r"toi|den|duoi|up to|upto|\bmax\b|lên đến|len den", s):
            hi = v
        else:
            lo = hi = v
    known = [v for v in (lo, hi) if not math.isnan(v)]
    mid = sum(known) / len(known)
    if not (SALARY_MIN_PLAUSIBLE <= mid <= SALARY_MAX_PLAUSIBLE):
        return Salary(status="invalid", unit=unit)
    return Salary(round(lo, 3) if not math.isnan(lo) else math.nan,
                  round(hi, 3) if not math.isnan(hi) else math.nan, round(mid, 3), "ok", unit)


def parse_salary_series(s: pd.Series) -> pd.DataFrame:
    uniq = s.dropna().astype(str).unique()
    parsed = {u: parse_salary(u) for u in uniq}
    keyed = s.astype(str).where(s.notna(), None)

    def col(attr):
        return keyed.map(lambda v: getattr(parsed[v], attr) if v in parsed else (math.nan if attr != "status" else "missing"))

    return pd.DataFrame({"salary_min": col("min"), "salary_max": col("max"),
                         "salary_mid": col("mid"), "salary_status": col("status")}, index=s.index)


# =============================================================== 3. Kinh nghiệm
def parse_experience(text) -> float:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return math.nan
    s = strip_accents(str(text)).lower().strip()
    if not s:
        return math.nan
    if re.search(r"khong yeu cau|chua co|khong can|no experience|^khong$|fresher|moi tot nghiep", s):
        return 0.0
    if re.search(r"duoi\s*1\s*nam|<\s*1", s):
        return 0.5
    nums = [float(x.replace(",", ".")) for x in re.findall(r"\d+(?:[.,]\d+)?", s)]
    if not nums:
        return math.nan
    v = nums[0]                     # dùng cận dưới của yêu cầu
    if "thang" in s or "month" in s:
        v = v / 12
    return v if 0 <= v <= 40 else math.nan


def experience_bucket(years: float) -> str:
    if years is None or (isinstance(years, float) and math.isnan(years)):
        return "Không rõ"
    if years == 0:
        return "0 năm"
    if years < 1:
        return "Dưới 1 năm"
    if years < 3:
        return "1-2 năm"
    if years < 5:
        return "3-4 năm"
    return "5+ năm"


EXP_ORDER = ["0 năm", "Dưới 1 năm", "1-2 năm", "3-4 năm", "5+ năm", "Không rõ"]

# =============================================================== 4. Địa điểm
PROVINCES = {
    "Hồ Chí Minh": ("ho chi minh", "hcm", "tp.hcm", "tphcm", "sai gon", "saigon", "thu duc", "phu nhuan", "tan binh",
                    "binh thanh", "go vap", "tan phu", "binh tan", "quan 1", "quan 3", "quan 7", "district 1", "district 7"),
    "Hà Nội": ("ha noi", "hanoi", "cau giay", "dong da", "hoan kiem", "ba dinh", "thanh xuan", "hai ba trung",
               "nam tu liem", "bac tu liem", "long bien", "hoang mai", "ha dong", "tay ho"),
    "Đà Nẵng": ("da nang", "danang"), "Hải Phòng": ("hai phong",), "Cần Thơ": ("can tho",),
    "Bình Dương": ("binh duong", "thu dau mot", "di an", "thuan an"), "Đồng Nai": ("dong nai", "bien hoa"),
    "Bà Rịa - Vũng Tàu": ("vung tau", "ba ria"), "Long An": ("long an",), "Bắc Ninh": ("bac ninh",),
    "Hưng Yên": ("hung yen",), "Hải Dương": ("hai duong",), "Vĩnh Phúc": ("vinh phuc",), "Quảng Ninh": ("quang ninh", "ha long"),
    "Thái Nguyên": ("thai nguyen",), "Bắc Giang": ("bac giang",), "Phú Thọ": ("phu tho",), "Hà Nam": ("ha nam",),
    "Nam Định": ("nam dinh",), "Thái Bình": ("thai binh",), "Ninh Bình": ("ninh binh",), "Thanh Hóa": ("thanh hoa",),
    "Nghệ An": ("nghe an", "vinh city"), "Hà Tĩnh": ("ha tinh",), "Quảng Bình": ("quang binh",), "Quảng Trị": ("quang tri",),
    "Thừa Thiên Huế": ("hue", "thua thien"), "Quảng Nam": ("quang nam", "hoi an"), "Quảng Ngãi": ("quang ngai",),
    "Bình Định": ("binh dinh", "quy nhon"), "Phú Yên": ("phu yen",), "Khánh Hòa": ("khanh hoa", "nha trang"),
    "Ninh Thuận": ("ninh thuan",), "Bình Thuận": ("binh thuan", "phan thiet"), "Kon Tum": ("kon tum",),
    "Gia Lai": ("gia lai", "pleiku"), "Đắk Lắk": ("dak lak", "dac lac", "buon ma thuot"), "Đắk Nông": ("dak nong",),
    "Lâm Đồng": ("lam dong", "da lat"), "Bình Phước": ("binh phuoc",), "Tây Ninh": ("tay ninh",),
    "Tiền Giang": ("tien giang", "my tho"), "Bến Tre": ("ben tre",), "Trà Vinh": ("tra vinh",), "Vĩnh Long": ("vinh long",),
    "Đồng Tháp": ("dong thap",), "An Giang": ("an giang", "long xuyen"), "Kiên Giang": ("kien giang", "phu quoc", "rach gia"),
    "Hậu Giang": ("hau giang",), "Sóc Trăng": ("soc trang",), "Bạc Liêu": ("bac lieu",), "Cà Mau": ("ca mau",),
    "Lạng Sơn": ("lang son",), "Cao Bằng": ("cao bang",), "Hà Giang": ("ha giang",), "Tuyên Quang": ("tuyen quang",),
    "Lào Cai": ("lao cai", "sa pa"), "Yên Bái": ("yen bai",), "Bắc Kạn": ("bac kan",), "Lai Châu": ("lai chau",),
    "Điện Biên": ("dien bien",), "Sơn La": ("son la",), "Hòa Bình": ("hoa binh",),
}
REGION = {
    "Bắc": {"Hà Nội", "Hải Phòng", "Bắc Ninh", "Hưng Yên", "Hải Dương", "Vĩnh Phúc", "Quảng Ninh", "Thái Nguyên",
            "Bắc Giang", "Phú Thọ", "Hà Nam", "Nam Định", "Thái Bình", "Ninh Bình", "Lạng Sơn", "Cao Bằng", "Hà Giang",
            "Tuyên Quang", "Lào Cai", "Yên Bái", "Bắc Kạn", "Lai Châu", "Điện Biên", "Sơn La", "Hòa Bình"},
    "Trung": {"Đà Nẵng", "Thanh Hóa", "Nghệ An", "Hà Tĩnh", "Quảng Bình", "Quảng Trị", "Thừa Thiên Huế", "Quảng Nam",
              "Quảng Ngãi", "Bình Định", "Phú Yên", "Khánh Hòa", "Ninh Thuận", "Bình Thuận", "Kon Tum", "Gia Lai",
              "Đắk Lắk", "Đắk Nông", "Lâm Đồng"},
}
_PROV_PATTERNS = sorted(((alias, prov) for prov, aliases in PROVINCES.items() for alias in aliases),
                        key=lambda t: -len(t[0]))
_PROV_RE = re.compile(r"(?<![a-z])(" + "|".join(re.escape(a) for a, _ in _PROV_PATTERNS) + r")(?![a-z])")
_ALIAS_TO_PROV = {a: p for a, p in _PROV_PATTERNS}


def normalize_province(text) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return "Không rõ"
    s = strip_accents(str(text)).lower()
    if re.search(r"remote|tu xa|work from home", s) and not _PROV_RE.search(s):
        return "Remote"
    for segment in s.split("|"):
        found = _PROV_RE.findall(segment)
        if found:
            return _ALIAS_TO_PROV[found[-1]]   # tỉnh thường nằm cuối địa chỉ
    if re.search(r"nuoc ngoai|overseas|japan|nhat ban|singapore|han quoc|korea", s):
        return "Nước ngoài"
    return "Khác"


def region_of(province: str) -> str:
    if province in REGION["Bắc"]:
        return "Miền Bắc"
    if province in REGION["Trung"]:
        return "Miền Trung"
    if province in PROVINCES:
        return "Miền Nam"
    return province


# =============================================================== 5. Nhận diện tin CNTT
_IT_INDUSTRY = re.compile(r"\bit\b|phan mem|phan cung|cong nghe thong tin|lap trinh|software|internet|"
                          r"data science|du lieu|thuong mai dien tu|game|an ninh mang|information technology", re.I)
_IT_TITLE = re.compile(
    r"developer|\bdev\b|lap trinh|programmer|software|phan mem|engineer.*(java|python|net|php|node|react|cloud|data|ai|ml)|"
    r"(java|python|php|golang|ruby|\.net|c#|c\+\+|node\.?js|react|angular|vue|flutter|ios|android|unity)\b|"
    r"front[\s-]?end|back[\s-]?end|full[\s-]?stack|devops|\bsre\b|cloud engineer|data (analyst|scientist|engineer)|"
    r"machine learning|tri tue nhan tao|tester|\btest(ing)? engineer|\bqa\b.*(engineer|automation|manual|phan mem)|qc phan mem|"
    r"automation test|it support|it helpdesk|helpdesk|ky thuat vien it|system admin|quan tri he thong|quan tri mang|"
    r"network engineer|security engineer|an ninh mang|pentest|brse|ky su cau noi|business analyst|"
    r"product owner|scrum master|ui/ux|ux/ui|ui ux|erp|sap\b|odoo|database|dba\b|blockchain|embedded|nhung|"
    r"game (developer|designer)|web developer|lap trinh vien|it manager|cto\b|tech lead|solution architect", re.I)
_IT_TITLE_UPPER = re.compile(r"\b(IT|AI|ML|BA|ERP|SAP)\b")
# Vai trò KHÔNG phải kỹ thuật CNTT dù đăng trong ngành IT (bán phần mềm, CNC/PLC trong sản xuất...)
_NON_IT_TITLE = re.compile(r"kinh doanh|sales|\bsale\b|ban hang|telesale|marketing|ke toan|tuyen sinh|cham soc khach hang|"
                           r"business develop|tu van ban|\bcnc\b|\bplc\b|may phay|may tien|cad/cam|van hanh may|"
                           r"thu ngan|le tan|tai xe|giao hang|bao ve|tap vu|phu bep|graphic|do hoa|video|editor|\bseo\b|nhap lieu|"
                           r"giao vien|quang cao|content|account|van phong|phong in|bao hanh|sua chua thiet bi|noi that|"
                           r"thiet ke cad|^nhan vien$|^(nhan vien|chuyen vien|thuc tap sinh|tuyen dung) thiet ke", re.I)

JOB_GROUP_RULES = [
    ("Data / AI", r"data|du lieu|machine learning|\bai\b|\bml\b|deep learning|computer vision|nlp|bi developer|power bi|analytics"),
    ("DevOps / Cloud", r"devops|\bsre\b|cloud|aws|azure|kubernetes|platform engineer|infrastructure"),
    ("QA / Tester", r"tester|\btest|\bqa\b|\bqc\b|kiem thu|quality assurance"),
    ("Mobile", r"mobile|android|ios|flutter|react native|kotlin|swift"),
    ("Fullstack", r"full[\s-]?stack"),
    ("Frontend", r"front[\s-]?end|reactjs|react\.js|angular|vue|ui developer|web developer"),
    ("Backend", r"back[\s-]?end|java\b|golang|\.net|c#|php|node|python|ruby|laravel|spring"),
    ("Embedded / IoT", r"embedded|nhung|firmware|iot|vi dieu khien"),
    ("Game", r"game|unity|unreal"),
    ("Security", r"security|bao mat|an ninh mang|pentest|soc analyst"),
    ("BA / PM / PO", r"business analyst|\bba\b|phan tich nghiep vu|product owner|product manager|project manager|scrum|\bpm\b|brse|cau noi"),
    ("UI/UX Design", r"ui/ux|ux/ui|ui ux|ux designer|ui designer|product designer"),
    ("IT Support / Network", r"support|helpdesk|ky thuat vien|system admin|quan tri|network|mang|it staff|nhan vien it"),
    ("ERP / SAP", r"\berp\b|\bsap\b|odoo|dynamics|oracle ebs"),
    ("Triển khai phần mềm", r"trien khai|implementation|consultant"),
    ("Software (chung)", r"software|phan mem|lap trinh|developer|programmer|engineer|tech lead|technical lead|architect|\bcto\b"),
    ("IT tổng hợp", r"\bit\b|cong nghe thong tin|may tinh|ky thuat"),
]
_GROUP_RES = [(g, re.compile(p, re.I)) for g, p in JOB_GROUP_RULES]


def is_it_posting(title, industry=None) -> bool:
    t_raw = "" if title is None or (isinstance(title, float) and math.isnan(title)) else str(title)
    i_raw = "" if industry is None or (isinstance(industry, float) and math.isnan(industry)) else str(industry)
    t, i = strip_accents(t_raw).lower(), strip_accents(i_raw).lower()
    if _NON_IT_TITLE.search(t):
        return False
    if _IT_TITLE.search(t) or _IT_TITLE_UPPER.search(t_raw):
        return True
    return bool(_IT_INDUSTRY.search(i))


def job_group(title) -> str:
    if title is None or (isinstance(title, float) and math.isnan(title)):
        return "Khác"
    t = strip_accents(str(title)).lower()
    for g, rx in _GROUP_RES:
        if rx.search(t):
            return g
    return "Khác"


# =============================================================== 6. Tạo đặc trưng
@dataclass
class FeatureReport:
    n_input: int = 0
    n_output: int = 0
    n_it: int = 0
    salary_status: dict = field(default_factory=dict)
    exp_parsed_pct: float = 0.0
    province_known_pct: float = 0.0
    created_columns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def build_job_features(df: pd.DataFrame, mapping: dict, it_only: bool = False) -> tuple[pd.DataFrame, FeatureReport]:
    rep = FeatureReport(n_input=len(df))
    out = df.copy()
    created = []
    title_col, ind_col = mapping.get("title"), mapping.get("industry")

    if title_col:
        titles = out[title_col]
        inds = out[ind_col] if ind_col else pd.Series([None] * len(out), index=out.index)
        pair_keys = titles.astype(str) + "␟" + inds.astype(str)
        uniq = pd.DataFrame({"t": titles, "i": inds, "k": pair_keys}).drop_duplicates("k")
        lookup = {k: is_it_posting(t, i) for t, i, k in uniq.itertuples(index=False)}
        out["is_it"] = pair_keys.map(lookup).astype(bool)
        gl = {t: job_group(t) for t in titles.dropna().unique()}
        out["job_group"] = titles.map(gl).fillna("Khác")
        out.loc[~out["is_it"], "job_group"] = "Ngoài CNTT"
        created += ["is_it", "job_group"]
        rep.n_it = int(out["is_it"].sum())
        if it_only:
            out = out[out["is_it"]].reset_index(drop=True)
            rep.notes.append(f"Đã lọc chỉ giữ {len(out):,} tin CNTT (từ {rep.n_input:,} tin).")

    if mapping.get("salary") and pd.api.types.is_numeric_dtype(out[mapping["salary"]]):
        # cột lương đã là số (vd LinkedIn: USD/năm) ⇒ giữ nguyên giá trị và đơn vị gốc, không áp quy tắc chuỗi VND
        v = pd.to_numeric(out[mapping["salary"]], errors="coerce")
        out["salary_min"] = out["salary_max"] = out["salary_mid"] = v
        out["salary_status"] = np.where(v.notna(), "ok", "missing")
        created += ["salary_min", "salary_max", "salary_mid", "salary_status"]
        rep.salary_status = out["salary_status"].value_counts().to_dict()
        rep.notes.append(f"Cột lương '{mapping['salary']}' đã là số ⇒ giữ nguyên đơn vị gốc (không quy đổi sang triệu VND).")
    elif mapping.get("salary"):
        sal = parse_salary_series(out[mapping["salary"]])
        for c in sal.columns:
            out[c] = sal[c].values
        created += list(sal.columns)
        rep.salary_status = out["salary_status"].value_counts().to_dict()
        rep.notes.append("Lương quy đổi về triệu VND/tháng; USD quy đổi theo tỷ giá %s VND (cấu hình USD_TO_VND)." % f"{USD_TO_VND:,.0f}")

    if mapping.get("experience"):
        uniq = {v: parse_experience(v) for v in out[mapping["experience"]].dropna().unique()}
        out["exp_years"] = out[mapping["experience"]].map(uniq).astype(float)
        out["exp_bucket"] = out["exp_years"].map(experience_bucket)
        created += ["exp_years", "exp_bucket"]
        rep.exp_parsed_pct = round(100 * out["exp_years"].notna().mean(), 2)

    if mapping.get("location"):
        uniq = {v: normalize_province(v) for v in out[mapping["location"]].dropna().unique()}
        out["province"] = out[mapping["location"]].map(uniq).fillna("Không rõ")
        out["region"] = out["province"].map(region_of)
        created += ["province", "region"]
        rep.province_known_pct = round(100 * (~out["province"].isin(["Không rõ", "Khác"])).mean(), 2)

    rep.n_output = len(out)
    rep.created_columns = created
    return out, rep
