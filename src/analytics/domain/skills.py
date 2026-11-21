"""Text mining: trích xuất & chuẩn hóa kỹ năng/công nghệ từ mô tả công việc.

Phương pháp: từ điển kỹ năng (skill dictionary) + biểu thức chính quy có ranh giới từ.
- Mỗi kỹ năng có nhiều biến thể (ReactJS, React.js, React → "React").
- Một tin được đếm tối đa 1 lần cho mỗi kỹ năng (đếm theo sự xuất hiện - presence).
- Ranh giới tùy biến để "SQL" không khớp trong "MySQL", "Java" không khớp "JavaScript".
Ưu điểm: minh bạch, nhanh, giải thích được. Hạn chế: không bắt được kỹ năng ngoài từ điển.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

# canonical: (category, [variants])
SKILLS: dict[str, tuple[str, list[str]]] = {
    # Ngôn ngữ lập trình
    "Java": ("Ngôn ngữ", ["java", "java core", "core java", "j2ee", "java ee"]),
    "JavaScript": ("Ngôn ngữ", ["javascript", "java script", "es6"]),
    "TypeScript": ("Ngôn ngữ", ["typescript"]),
    "Python": ("Ngôn ngữ", ["python", "python3"]),
    "C#": ("Ngôn ngữ", ["c#", "c sharp", "csharp"]),
    "C++": ("Ngôn ngữ", ["c++", "cpp", "c/c++"]),
    "C": ("Ngôn ngữ", ["ngôn ngữ c", "lập trình c", "c/c++", "c language", "embedded c"]),
    "PHP": ("Ngôn ngữ", ["php"]),
    "Go": ("Ngôn ngữ", ["golang", "go lang", "go language"]),
    "Ruby": ("Ngôn ngữ", ["ruby", "ruby on rails", "rails"]),
    "Kotlin": ("Ngôn ngữ", ["kotlin"]),
    "Swift": ("Ngôn ngữ", ["swift", "swiftui"]),
    "Dart": ("Ngôn ngữ", ["dart"]),
    "Rust": ("Ngôn ngữ", ["rust"]),
    "Scala": ("Ngôn ngữ", ["scala"]),
    "R": ("Ngôn ngữ", ["ngôn ngữ r", "r language", "rstudio", "r programming"]),
    "SQL": ("Cơ sở dữ liệu", ["sql", "t-sql", "pl/sql", "plsql", "tsql"]),
    "HTML/CSS": ("Web", ["html", "html5", "css", "css3", "scss", "sass", "tailwind", "bootstrap"]),
    # Framework / thư viện
    "React": ("Framework", ["react", "reactjs", "react.js", "react js"]),
    "React Native": ("Framework", ["react native", "react-native"]),
    "Angular": ("Framework", ["angular", "angularjs", "angular.js"]),
    "Vue.js": ("Framework", ["vue", "vuejs", "vue.js", "nuxt", "nuxtjs"]),
    "Next.js": ("Framework", ["next.js", "nextjs"]),
    "Node.js": ("Framework", ["node.js", "nodejs", "node js", "expressjs", "express.js", "nestjs", "nest.js"]),
    "Spring": ("Framework", ["spring", "spring boot", "springboot", "spring framework", "spring mvc"]),
    ".NET": ("Framework", [".net", "dotnet", "asp.net", ".net core", "asp.net core", "net core"]),
    "Django": ("Framework", ["django"]),
    "Flask/FastAPI": ("Framework", ["flask", "fastapi"]),
    "Laravel": ("Framework", ["laravel"]),
    "Flutter": ("Framework", ["flutter"]),
    "jQuery": ("Framework", ["jquery"]),
    "Unity": ("Framework", ["unity", "unity3d"]),
    # Cơ sở dữ liệu
    "MySQL": ("Cơ sở dữ liệu", ["mysql", "mariadb"]),
    "PostgreSQL": ("Cơ sở dữ liệu", ["postgresql", "postgres", "postgre"]),
    "SQL Server": ("Cơ sở dữ liệu", ["sql server", "mssql", "ms sql"]),
    "Oracle": ("Cơ sở dữ liệu", ["oracle", "oracle db"]),
    "MongoDB": ("Cơ sở dữ liệu", ["mongodb", "mongo"]),
    "Redis": ("Cơ sở dữ liệu", ["redis"]),
    "Elasticsearch": ("Cơ sở dữ liệu", ["elasticsearch", "elastic search", "elk"]),
    # Cloud / DevOps
    "AWS": ("Cloud/DevOps", ["aws", "amazon web services", "ec2"]),
    "Azure": ("Cloud/DevOps", ["azure", "microsoft azure"]),
    "GCP": ("Cloud/DevOps", ["gcp", "google cloud"]),
    "Docker": ("Cloud/DevOps", ["docker"]),
    "Kubernetes": ("Cloud/DevOps", ["kubernetes", "k8s", "k3s"]),
    "CI/CD": ("Cloud/DevOps", ["ci/cd", "cicd", "jenkins", "gitlab ci", "github actions"]),
    "Linux": ("Cloud/DevOps", ["linux", "ubuntu", "centos", "unix"]),
    "Git": ("Công cụ", ["git", "github", "gitlab", "bitbucket", "svn"]),
    "Microservices": ("Kiến trúc", ["microservice", "microservices", "micro service"]),
    "API/RESTful": ("Kiến trúc", ["restful", "rest api", "restful api", "api", "web api", "graphql"]),
    "Kafka/RabbitMQ": ("Kiến trúc", ["kafka", "rabbitmq", "message queue"]),
    # Data / AI
    "Machine Learning": ("Data/AI", ["machine learning"]),
    "Deep Learning": ("Data/AI", ["deep learning", "học sâu", "neural network"]),
    "AI": ("Data/AI", ["trí tuệ nhân tạo", "artificial intelligence", "generative ai"]),
    "LLM": ("Data/AI", ["llm", "chatgpt", "openai", "langchain", "rag", "prompt engineering"]),
    "TensorFlow/PyTorch": ("Data/AI", ["tensorflow", "pytorch", "keras"]),
    "Pandas/NumPy": ("Data/AI", ["pandas", "numpy", "scikit-learn", "sklearn"]),
    "Spark/Hadoop": ("Data/AI", ["spark", "pyspark", "hadoop", "databricks"]),
    "Power BI": ("Data/AI", ["power bi", "powerbi"]),
    "Tableau": ("Data/AI", ["tableau"]),
    "Excel": ("Công cụ", ["excel", "ms excel", "microsoft excel"]),
    "ETL/Data Warehouse": ("Data/AI", ["etl", "data warehouse", "data lake", "lakehouse", "dwh", "airflow"]),
    # Kiểm thử & quy trình
    "Selenium/Automation Test": ("Kiểm thử", ["selenium", "automation test", "automation testing", "cypress", "playwright", "appium"]),
    "Manual Test": ("Kiểm thử", ["manual test", "manual testing", "test case", "testcase"]),
    "Agile/Scrum": ("Quy trình", ["agile", "scrum", "kanban"]),
    "Jira": ("Công cụ", ["jira", "confluence"]),
    "Figma": ("Công cụ", ["figma", "adobe xd"]),
    "SAP/ERP": ("Hệ thống", ["sap", "erp", "odoo"]),
    "Networking": ("Hạ tầng", ["ccna", "ccnp", "tcp/ip", "lan/wan", "firewall", "mạng máy tính", "network engineer"]),
    "Security": ("Hạ tầng", ["an ninh mạng", "pentest", "owasp", "cybersecurity", "cyber security", "bảo mật thông tin", "information security"]),
    # Ngoại ngữ
    "Tiếng Anh": ("Ngoại ngữ", ["tiếng anh", "english", "toeic", "ielts"]),
    "Tiếng Nhật": ("Ngoại ngữ", ["tiếng nhật", "japanese", "jlpt"]),
    "Tiếng Trung": ("Ngoại ngữ", ["tiếng trung", "chinese", "tiếng hoa", "hsk"]),
    "Tiếng Hàn": ("Ngoại ngữ", ["tiếng hàn", "korean", "topik"]),
}

# Các từ viết tắt dễ trùng từ tiếng Việt thường (vd "ai" = "người nào") chỉ khớp khi VIẾT HOA trong văn bản gốc.
CASE_SENSITIVE = {"AI": ["AI", "GenAI"], "Machine Learning": ["ML"], "JavaScript": ["JS"], "TypeScript": ["TS"]}

_BOUNDARY_L = r"(?<![a-z0-9à-ỹđ+#./])"
_BOUNDARY_R = r"(?![a-z0-9à-ỹđ+#])"


def _build():
    variant_map: dict[str, set[str]] = {}
    for canon, (_, variants) in SKILLS.items():
        for v in variants:
            variant_map.setdefault(v.lower(), set()).add(canon)
    alts = sorted(variant_map, key=len, reverse=True)
    pattern = re.compile(_BOUNDARY_L + "(" + "|".join(re.escape(a) for a in alts) + ")" + _BOUNDARY_R)
    return pattern, variant_map


_PATTERN, _VARIANTS = _build()
_CS_MAP = {v: canon for canon, vs in CASE_SENSITIVE.items() for v in vs}
_CS_PATTERN = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(sorted(map(re.escape, _CS_MAP), key=len, reverse=True)) + r")(?![A-Za-z0-9])")
CATEGORY = {k: v[0] for k, v in SKILLS.items()}


def extract_skills(text) -> set[str]:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return set()
    original = str(text)
    s = re.sub(r"\s+", " ", original.lower())
    found: set[str] = set()
    for m in _PATTERN.findall(s):
        found |= _VARIANTS.get(m, set())
    for m in _CS_PATTERN.findall(original):
        found.add(_CS_MAP[m])
    return found


def extract_skills_frame(df: pd.DataFrame, text_cols: list[str]) -> pd.Series:
    """Trả về Series các set kỹ năng cho từng tin (ghép nhiều cột văn bản)."""
    cols = [c for c in text_cols if c and c in df.columns]
    if not cols:
        return pd.Series([set()] * len(df), index=df.index)
    combined = df[cols].fillna("").astype(str).agg(" \n ".join, axis=1)
    return combined.map(extract_skills)


def skill_counts(skill_sets: pd.Series) -> pd.DataFrame:
    n = max(len(skill_sets), 1)
    c = Counter()
    for s in skill_sets:
        c.update(s)
    rows = [{"Kỹ năng": k, "Nhóm": CATEGORY.get(k, "Khác"), "Số tin": v, "Tỷ lệ tin (%)": round(100 * v / n, 2)}
            for k, v in c.most_common()]
    return pd.DataFrame(rows, columns=["Kỹ năng", "Nhóm", "Số tin", "Tỷ lệ tin (%)"])


def skill_matrix(skill_sets: pd.Series, skills: list[str]) -> pd.DataFrame:
    return pd.DataFrame({s: skill_sets.map(lambda x, s=s: s in x).astype(np.int8) for s in skills}, index=skill_sets.index)


def skill_share_by_period(skill_sets: pd.Series, periods: pd.Series, skills: list[str]) -> pd.DataFrame:
    """Tỷ lệ (%) tin có nhắc tới từng kỹ năng theo kỳ (chuẩn hóa theo số tin mỗi kỳ ⇒ so sánh được giữa các kỳ)."""
    m = skill_matrix(skill_sets, skills)
    m["_period"] = periods.values
    share = m.groupby("_period")[skills].mean() * 100
    share["Số tin"] = m.groupby("_period").size()
    return share.round(3)


def cooccurrence(skill_sets: pd.Series, skills: list[str]) -> pd.DataFrame:
    m = skill_matrix(skill_sets, skills).astype(np.int32)
    co = m.T @ m
    return co


def salary_premium(skill_sets: pd.Series, salary: pd.Series, skills: list[str], min_n: int = 30) -> pd.DataFrame:
    """So sánh trung vị lương của tin CÓ và KHÔNG có kỹ năng (tương quan, không phải nhân quả)."""
    rows = []
    valid = salary.notna()
    for s in skills:
        has = skill_sets.map(lambda x, s=s: s in x) & valid
        no = (~skill_sets.map(lambda x, s=s: s in x)) & valid
        if has.sum() < min_n or no.sum() < min_n:
            continue
        med_has, med_no = float(salary[has].median()), float(salary[no].median())
        rows.append({"Kỹ năng": s, "Số tin có lương": int(has.sum()), "Trung vị lương (có)": round(med_has, 2),
                     "Trung vị lương (không)": round(med_no, 2), "Chênh lệch (triệu)": round(med_has - med_no, 2),
                     "Chênh lệch (%)": round(100 * (med_has - med_no) / med_no, 1) if med_no else np.nan})
    return pd.DataFrame(rows).sort_values("Chênh lệch (triệu)", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def forecast_skill_shares(share: pd.DataFrame, skills: list[str], exclude_periods: list | None = None) -> dict:
    """Dự báo tỷ lệ kỹ năng kỳ kế tiếp bằng xu hướng tuyến tính, đánh giá backtest so với baseline Naive.

    Backtest: với mỗi kỹ năng, huấn luyện trên tất cả kỳ trừ kỳ cuối, dự báo kỳ cuối;
    so sánh sai số tuyệt đối (điểm %) của Linear Trend với Naive (giữ nguyên giá trị kỳ trước).
    """
    data = share.drop(index=exclude_periods or [], errors="ignore")
    periods = list(data.index)
    if len(periods) < 3:
        return {"ok": False, "reason": f"Cần ít nhất 3 kỳ để backtest, hiện có {len(periods)}."}
    x = np.arange(len(periods), dtype=float)
    rows, err_lin, err_naive = [], [], []
    for s in skills:
        y = data[s].to_numpy(dtype=float)
        coef = np.polyfit(x[:-1], y[:-1], 1)
        pred_lin = float(np.polyval(coef, x[-1]))
        pred_naive = float(y[-2])
        err_lin.append(abs(pred_lin - y[-1]))
        err_naive.append(abs(pred_naive - y[-1]))
        coef_full = np.polyfit(x, y, 1)
        nxt = float(np.clip(np.polyval(coef_full, len(periods)), 0, 100))
        rows.append({"Kỹ năng": s, "Tỷ lệ kỳ cuối (%)": round(y[-1], 2), "Độ dốc (điểm %/kỳ)": round(coef_full[0], 3),
                     "Dự báo kỳ tới (%)": round(nxt, 2), "Xu hướng": "Tăng" if coef_full[0] > 0.1 else ("Giảm" if coef_full[0] < -0.1 else "Ổn định"),
                     "Lỗi backtest Linear": round(abs(pred_lin - y[-1]), 3), "Lỗi backtest Naive": round(abs(pred_naive - y[-1]), 3)})
    mae_lin, mae_naive = float(np.mean(err_lin)), float(np.mean(err_naive))
    rmse_lin = float(np.sqrt(np.mean(np.square(err_lin))))
    rmse_naive = float(np.sqrt(np.mean(np.square(err_naive))))
    return {"ok": True, "periods": [str(p) for p in periods], "table": pd.DataFrame(rows),
            "mae_linear": round(mae_lin, 3), "mae_naive": round(mae_naive, 3),
            "rmse_linear": round(rmse_lin, 3), "rmse_naive": round(rmse_naive, 3),
            "test_period": str(periods[-1]), "better": "Linear Trend" if mae_lin < mae_naive else "Naive"}
