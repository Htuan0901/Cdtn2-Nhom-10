"""Từ điển kỹ năng/công nghệ CNTT và các hàm trích xuất (text mining dựa trên từ điển + regex).

Mỗi kỹ năng có: tên chuẩn hóa, nhóm, và danh sách biến thể (regex) xuất hiện trong mô tả công việc.
Ví dụ "k8s", "kubernetes" đều được chuẩn hóa thành "Kubernetes"; "golang", "go (lang)" thành "Go".
"""
from __future__ import annotations

import re

import pandas as pd

# (Tên chuẩn, Nhóm, [mẫu regex]) – mẫu viết thường, \b được thêm tự động khi cần
SKILLS: list[tuple[str, str, list[str]]] = [
    # ---- Ngôn ngữ lập trình ----
    ("Python", "Ngôn ngữ lập trình", [r"python\d?"]),
    ("Java", "Ngôn ngữ lập trình", [r"java(?!\s*script)(?![a-z])"]),
    ("JavaScript", "Ngôn ngữ lập trình", [r"javascript", r"java script", r"ecmascript", r"\bjs\b", r"es6"]),
    ("TypeScript", "Ngôn ngữ lập trình", [r"typescript", r"\bts\b(?=\s*[,/)])"]),
    ("C#", "Ngôn ngữ lập trình", [r"c#", r"c sharp", r"csharp"]),
    ("C++", "Ngôn ngữ lập trình", [r"c\+\+", r"cpp"]),
    ("C", "Ngôn ngữ lập trình", [r"(?<![\w#+])c(?=\s*(?:,|/|\)|and\b|or\b))(?<!objective-c)"]),
    ("Go", "Ngôn ngữ lập trình", [r"golang", r"(?<![\w-])go(?=\s*(?:,|/|\)|\(golang\)|lang))"]),
    ("Rust", "Ngôn ngữ lập trình", [r"rust(?:lang)?"]),
    ("Ruby", "Ngôn ngữ lập trình", [r"ruby(?! on rails)"]),
    ("PHP", "Ngôn ngữ lập trình", [r"php"]),
    ("Kotlin", "Ngôn ngữ lập trình", [r"kotlin"]),
    ("Swift", "Ngôn ngữ lập trình", [r"swift(?:ui)?"]),
    ("Scala", "Ngôn ngữ lập trình", [r"scala"]),
    ("R", "Ngôn ngữ lập trình", [r"(?<![\w&'’-])r(?=\s*(?:,|/|\)|\.\s|\.$|$|programming|studio|language))", r"rstudio"]),
    ("Elixir", "Ngôn ngữ lập trình", [r"elixir"]),
    ("Clojure", "Ngôn ngữ lập trình", [r"clojure"]),
    ("Haskell", "Ngôn ngữ lập trình", [r"haskell"]),
    ("Dart", "Ngôn ngữ lập trình", [r"dart"]),
    ("SQL", "Cơ sở dữ liệu", [r"sql(?!\s*server)", r"t-sql", r"pl/sql"]),
    ("Bash/Shell", "Ngôn ngữ lập trình", [r"bash", r"shell script(?:ing)?", r"powershell"]),
    # ---- Frontend / Web ----
    ("React", "Frontend & Web", [r"react(?:\.?js)?(?! native)", r"next\.?js"]),
    ("Angular", "Frontend & Web", [r"angular(?:js)?"]),
    ("Vue", "Frontend & Web", [r"vue(?:\.?js)?", r"nuxt"]),
    ("HTML/CSS", "Frontend & Web", [r"html5?", r"css3?", r"sass", r"tailwind"]),
    ("Node.js", "Backend & Framework", [r"node(?:\.?js)?", r"express(?:\.?js)?", r"nestjs"]),
    ("Django", "Backend & Framework", [r"django"]),
    ("Flask/FastAPI", "Backend & Framework", [r"flask", r"fastapi"]),
    ("Spring", "Backend & Framework", [r"spring(?:\s*boot)?"]),
    (".NET", "Backend & Framework", [r"\.net", r"asp\.net", r"dotnet"]),
    ("Ruby on Rails", "Backend & Framework", [r"ruby on rails", r"rails", r"\bror\b"]),
    ("GraphQL", "Backend & Framework", [r"graphql"]),
    ("REST API", "Backend & Framework", [r"rest(?:ful)?\s*api", r"restful"]),
    ("Microservices", "Backend & Framework", [r"micro-?services?"]),
    # ---- Mobile ----
    ("React Native", "Mobile", [r"react native"]),
    ("Flutter", "Mobile", [r"flutter"]),
    ("iOS", "Mobile", [r"ios"]),
    ("Android", "Mobile", [r"android"]),
    # ---- Cloud / DevOps ----
    ("AWS", "Cloud & DevOps", [r"aws", r"amazon web services", r"\bec2\b", r"\bs3\b"] ),
    ("Azure", "Cloud & DevOps", [r"azure"]),
    ("GCP", "Cloud & DevOps", [r"gcp", r"google cloud"]),
    ("Docker", "Cloud & DevOps", [r"docker", r"containeri[sz]ation"]),
    ("Kubernetes", "Cloud & DevOps", [r"kubernetes", r"k8s", r"\beks\b", r"\bgke\b", r"\baks\b"]),
    ("Terraform", "Cloud & DevOps", [r"terraform", r"infrastructure as code", r"\biac\b"]),
    ("CI/CD", "Cloud & DevOps", [r"ci/cd", r"ci\s*/\s*cd", r"continuous integration", r"jenkins", r"github actions", r"gitlab ci"]),
    ("Linux", "Cloud & DevOps", [r"linux", r"unix"]),
    ("Git", "Cloud & DevOps", [r"git(?!hub actions)(?:hub|lab)?"]),
    # ---- Dữ liệu & AI ----
    ("Machine Learning", "Dữ liệu & AI", [r"machine learning", r"\bml\b", r"scikit-?learn", r"sklearn"]),
    ("Deep Learning", "Dữ liệu & AI", [r"deep learning", r"neural network", r"pytorch", r"tensorflow", r"keras"]),
    ("LLM/GenAI", "Dữ liệu & AI", [r"\bllms?\b", r"large language model", r"generative ai", r"gen ?ai", r"gpt", r"openai", r"langchain", r"\brag\b", r"prompt engineering"]),
    ("NLP", "Dữ liệu & AI", [r"\bnlp\b", r"natural language processing"]),
    ("Computer Vision", "Dữ liệu & AI", [r"computer vision", r"opencv", r"image recognition"]),
    ("Pandas/NumPy", "Dữ liệu & AI", [r"pandas", r"numpy"]),
    ("Spark", "Dữ liệu & AI", [r"spark", r"pyspark", r"databricks"]),
    ("Hadoop", "Dữ liệu & AI", [r"hadoop", r"hive", r"hdfs"]),
    ("Kafka", "Dữ liệu & AI", [r"kafka"]),
    ("Airflow", "Dữ liệu & AI", [r"airflow"]),
    ("Snowflake", "Dữ liệu & AI", [r"snowflake"]),
    ("Tableau", "Dữ liệu & AI", [r"tableau"]),
    ("Power BI", "Dữ liệu & AI", [r"power\s?bi"]),
    ("Excel", "Dữ liệu & AI", [r"excel"]),
    ("Statistics", "Dữ liệu & AI", [r"statistic(?:s|al)"]),
    ("ETL", "Dữ liệu & AI", [r"\betl\b", r"\belt\b", r"data pipelines?"]),
    # ---- Cơ sở dữ liệu ----
    ("PostgreSQL", "Cơ sở dữ liệu", [r"postgres(?:ql)?"]),
    ("MySQL", "Cơ sở dữ liệu", [r"mysql", r"mariadb"]),
    ("SQL Server", "Cơ sở dữ liệu", [r"sql server", r"mssql"]),
    ("Oracle", "Cơ sở dữ liệu", [r"oracle"]),
    ("MongoDB", "Cơ sở dữ liệu", [r"mongo(?:db)?"]),
    ("Redis", "Cơ sở dữ liệu", [r"redis"]),
    ("Elasticsearch", "Cơ sở dữ liệu", [r"elastic\s?search", r"\belk\b"]),
    ("NoSQL", "Cơ sở dữ liệu", [r"nosql", r"dynamodb", r"cassandra"]),
    # ---- Bảo mật / Hạ tầng / Khác ----
    ("Cybersecurity", "Bảo mật & Hạ tầng", [r"cyber\s?security", r"information security", r"infosec", r"siem", r"penetration test", r"soc\b"]),
    ("Networking", "Bảo mật & Hạ tầng", [r"networking", r"tcp/ip", r"cisco", r"firewall", r"\bvpn\b", r"\blan\b", r"\bwan\b"]),
    ("Active Directory", "Bảo mật & Hạ tầng", [r"active directory", r"\bldap\b", r"office 365", r"microsoft 365"]),
    ("Salesforce", "Nền tảng doanh nghiệp", [r"salesforce", r"apex"]),
    ("SAP", "Nền tảng doanh nghiệp", [r"\bsap\b"]),
    ("ServiceNow", "Nền tảng doanh nghiệp", [r"servicenow"]),
    ("Agile/Scrum", "Quy trình", [r"agile", r"scrum", r"kanban", r"jira"]),
    ("Testing/QA", "Quy trình", [r"unit test(?:s|ing)?", r"test automation", r"selenium", r"cypress", r"jest", r"pytest", r"\btdd\b"]),
]

SKILL_GROUP = {name: grp for name, grp, _ in SKILLS}
# Kỹ năng "chung" (không đặc thù CNTT) – không dùng làm bằng chứng để xác định tin CNTT
GENERIC_SKILLS = {"Excel", "Agile/Scrum", "Statistics", "Networking", "SAP", "Salesforce", "Oracle"}
SKILL_NAMES = [s[0] for s in SKILLS]


def _compile(pats: list[str]) -> re.Pattern:
    parts = []
    for p in pats:
        # thêm ranh giới từ nếu mẫu chưa tự xử lý
        left = "" if p.startswith(("\\b", "(?<")) else r"(?<![a-z0-9])"
        right = "" if p.endswith("\\b") else r"(?![a-z0-9])"
        parts.append(f"{left}(?:{p}){right}")
    return re.compile("|".join(parts), flags=re.IGNORECASE)


_PATTERNS = {name: _compile(pats) for name, _, pats in SKILLS}


def _literal(p: str) -> str | None:
    """Phần chữ cố định ở đầu mẫu regex (vd 'python\\d?' → 'python'), dùng để lọc nhanh bằng phép 'in'."""
    m = re.match(r"(?:\\b|\(\?<[!=][^)]*\))*([a-z0-9 #/-]+)", p)
    return m.group(1) if m else None


# Mỗi kỹ năng: các chuỗi con bắt buộc. Nếu văn bản không chứa chuỗi nào thì bỏ qua regex (nhanh hơn ~20 lần).
_KEYS = {}
for _name, _, _pats in SKILLS:
    lits = [_literal(p) for p in _pats]
    _KEYS[_name] = None if any(l is None for l in lits) else tuple(set(lits))


def extract_skills(text: str) -> list[str]:
    """Trả về danh sách kỹ năng (đã chuẩn hóa, không trùng) xuất hiện trong văn bản."""
    if not isinstance(text, str) or not text:
        return []
    t = text.lower()
    out = []
    for name, pat in _PATTERNS.items():
        keys = _KEYS[name]
        if keys is not None and not any(k in t for k in keys):
            continue
        if pat.search(t):
            out.append(name)
    return out


def highlight_skills(text: str) -> str:
    """Bọc các cụm từ khớp kỹ năng bằng **…** để hiển thị (dùng ở trang Trích xuất kỹ năng)."""
    spans = []
    for name, pat in _PATTERNS.items():
        for m in pat.finditer(text):
            spans.append((m.start(), m.end(), name))
    spans.sort()
    out, last = [], 0
    for s, e, name in spans:
        if s < last:
            continue
        out.append(text[last:s])
        out.append(f"**:blue[{text[s:e]}]**")
        last = e
    out.append(text[last:])
    return "".join(out)


def skills_matrix(skill_series: pd.Series) -> pd.DataFrame:
    """Chuỗi 'A;B;C' → ma trận 0/1 (mỗi kỹ năng một cột). Đây là bảng đặc trưng (feature table)."""
    m = skill_series.fillna("").str.get_dummies(sep=";")
    return m.reindex(columns=[c for c in SKILL_NAMES if c in m.columns])


# ---------------- Phân loại vị trí công việc từ tiêu đề ----------------
ROLE_RULES: list[tuple[str, str]] = [
    ("Data Scientist / AI-ML", r"data scien|machine learning|\bml\b|\bai\b|artificial intelligence|deep learning|nlp|computer vision|llm"),
    ("Data Engineer", r"data engineer|big data|etl|data architect|data platform|analytics engineer"),
    ("Data Analyst / BI", r"data analy|business intelligence|\bbi\b|analytics|reporting analyst|tableau|power bi"),
    ("DevOps / Cloud / SRE", r"devops|dev ops|cloud|site reliability|\bsre\b|platform engineer|infrastructure engineer|kubernetes|aws|azure"),
    ("Cybersecurity", r"secur|cyber|soc analyst|penetration|infosec|identity & access|\biam\b"),
    ("QA / Test", r"\bqa\b|quality assurance|test(?:er|ing)? (?:engineer|analyst|automation)|\bsdet\b|automation test"),
    ("Mobile", r"\bios\b|android|mobile|flutter|react native"),
    ("Frontend / Web", r"front[- ]?end|frontend|web developer|ui developer|ui engineer|react developer|javascript developer"),
    ("Backend / Full-stack", r"back[- ]?end|backend|full[- ]?stack|fullstack|\.net|java developer|python developer|php|node"),
    ("IT Project / Product", r"project manager|product manager|product owner|scrum master|program manager|it manager|delivery manager"),
    ("Software Engineer", r"software|developer|programmer|\bswe\b|application engineer|coder|architect|app(?:lication)? development"),
    ("Network / System Admin", r"network|system(?:s)? admin|sysadmin|systems engineer|infrastructure|linux admin|windows admin"),
    ("IT Support / Helpdesk", r"help ?desk|service desk|desktop support|it support|technical support|support technician|it technician|it specialist"),
    ("Database / DBA", r"database|\bdba\b|sql developer"),
    ("Business / Systems Analyst", r"business analyst|systems analyst|functional analyst|erp|salesforce|sap\b|servicenow"),
]
_ROLE_PATTERNS = [(r, re.compile(p, re.IGNORECASE)) for r, p in ROLE_RULES]

IT_TITLE_PATTERN = re.compile(
    r"software|developer|programmer|data scien|data engineer|data analy|machine learning|\bml\b|\bai\b|devops|cloud|"
    r"\bsre\b|site reliability|cyber|information security|network engineer|network admin|system(?:s)? admin|sysadmin|"
    r"help ?desk|service desk|desktop support|it support|\bit\b|information technology|\bqa\b engineer|test automation|"
    r"\bsdet\b|front[- ]?end|back[- ]?end|full[- ]?stack|web developer|mobile developer|\bios\b|android|database|\bdba\b|"
    r"solutions architect|technical architect|platform engineer|infrastructure engineer|business intelligence|"
    r"\bbi\b developer|\betl\b|salesforce|servicenow|sap\b|\.net|java\b|python|javascript|scrum master|computer",
    re.IGNORECASE,
)


def classify_role(title: str) -> str:
    if not isinstance(title, str):
        return "Khác (IT)"
    for role, pat in _ROLE_PATTERNS:
        if pat.search(title):
            return role
    return "Khác (IT)"


def seniority(title: str, level: str | None = None) -> str:
    """Suy ra cấp bậc từ tiêu đề (dùng cho HN, nơi không có cột kinh nghiệm)."""
    t = (title or "").lower()
    if re.search(r"intern|internship|thực tập", t):
        return "Thực tập"
    if re.search(r"junior|\bjr\b|entry|graduate|associate", t):
        return "Junior"
    if re.search(r"principal|staff|lead|head of|director|vp|chief|cto|architect|manager", t):
        return "Lead/Quản lý"
    if re.search(r"senior|\bsr\b|\biii\b", t):
        return "Senior"
    return "Không rõ"
