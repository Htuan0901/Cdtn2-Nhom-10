"""Fixtures dùng chung. Lưu trữ test được tách sang thư mục tạm để không ghi đè dữ liệu thật."""
import os
import sys
import tempfile
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="jobmarket_test_")
os.environ["APP_STORAGE_DIR"] = _TMP
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def job_df():
    rng = np.random.default_rng(0)
    n = 400
    titles = ["Python Developer", "Java Backend Engineer", "Data Analyst", "Tester", "Nhân viên kinh doanh",
              "Frontend Developer ReactJS", "DevOps Engineer", "Kế toán tổng hợp"]
    sal = ["10 - 15 triệu", "15.000.000 - 25.000.000 VND", "1000 - 1500 USD", "Thỏa thuận", "20 triệu", "Trên 30 triệu"]
    exp = ["1 năm", "2 năm", "Không yêu cầu", "3 - 5 năm", "Dưới 1 năm"]
    loc = ["Hồ Chí Minh", "Hà Nội", "Quận Cầu Giấy, Hà Nội", "Đà Nẵng", "Tân Bình, TP.HCM"]
    desc = ["Yêu cầu Python, SQL, Docker và tiếng Anh", "Java Spring Boot, MySQL, Git", "Power BI, SQL, Excel",
            "Manual test, test case, Jira", "Bán hàng, chăm sóc khách hàng", "ReactJS, TypeScript, HTML/CSS",
            "AWS, Kubernetes, CI/CD, Linux", "Kế toán thuế, Excel"]
    idx = rng.integers(0, len(titles), n)
    df = pd.DataFrame({
        "id": np.arange(1, n + 1),
        "job_title": [titles[i] for i in idx],
        "salary": rng.choice(sal, n),
        "location": rng.choice(loc, n),
        "experience_level": rng.choice(exp, n),
        "job_industry": ["IT phần mềm" if i not in (4, 7) else "Kinh doanh" for i in idx],
        "job_description": [desc[i] for i in idx],
        "year": rng.choice([2022, 2023, 2024, 2025], n),
    })
    return df


@pytest.fixture
def numeric_df():
    rng = np.random.default_rng(1)
    n = 300
    x1 = rng.normal(50, 10, n)
    x2 = 2 * x1 + rng.normal(0, 5, n)
    x3 = rng.uniform(0, 1, n)
    cat = rng.choice(["A", "B", "C"], n)
    y = 3 * x1 - 2 * x3 * 10 + (cat == "B") * 15 + rng.normal(0, 5, n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "group": cat, "target": y,
                       "date": pd.date_range("2020-01-01", periods=n, freq="D")})
    df.loc[::25, "x3"] = np.nan
    return df


@pytest.fixture
def monthly_series_df():
    idx = pd.date_range("2018-01-01", periods=60, freq="MS")
    t = np.arange(60)
    y = 100 + 2 * t + 10 * np.sin(2 * np.pi * t / 12) + np.random.default_rng(2).normal(0, 2, 60)
    return pd.DataFrame({"month": idx.strftime("%Y-%m-%d"), "pm25": y})
