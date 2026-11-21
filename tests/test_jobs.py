"""Kiểm thử miền tuyển dụng: chuẩn hóa lương, kinh nghiệm, tỉnh thành, lọc CNTT, trích xuất kỹ năng."""
import math

import pandas as pd
import pytest

from src.analytics.config import USD_TO_VND
from src.analytics.domain import jobs, skills


@pytest.mark.parametrize("text,lo,hi,status", [
    ("10 - 15 triệu", 10, 15, "ok"),
    ("10.000.000 - 15.000.000 VND", 10, 15, "ok"),
    ("10000000 - 15000000 VND MONTH", 10, 15, "ok"),
    ("26-36 triệu", 26, 36, "ok"),
    ("Thỏa thuận", math.nan, math.nan, "negotiable"),
    ("Đang cập nhật", math.nan, math.nan, "negotiable"),
    ("0 VND", math.nan, math.nan, "invalid"),
    ("Trên 30 triệu", 30, math.nan, "ok"),
    ("Tới 20 triệu", math.nan, 20, "ok"),
    ("9.5 triệu", 9.5, 9.5, "ok"),
])
def test_parse_salary(text, lo, hi, status):
    s = jobs.parse_salary(text)
    assert s.status == status
    for got, exp in ((s.min, lo), (s.max, hi)):
        assert (math.isnan(got) and math.isnan(exp)) or abs(got - exp) < 1e-6


def test_parse_salary_usd_and_weekly():
    s = jobs.parse_salary("1000 - 2000 USD")
    assert abs(s.mid - 1500 * USD_TO_VND / 1e6) < 1e-6
    w = jobs.parse_salary("2 - 3 triệu/tuần")
    assert 8 < w.mid < 12


def test_parse_salary_bad_upper_bound():
    s = jobs.parse_salary("15.000.000 - 9 VND")
    assert s.status == "ok" and s.min == 15 and math.isnan(s.max)


@pytest.mark.parametrize("text,years", [("3 năm", 3), ("Không yêu cầu", 0), ("Dưới 1 năm", 0.5),
                                        ("2 - 5 năm kinh nghiệm", 2), ("6 tháng", 0.5), ("Không", 0)])
def test_parse_experience(text, years):
    assert jobs.parse_experience(text) == years


@pytest.mark.parametrize("text,prov", [
    ("Hồ Chí Minh", "Hồ Chí Minh"), ("Đức Huệ, Long An", "Long An"), ("Quận Cầu Giấy, Hà Nội", "Hà Nội"),
    ("198 Hoàng Văn Thụ Phường 9 Phú Nhuận", "Hồ Chí Minh"), ("Tân Bình, TP.HCM", "Hồ Chí Minh"),
    ("Tất cả các huyện | Tầng 8, 24C Phan Đăng Lưu, TP Hồ Chí Minh, Việt Nam", "Hồ Chí Minh"),
    ("Bình Dương", "Bình Dương"), ("Bình Định", "Bình Định"), ("Remote", "Remote"), (None, "Không rõ"),
])
def test_normalize_province(text, prov):
    assert jobs.normalize_province(text) == prov


def test_region():
    assert jobs.region_of("Hà Nội") == "Miền Bắc"
    assert jobs.region_of("Đà Nẵng") == "Miền Trung"
    assert jobs.region_of("Cần Thơ") == "Miền Nam"


@pytest.mark.parametrize("title,industry,expected", [
    ("Java Software Engineer", "QA-QC", True), ("Nhân Viên IT", "Chưa xác định", True),
    ("Nhân Viên Kinh Doanh Phần Mềm", "IT phần mềm", False), ("Nhân Viên Lập Trình Máy CNC", "Cơ khí", False),
    ("Kế toán tổng hợp", "Kế toán", False), ("Data Analyst", "Chưa xác định", True),
    ("Nhân Viên Kỹ Thuật (Bà Rịa - Vũng Tàu)", "Vận hành", False),
])
def test_is_it_posting(title, industry, expected):
    assert jobs.is_it_posting(title, industry) is expected


def test_job_group():
    assert jobs.job_group("Senior Backend Developer (Golang)") == "Backend"
    assert jobs.job_group("Data Engineer") == "Data / AI"
    assert jobs.job_group("Automation Tester") == "QA / Tester"


def test_auto_map_and_build_features(job_df):
    m = jobs.auto_map_columns(job_df)
    assert m["title"] == "job_title" and m["salary"] == "salary" and m["time"] == "year"
    assert jobs.is_job_dataset(job_df)
    out, rep = jobs.build_job_features(job_df, m, it_only=True)
    assert out["is_it"].all() and rep.n_it == len(out) < len(job_df)
    assert {"salary_mid", "exp_years", "province", "region", "job_group"} <= set(out.columns)
    assert out["salary_mid"].dropna().between(0.5, 500).all()


def test_skill_extraction_boundaries():
    s = skills.extract_skills("Thành thạo JavaScript, ReactJS, MySQL; biết Java Spring Boot và C/C++. Ưu tiên AI, tiếng Anh tốt.")
    assert {"JavaScript", "React", "MySQL", "Java", "Spring", "C++", "C", "AI", "Tiếng Anh"} <= s
    assert "SQL" not in s                     # "MySQL" không bị đếm thành "SQL"
    assert "AI" not in skills.extract_skills("bất kỳ ai cũng có thể ứng tuyển")   # từ tiếng Việt "ai"


def test_skill_share_and_forecast():
    sets = pd.Series([{"Python"}, {"Python", "SQL"}, {"SQL"}, {"Python"}, set(), {"Python"}] * 10)
    periods = pd.Series([2021, 2022, 2023, 2024, 2025, 2025] * 10)
    share = skills.skill_share_by_period(sets, periods, ["Python", "SQL"])
    assert share.loc[2021, "Python"] == 100.0
    fc = skills.forecast_skill_shares(share, ["Python", "SQL"])
    assert fc["ok"] and {"mae_linear", "mae_naive"} <= set(fc)
    assert fc["table"]["Dự báo kỳ tới (%)"].between(0, 100).all()
