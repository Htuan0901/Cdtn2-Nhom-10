"""Kiểm thử AI Assistant: router, công cụ dữ liệu, chế độ không có API Key, Gemini giả lập (mock)."""
import pandas as pd
import pytest

from src.analytics.ai import router
from src.analytics.ai.agent import DataAgent
from src.analytics.ai.gemini_client import GeminiClient, GeminiError, GeminiReply, _map_error
from src.analytics.ai.tools import DataTools
from src.analytics.domain import jobs


def make_tools(df, results=None, time_col=None):
    st = {"df": df, "meta": {"name": "test.csv"}, "results": results or {}, "time_col": time_col, "ws_clean_log": []}
    return DataTools(lambda: st)


class FakeClient:
    """Giả lập Gemini: ghi lại prompt để kiểm tra agent đã gửi số liệu thật."""

    def __init__(self):
        self.configured = True
        self.last_contents = None
        self.model = "fake"

    def generate(self, contents, system, functions=None, **kw):
        self.last_contents = contents
        self.last_system = system
        return GeminiReply(text="Câu trả lời giả lập")

    def generate_with_search(self, contents, system, **kw):
        return GeminiReply(text="Tin mới", sources=[{"title": "Nguồn A", "uri": "https://example.com"}])


# ------------------------------------------------------------------ Router
@pytest.mark.parametrize("q,kind", [
    ("Dataset có bao nhiêu bản ghi?", "data"),
    ("Có bao nhiêu giá trị bị thiếu?", "data"),
    ("Biến nào có tương quan cao nhất?", "data"),
    ("Machine Learning là gì?", "general"),
    ("ARIMA hoạt động như thế nào?", "general"),
    ("Random Forest khác Decision Tree thế nào?", "general"),
    ("Tin tức mới nhất về thị trường AI hiện nay", "web"),
    ("Hãy giải thích kết quả dự báo", "data"),
    ("Những phương pháp xử lý dữ liệu thiếu phổ biến là gì?", "general"),
    ("Giải thích thuật toán K-Means", "general"),
    ("Hãy giải thích biểu đồ tương quan", "data"),
])
def test_router_kinds(numeric_df, q, kind):
    assert router.route(q, numeric_df).kind == kind


def test_router_column_and_month(monthly_series_df):
    r = router.route("Tháng nào có nồng độ PM25 cao nhất?", monthly_series_df)
    assert r.kind == "data"
    assert ("analyze_time_trend", {"value_column": "pm25", "granularity": "thang"}) in r.calls


def test_router_quantitative_question_with_la_gi(job_df):
    feat, _ = jobs.build_job_features(job_df, jobs.auto_map_columns(job_df))
    r = router.route("Top 10 kỹ năng được yêu cầu nhiều nhất là gì?", feat)
    assert r.kind == "data" and ("get_top_skills", {}) in r.calls


def test_router_pm25_general_vs_data():
    df = pd.DataFrame({"PM2.5": [1.0, 2.0, 3.0] * 10, "station": ["A", "B", "C"] * 10})
    assert router.route("PM2.5 ảnh hưởng thế nào đến môi trường?", df).kind == "general"
    r = router.route("Giá trị trung bình của PM2.5 là bao nhiêu?", df)
    assert r.kind == "data" and ("get_column_statistics", {"column": "PM2.5"}) in r.calls


def test_router_no_false_skill_c(numeric_df):
    r = router.route("Có bao nhiêu dữ liệu bất thường?", numeric_df)
    assert all(c[0] != "get_skill_trend" for c in r.calls)


def test_salary_by_job_group(job_df):
    feat, _ = jobs.build_job_features(job_df, jobs.auto_map_columns(job_df))
    t = make_tools(feat)
    r = router.route("Lương trung vị theo nhóm vị trí như thế nào?", feat)
    name, kw = next(c for c in r.calls if c[0] == "get_salary_statistics")
    res = t.get_salary_statistics(**kw)
    assert res["cot_nhom"] == "job_group"
    res2 = t.get_salary_statistics("kinh nghiem")
    assert res2["cot_nhom"] == "exp_bucket"


def test_router_no_data():
    assert router.route("Dataset có bao nhiêu bản ghi?", None).kind == "no_data"


def test_router_skill_question(job_df):
    feat, _ = jobs.build_job_features(job_df, jobs.auto_map_columns(job_df))
    r = router.route("Xu hướng kỹ năng Python qua các năm?", feat)
    assert ("get_skill_trend", {"skill": "Python"}) in r.calls


# ------------------------------------------------------------------ Tools tính đúng số liệu
def test_tools_exact_numbers(numeric_df):
    t = make_tools(numeric_df)
    s = t.get_dataset_summary()
    assert s["so_ban_ghi"] == len(numeric_df) and s["so_thuoc_tinh"] == numeric_df.shape[1]
    m = t.get_missing_values()
    assert m["tong_o_thieu"] == int(numeric_df.isna().sum().sum())
    c = t.get_column_statistics("X1")
    assert c["mean"] == round(numeric_df["x1"].mean(), 4)
    cor = t.calculate_correlation()
    top = cor["cac_cap_tuong_quan_manh_nhat"][0]
    assert {top["Biến 1"], top["Biến 2"]} == {"x1", "x2"}


def test_tools_missing_info_reported(numeric_df):
    t = make_tools(numeric_df)
    assert t.get_column_statistics("không_tồn_tại")["available"] is False
    assert t.get_model_metrics()["available"] is False      # chưa huấn luyện ⇒ nói rõ, không bịa
    assert t.get_forecast_results()["available"] is False


def test_tools_time_granularity_insufficient(job_df):
    t = make_tools(job_df, time_col="year")
    r = t.analyze_time_trend("", "thang")
    assert r["available"] is False and "Theo năm" in r["reason"]
    ok = t.analyze_time_trend("", "nam")
    assert ok["available"] and ok["ky_cao_nhat"]["gia_tri"] == job_df["year"].value_counts().max()


# ------------------------------------------------------------------ Agent
def test_agent_without_api_key_answers_data_offline(numeric_df):
    agent = DataAgent(make_tools(numeric_df), client=GeminiClient(api_key=""))
    r = agent.answer("Dataset có bao nhiêu bản ghi?")
    assert r.error == "no_key" and not r.used_llm
    assert str(len(numeric_df)) in r.text                    # vẫn trả lời bằng số liệu thật


def test_agent_without_api_key_general_question(numeric_df):
    agent = DataAgent(make_tools(numeric_df), client=GeminiClient(api_key=""))
    r = agent.answer("K-Means là gì?")
    assert r.error == "no_key" and "GEMINI_API_KEY" in r.text


def test_gemini_client_no_key_raises():
    with pytest.raises(GeminiError) as e:
        GeminiClient(api_key="").generate([], "sys")
    assert e.value.kind == "no_key"


def test_agent_data_question_sends_computed_numbers(numeric_df):
    fake = FakeClient()
    agent = DataAgent(make_tools(numeric_df), client=fake)
    r = agent.answer("Giá trị trung bình của x1 là bao nhiêu?")
    assert r.used_llm and r.kind == "data"
    prompt = fake.last_contents[-1]["parts"][0]["text"]
    assert str(round(numeric_df["x1"].mean(), 4)) in prompt     # Gemini nhận số do Python tính
    assert "không tự tính" in fake.last_system.lower() or "tuyệt đối không" in fake.last_system.lower()


def test_agent_web_question_returns_sources(numeric_df):
    agent = DataAgent(make_tools(numeric_df), client=FakeClient())
    r = agent.answer("Tin tức mới nhất về AI hiện nay?")
    assert r.kind == "web" and "https://example.com" in r.text


def test_error_mapping():
    class E(Exception):
        code = 429
    assert _map_error(E("RESOURCE_EXHAUSTED")).kind == "rate_limit"

    class A(Exception):
        code = 400
    assert _map_error(A("API key not valid")).kind == "auth"


def test_context_size_limited(numeric_df):
    from src.analytics.ai.agent import _clip
    big = {"x": "a" * 50_000}
    assert len(_clip(big, 1000)) < 1100


def test_logger_redacts_key(monkeypatch):
    from src.analytics.logger import redact
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyTESTKEY1234567890abcdefghijk")
    assert "AIza" not in redact("key=AIzaSyTESTKEY1234567890abcdefghijk")
