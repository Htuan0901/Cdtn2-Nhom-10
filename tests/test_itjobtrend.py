"""Kiểm thử các module gốc của ITJobTrend: pipeline, kỹ năng, dự báo theo tháng (kể cả ARIMA/SARIMA), mô hình lương, công cụ chatbot."""
import numpy as np
import pandas as pd
import pytest

from src import chatbot, forecast as fc, salary_model
from src.pipeline import load_processed, parse_salary
from src.skills import classify_role, extract_skills


@pytest.fixture(scope="module")
def data():
    res = load_processed()
    if res is None:
        pytest.skip("Chưa có data/processed")
    li, hn, log = res
    return li, hn, log


def test_parse_salary_hn():
    assert parse_salary("Salary $120k - $160k + equity") == 140_000
    assert parse_salary("$120,000 - $160,000") == 140_000
    assert parse_salary("no salary here") is None


def test_extract_skills_and_role():
    sk = extract_skills("We use Python, TypeScript and React on AWS with Kubernetes")
    assert {"Python", "TypeScript", "React", "AWS", "Kubernetes"} <= set(sk)
    assert classify_role("Senior Data Engineer") != classify_role("Frontend Developer")


def test_processed_data_loaded(data):
    li, hn, log = data
    assert len(li) > 10_000 and len(hn) > 40_000
    assert hn["month"].nunique() >= 60
    assert not log.empty


def test_monthly_backtest_baseline_and_time_split(data):
    _, hn, _ = data
    s = fc.monthly_series(hn)
    res, preds = fc.backtest(s, 6, ["Naive (baseline)", "Trung bình trượt 3T", "Hồi quy xu hướng + mùa vụ"])
    assert "Naive (baseline)" in set(res["Mô hình"])
    assert len(preds) == 6 and preds.index.min() > s.index[-7]      # test là 6 tháng cuối
    assert {"MAE", "RMSE", "MAPE (%)"} <= set(res.columns)


def test_monthly_arima_sarima(data):
    pytest.importorskip("statsmodels")
    _, hn, _ = data
    s = fc.monthly_series(hn)
    res, _ = fc.backtest(s, 6, ["Naive (baseline)", "ARIMA (chọn theo AIC)", "SARIMA (mùa vụ 12)"])
    assert {"ARIMA (chọn theo AIC)", "SARIMA (mùa vụ 12)"} <= set(res["Mô hình"])


def test_arima_rejected_on_short_series():
    s = pd.Series(np.arange(10.0), index=pd.date_range("2020-01-01", periods=10, freq="MS"))
    with pytest.raises(ValueError):
        fc._arima(s, 3)
    with pytest.raises(ValueError):
        fc._sarima(s, 3)


def test_salary_model_beats_baseline(data):
    li, _, _ = data
    b = salary_model.train_salary_models(li)
    res = b["results"].set_index("Mô hình")
    assert res.loc[b["best"], "MAE (USD)"] < res.loc["Baseline (trung vị)", "MAE (USD)"]


def test_chatbot_tools_compute_real_numbers(data):
    li, hn, _ = data
    chatbot.set_data(li, hn)
    info = chatbot.thong_tin_du_lieu()
    assert info["linkedin"]["so_dong"] == len(li)
    top = chatbot.top_ky_nang("linkedin", 5)
    assert len(top["ket_qua"]) == 5 and top["so_tin"] == len(li)
    trend = chatbot.xu_huong_ky_nang("Python", "Y")
    assert trend["ky_nang"] == "Python" and len(trend["ket_qua"]) >= 5


def test_model_selection_ignores_test_values(monkeypatch):
    """Nếu chỉ đổi y_test thì mô hình đã được chọn bằng validation KHÔNG đổi."""
    idx = pd.date_range("2020-01-01", periods=36, freq="MS")
    original = pd.Series(np.zeros(36), index=idx)
    original.iloc[-6:] = 10   # test khác validation
    changed = original.copy()
    changed.iloc[-6:] = 10000

    def ten_model(_train, h):
        return np.full(h, 10.)

    monkeypatch.setitem(fc.MODELS, "Thử nghiệm: dự báo 10", ten_model)
    models = ["Naive (baseline)", "Thử nghiệm: dự báo 10"]
    r1 = fc.evaluate_models(original, h=6, validation_size=6, models=models)
    r2 = fc.evaluate_models(changed, h=6, validation_size=6, models=models)
    assert r1["best"] == r2["best"] == "Naive (baseline)"
    assert r1["train"].index.max() < r1["validation"].index.min()
    assert r1["validation"].index.max() < r1["test"].index.min()
    assert r1["validation_table"].equals(r2["validation_table"])
    assert not r1["test_table"].equals(r2["test_table"])


def test_generic_forecaster_selection_ignores_test_values(monkeypatch):
    """Tập test của module dự báo tổng quát chỉ dùng đánh giá."""
    from src.analytics.forecasting import forecaster as generic
    idx = pd.date_range("2022-01-01", periods=36, freq="MS")
    s1 = pd.Series(np.zeros(36), index=idx, name="test")
    s2 = s1.copy()
    s2.iloc[-6:] = 2000

    def constant_ten(_train, h, **_kw):
        p = np.repeat(10., h)
        return p, p, p, {}

    monkeypatch.setitem(generic.FORECASTERS, "moving_average", constant_ten)
    def run(s):
        info = generic.SeriesInfo(s, "MS", len(s), 0, "không cần")
        return generic.run_forecast(info, ["moving_average"], test_size=6, validation_size=6)

    r1, r2 = run(s1), run(s2)
    assert r1["ok"] and r2["ok"]
    assert r1["best"] == r2["best"] == "naive"
    assert r1["validation_metrics"] == r2["validation_metrics"]
    assert r1["train"].index.max() < r1["validation"].index.min() < r1["test"].index.min()
    assert r1["results"]["naive"]["metrics"]["RMSE"] != r2["results"]["naive"]["metrics"]["RMSE"]


def test_salary_trend_monthly_group_median_and_minimum_count():
    d = pd.DataFrame({"listed_date": ["2024-03-01", "2024-03-15", "2024-03-18", "2024-04-01"],
                      "role": ["Data Analyst"] * 4, "salary_usd": [90_000, 110_000, 200_000, 150_000]})
    trend = salary_model.salary_trend_table(d, "MS", min_posts=2)
    assert len(trend) == 1
    assert trend.iloc[0]["Lương trung vị (USD/năm)"] == 110_000
    assert trend.iloc[0]["Số tin"] == 3
    assert salary_model.salary_trend_table(d, "MS", min_posts=4).empty
    assert len(salary_model.salary_trend_table(d, "QS", min_posts=1)) == 2


def test_salary_selected_by_validation_not_test(data):
    li, _, _ = data
    result = salary_model.train_salary_models(li)
    scores = result["validation_scores"]
    candidates = [k for k in scores if not k.startswith("Baseline")]
    assert result["best"] == min(candidates, key=lambda k: scores[k]["MAE"])
    assert result["n_train"] + result["n_validation"] + result["n_test"] == int((li["salary_usd"].notna() & (li["salary_usd"] > 0)).sum())


def test_generic_holt_included_and_runs_on_sufficient_history():
    """Holt xuất hiện trong module dự báo tổng quát (không chỉ trang gốc)."""
    pytest.importorskip("statsmodels")
    from src.analytics.forecasting import forecaster as generic
    idx = pd.date_range("2018-01-01", periods=48, freq="MS")
    s = pd.Series(np.arange(48, dtype=float) * 0.7 + 100, index=idx)
    info = generic.SeriesInfo(s, "MS", len(s), 0, "không cần")
    r = generic.run_forecast(info, ["holt"], test_size=6, validation_size=6)
    assert r["ok"], r.get("reason")
    assert "holt" in r["results"]
    assert np.isfinite(r["table"].loc[r["table"]["Mô hình"] == "Holt (damped)", "RMSE"]).all()


def test_generic_preparation_fills_gaps_using_only_past_data():
    """Điền kỳ thiếu bằng quá khứ, không interpolate nhìn xuyên sang test."""
    from src.analytics.forecasting import forecaster as generic
    df1 = pd.DataFrame({"date": ["2024-01-01", "2024-03-01", "2024-04-01"], "v": [10, 100, 100]})
    df2 = df1.copy()
    df2.loc[1:, "v"] = [1000, 1000]
    s1 = generic.prepare_series(df1, "date", "v", freq="MS").series
    s2 = generic.prepare_series(df2, "date", "v", freq="MS").series
    assert s1.loc["2024-02-01"] == s2.loc["2024-02-01"] == 10


def test_salary_model_selection_not_affected_by_modified_test_labels():
    """Giữ nguyên train/validation; sửa lương test không được đổi model đã chọn."""
    from sklearn.model_selection import train_test_split
    rng = np.random.default_rng(51)
    n = 90
    d = pd.DataFrame({
        "salary_usd": np.clip(rng.normal(130_000, 30_000, n), 20_000, 250_000),
        "role": rng.choice(["Software Engineer", "Data Analyst / BI"], n),
        "experience": rng.choice(["Mid-Senior", "Associate"], n),
        "state": rng.choice(["CA", "TX"], n),
        "work_type": ["Full-time"] * n,
        "remote": rng.choice([True, False], n),
        "skills": rng.choice(["Python;SQL", "React;JavaScript"], n),
    })
    dev, te = train_test_split(np.arange(n), test_size=0.2, random_state=42)
    r1 = salary_model.train_salary_models(d, seed=42)
    d2 = d.copy()
    d2.loc[te, "salary_usd"] = 490_000  # nhãn test thay đổi cực mạnh
    r2 = salary_model.train_salary_models(d2, seed=42)
    assert r1["best"] == r2["best"]
    for name in r1["validation_scores"]:
        for metric in ("MAE", "R²"):
            assert r1["validation_scores"][name][metric] == pytest.approx(r2["validation_scores"][name][metric], abs=1e-6)
    assert not np.array_equal(r1["y_test"], r2["y_test"])
