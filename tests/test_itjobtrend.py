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


def test_rolling_origin_windows_never_touch_test():
    """Mọi mốc rolling-origin dùng để chọn mô hình phải kết thúc trước tháng test đầu tiên."""
    idx = pd.date_range("2018-01-01", periods=66, freq="MS")
    s = pd.Series(np.arange(66, dtype=float) + 100, index=idx)
    r = fc.evaluate_models(s, h=6, models=["Naive (baseline)", "Trung bình trượt 3T"])
    test_start = r["test"].index.min()
    assert r["n_origins"] == 4 and r["origin_step"] == 3 and len(r["validation"]) == 12
    last_origin = r["validation_detail"]["Mốc"].max()
    assert last_origin + pd.DateOffset(months=r["origin_step"] - 1) < test_start
    assert r["validation_predictions"].index.max() < test_start
    assert set(r["validation_detail"]["Mốc"]) == set(r["validation"].index[::3])
    rb = fc.rolling_backtest(s, h=3, n_origins=6, models=["Naive (baseline)"], exclude_last=6)
    assert not rb.empty                                       # vẫn chạy được trên phần trước test


def test_parsimony_rule_keeps_naive_unless_clearly_better(monkeypatch):
    """Mô hình khác Naive chỉ được chọn khi giảm RMSE validation ≥ NAIVE_MARGIN."""
    idx = pd.date_range("2018-01-01", periods=48, freq="MS")
    s = pd.Series(np.r_[np.zeros(36), np.full(12, 10.0)], index=idx)   # validation = 6 tháng có giá trị 10
    monkeypatch.setitem(fc.MODELS, "Tốt hơn 2%", lambda tr, h: np.asarray(tr)[-1] + np.full(h, 0.2))
    monkeypatch.setitem(fc.MODELS, "Biết trước 10", lambda tr, h: np.full(h, 10.0))
    s.iloc[30:36] = 0
    s.iloc[36:42] = 10                                      # validation: 07/2021–12/2021 (6 tháng, 2 mốc)
    small = fc.evaluate_models(s, h=6, validation_size=6, models=["Naive (baseline)", "Tốt hơn 2%"])
    assert small["best"] == "Naive (baseline)" and "giữ Naive" in small["selection_rule"]
    big = fc.evaluate_models(s, h=6, validation_size=6, models=["Naive (baseline)", "Biết trước 10"])
    assert big["best"] == "Biết trước 10"


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


def test_salary_trend_yearly_halfyear_and_overall_line():
    d = pd.DataFrame({"month": pd.to_datetime(["2018-02-01", "2018-08-01", "2019-03-01", "2019-11-01", "2019-12-01"]),
                      "role": ["A", "B", "A", "A", "B"], "salary_usd": [100_000, 120_000, 130_000, 150_000, 90_000]})
    year = salary_model.salary_trend_table(d, "YS", min_posts=1, roles=["A"], date_col="month", include_overall=True)
    a = year[year["Vị trí"] == "A"].set_index("Kỳ")["Lương trung vị (USD/năm)"]
    assert a.to_dict() == {pd.Timestamp("2018-01-01"): 100_000, pd.Timestamp("2019-01-01"): 140_000}
    overall = year[year["Vị trí"] == salary_model.OVERALL_LABEL].set_index("Kỳ")
    assert overall.loc[pd.Timestamp("2019-01-01"), "Số tin"] == 3          # đường tổng không bị lọc theo roles
    half = salary_model.salary_trend_table(d, "6MS", min_posts=1, date_col="month")
    assert set(half["Kỳ"].dt.month) == {1, 7}
    chg = salary_model.salary_trend_change(year)
    assert chg.set_index("Vị trí").loc["A", "Thay đổi (%)"] == pytest.approx(40.0)
    with pytest.raises(ValueError):
        salary_model.salary_trend_table(d, "W")


def test_salary_trend_hacker_news_covers_several_years(data):
    """Yêu cầu 'xu hướng mức lương': dữ liệu HN phải cho ≥ 5 năm có đủ mẫu."""
    _, hn, _ = data
    t = salary_model.salary_trend_table(hn, "YS", min_posts=30, roles=[], date_col="month", include_overall=True)
    assert t["Kỳ"].nunique() >= 5
    assert (t["Số tin"] >= 30).all()


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


def test_chatbot_salary_trend_tool(data):
    """Trợ lý AI có hàm trả lời câu hỏi 'xu hướng lương' bằng số liệu tính từ Hacker News."""
    from src import chatbot
    li, hn, _ = data
    chatbot.set_data(li, hn)
    r = chatbot.xu_huong_luong()
    years = [x["ky"] for x in r["ket_qua"]]
    assert years == sorted(years) and len(set(years)) >= 5
    assert all(x["so_tin"] >= 10 for x in r["ket_qua"])
    assert r["thay_doi"] and isinstance(r["thay_doi"][0]["ky_dau"], str)
    assert "loi" in chatbot.xu_huong_luong("khong-ton-tai")
    assert chatbot.xu_huong_luong in chatbot.DATA_TOOLS


def test_ui_has_no_emoji_icons():
    """Yêu cầu giao diện: dùng icon Material/SVG, không dùng emoji thay icon."""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    emoji = re.compile("[\U0001F300-\U0001FAFF☀-➿⭐]")
    bad = [f"{p.relative_to(root)}:{i}" for folder in ("views", "src") for p in (root / folder).rglob("*.py")
           for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if emoji.search(line)]
    assert not bad, bad
