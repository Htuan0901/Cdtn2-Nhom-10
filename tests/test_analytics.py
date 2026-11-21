"""Kiểm thử: EDA, khai phá dữ liệu, huấn luyện & đánh giá mô hình, dự báo chuỗi thời gian, phát hiện bất thường."""
import numpy as np
import pandas as pd
import pytest

from src.analytics.analysis import eda
from src.analytics.anomaly import detector
from src.analytics.forecasting import forecaster as fc
from src.analytics.mining import clustering
from src.analytics.ml import trainer


# ------------------------------------------------------------------ EDA
def test_descriptive_and_correlation(numeric_df):
    st = eda.descriptive_stats(numeric_df)
    assert set(["Mean", "Median", "Mode", "Variance", "Std", "Min", "Max"]) <= set(st.columns)
    row = st[st["Biến"] == "x1"].iloc[0]
    assert abs(row["Mean"] - numeric_df["x1"].mean()) < 1e-3
    corr = eda.correlation_matrix(numeric_df)
    assert "group" not in corr.columns and "date" not in corr.columns   # không tính tương quan cột không phù hợp
    top = eda.top_correlations(corr, 1).iloc[0]
    assert {top["Biến 1"], top["Biến 2"]} == {"x1", "x2"} and top["Hệ số"] > 0.9


def test_correlation_excludes_id(job_df):
    assert eda.correlation_matrix(job_df).empty    # chỉ có id & year ⇒ không có biến số hợp lệ


def test_group_and_time(numeric_df):
    g = eda.group_stats(numeric_df, "group", "target")
    assert set(g["group"]) == {"A", "B", "C"}
    ts, freq = eda.time_aggregate(numeric_df, "date", "x1", freq="MS")
    assert freq == "MS" and len(ts) == 10
    assert "xu hướng" in eda.describe_trend_text(ts)
    assert eda.key_findings(numeric_df)


def test_frequency_table(job_df):
    ft = eda.frequency_table(job_df, "job_title", top=3)
    assert ft["Tần số"].sum() == len(job_df)


# ------------------------------------------------------------------ Data mining
def test_kmeans_elbow_silhouette_pca():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.5, (100, 3))
    b = rng.normal(5, 0.5, (100, 3))
    df = pd.DataFrame(np.vstack([a, b]), columns=["f1", "f2", "f3"])
    X, _, _ = clustering.prepare_matrix(df, ["f1", "f2", "f3"])
    tab = clustering.elbow_silhouette(X, 2, 5)
    assert clustering.suggest_k(tab) == 2
    res = clustering.run_kmeans(df, ["f1", "f2", "f3"], 2)
    assert res["silhouette"] > 0.7
    assert abs(sum(res["explained_variance"]) - 1) < 0.2
    assert len(res["descriptions"]) == 2
    with pytest.raises(ValueError):
        clustering.run_kmeans(df, ["f1"], 2)


# ------------------------------------------------------------------ Machine learning
def test_detect_problem_type():
    assert trainer.detect_problem_type(pd.Series(np.random.rand(100)))[0] == "regression"
    assert trainer.detect_problem_type(pd.Series(["a", "b"] * 50))[0] == "classification"
    assert trainer.detect_problem_type(pd.Series([0, 1] * 50))[0] == "classification"


def test_train_regression_and_evaluate(numeric_df):
    feats, excluded = trainer.usable_features(numeric_df, "target")
    assert "date" in excluded
    res = trainer.train_models(numeric_df, "target", ["x1", "x3", "group"],
                               model_names=["Linear Regression", "Random Forest Regressor"])
    comp = res.comparison.set_index("Mô hình")
    assert {"MAE", "MSE", "RMSE", "R2"} <= set(comp.columns)
    assert trainer.BASELINE_NAME in comp.index
    best = comp.loc[res.best_name]
    assert best["RMSE"] < comp.loc[trainer.BASELINE_NAME, "RMSE"]      # tốt hơn baseline
    assert best["R2"] > 0.8
    assert abs(best["RMSE"] ** 2 - best["MSE"]) / best["MSE"] < 1e-3
    path = trainer.save_model(res, "testds")
    loaded = trainer.load_model(path)
    pred, _ = trainer.predict_one(loaded["pipeline"], loaded["features"], {"x1": 50, "x3": 0.5, "group": "B"})
    assert np.isfinite(pred)


def test_train_classification_imbalanced():
    rng = np.random.default_rng(3)
    n = 600
    x = rng.normal(0, 1, (n, 2))
    y = np.where(x[:, 0] + 0.5 * x[:, 1] > 1.3, "high", "low")     # ~10% lớp thiểu số
    df = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "label": y})
    res = trainer.train_models(df, "label", ["a", "b"], model_names=["Logistic Regression", "Decision Tree Classifier"])
    assert res.problem == "classification"
    assert any("Mất cân bằng" in n for n in res.notes)
    comp = res.comparison.set_index("Mô hình")
    assert {"Accuracy", "Precision", "Recall", "F1 (macro)", "ROC-AUC"} <= set(comp.columns)
    assert comp.loc[res.best_name, "F1 (macro)"] > comp.loc[trainer.BASELINE_NAME, "F1 (macro)"]
    cm = res.details[res.best_name]["confusion"]
    assert cm.values.sum() == res.n_test


def test_leakage_check(numeric_df):
    df = numeric_df.copy()
    df["target_copy"] = df["target"] * 1.0001
    sus = trainer.leakage_check(df, ["x1", "x3", "target_copy"], "target")
    assert "target_copy" in sus and "x3" not in sus


def test_train_rejects_target_in_features(numeric_df):
    with pytest.raises(ValueError):
        trainer.train_models(numeric_df, "target", ["target", "x1"])


# ------------------------------------------------------------------ Forecasting
def test_forecast_monthly_with_baseline(monthly_series_df):
    info = fc.prepare_series(monthly_series_df, "month", "pm25")
    assert info.freq == "MS" and info.n_periods == 60
    res = fc.run_forecast(info, ["moving_average", "linear_trend"], horizon=6)
    assert res["ok"]
    assert "naive" in res["results"]                       # baseline luôn được so sánh
    assert len(res["test"]) == res["test_size"] == 12
    assert res["train"].index.max() < res["test"].index.min()   # chia theo thời gian
    lt = res["results"]["linear_trend"]
    assert lt["metrics"]["RMSE"] < res["results"]["naive"]["metrics"]["RMSE"]
    assert len(lt["future"]) == 6 and (lt["future_hi"] >= lt["future_lo"]).all()
    assert res["explanation"]


def test_forecast_arima_requires_statsmodels(monthly_series_df):
    pytest.importorskip("statsmodels")
    info = fc.prepare_series(monthly_series_df, "month", "pm25")
    res = fc.run_forecast(info, ["arima", "sarima"], horizon=3)
    assert "arima" in res["results"] and np.isfinite(res["results"]["arima"]["metrics"]["RMSE"])


def test_forecast_yearly_short_series_rejects_arima(job_df):
    info = fc.prepare_series(job_df, "year", None)         # đếm số tin mỗi năm (4 kỳ)
    assert info.freq == "YS" and info.n_periods == 4
    elig = fc.check_eligibility(info.n_periods, info.freq, 1)
    assert not elig["arima"][0] and not elig["sarima"][0]
    res = fc.run_forecast(info, ["arima", "linear_trend"], horizon=1)
    assert res["ok"] and "arima" in res["skipped"]


def test_forecast_rejects_without_time_column(numeric_df):
    with pytest.raises(ValueError):
        fc.prepare_series(numeric_df.assign(bad="abc"), "bad", "x1")


def test_forecast_too_short():
    df = pd.DataFrame({"year": [2023, 2024, 2025], "v": [1, 2, 3]})
    res = fc.run_forecast(fc.prepare_series(df, "year", "v"), ["linear_trend"])
    assert not res["ok"] and "không đủ" in res["reason"].lower()


def test_forecast_missing_periods_filled():
    df = pd.DataFrame({"d": ["2024-01-01", "2024-02-01", "2024-04-01", "2024-05-01", "2024-06-01"], "v": [1, 2, 4, 5, 6]})
    info = fc.prepare_series(df, "d", "v", freq="MS")
    assert info.n_missing_periods == 1 and info.series.isna().sum() == 0


# ------------------------------------------------------------------ Anomaly
def test_isolation_forest(numeric_df):
    df = numeric_df.copy()
    df.loc[0, ["x1", "x2"]] = [500, -400]
    res = detector.detect(df, ["x1", "x2", "x3"], contamination=0.02)
    assert res["n_anomalies"] == pytest.approx(0.02 * len(df), abs=2)
    assert bool(res["frame"].loc[0, "is_anomaly"])
    assert res["frame"]["anomaly_score"].idxmax() == 0
    assert detector.describe(res)
    with pytest.raises(ValueError):
        detector.detect(df, ["x1"], contamination=0.9)
