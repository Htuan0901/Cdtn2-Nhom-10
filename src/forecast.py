"""Phân tích xu hướng & dự báo chuỗi thời gian theo tháng (dữ liệu Hacker News 2018–2023).

Chuỗi mục tiêu (tùy chọn):
- Tổng số tin tuyển dụng mỗi tháng.
- Số tin nhắc tới một kỹ năng mỗi tháng.
- Tỉ lệ % tin nhắc tới một kỹ năng mỗi tháng (đã loại ảnh hưởng của tổng số tin).

Mô hình (đều là thống kê/ML cơ bản, dễ giải thích – đúng phạm vi đề tài):
- Baseline: Naive (giá trị tháng trước), Seasonal Naive (cùng tháng năm trước), Trung bình trượt 3 tháng.
- Hồi quy tuyến tính theo thời gian (Linear Trend) + biến mùa vụ (tháng).
- Holt (làm trơn mũ có xu hướng, damped) và Holt-Winters (xu hướng + mùa vụ 12 tháng) – statsmodels.
- Ridge hồi quy với biến trễ (lag 1, 2, 3, 12) – dự báo đệ quy.
Đánh giá: chia train/validation/test theo thời gian, chỉ chọn mô hình bằng validation RMSE;
chỉ số MAE/RMSE/MAPE báo cáo trên test độc lập; so sánh Naive.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

warnings.filterwarnings("ignore")

MODEL_INFO = {
    "Naive (baseline)": "Dự báo = giá trị của tháng gần nhất. Mốc so sánh tối thiểu.",
    "Seasonal Naive": "Dự báo = giá trị cùng tháng của năm trước (bắt mùa vụ).",
    "Trung bình trượt 3T": "Dự báo = trung bình 3 tháng gần nhất.",
    "Hồi quy xu hướng + mùa vụ": "Hồi quy tuyến tính theo chỉ số thời gian t và 11 biến giả tháng.",
    "Holt (damped)": "Làm trơn mũ có thành phần xu hướng tắt dần (statsmodels).",
    "Holt-Winters": "Làm trơn mũ có xu hướng + mùa vụ cộng tính chu kỳ 12 tháng (statsmodels).",
    "Ridge + biến trễ": "Hồi quy Ridge trên giá trị trễ 1, 2, 3, 12 tháng + tháng trong năm; dự báo đệ quy.",
    "ARIMA (chọn theo AIC)": "ARIMA(p,d,q) với p,q ∈ {0,1,2}, d ∈ {0,1}; chọn bậc có AIC nhỏ nhất trên tập train (statsmodels).",
    "SARIMA (mùa vụ 12)": "SARIMA(p,d,q)(1,1,1,12): ARIMA có thành phần mùa vụ chu kỳ 12 tháng; chỉ chạy khi train ≥ 36 tháng.",
}


# ============================ TẠO CHUỖI THỜI GIAN ============================

def monthly_series(hn: pd.DataFrame, skill: str | None = None, metric: str = "count") -> pd.Series:
    """metric: 'count' (số tin) hoặc 'share' (% tin nhắc tới skill)."""
    months = pd.date_range(hn["month"].min(), hn["month"].max(), freq="MS")
    total = hn.groupby("month").size().reindex(months, fill_value=0)
    if not skill:
        s = total.astype(float)
    else:
        has = hn["skills"].str.split(";").map(lambda xs: skill in xs)
        cnt = hn[has].groupby("month").size().reindex(months, fill_value=0)
        s = cnt.astype(float) if metric == "count" else (cnt / total.replace(0, np.nan) * 100).fillna(0)
    s.index.name = "month"
    return s


def skill_share_table(hn: pd.DataFrame, skills: list[str], freq: str = "QS") -> pd.DataFrame:
    """Tỉ lệ % tin nhắc tới từng kỹ năng theo kỳ (quý/năm) – dùng cho heatmap & line chart."""
    e = hn[["month", "skills"]].assign(skill=hn["skills"].str.split(";")).explode("skill")
    e = e[e["skill"].isin(skills)]
    per = hn.set_index("month").resample(freq).size()
    cnt = e.set_index("month").groupby("skill").resample(freq).size().unstack(0).reindex(per.index).fillna(0)
    return (cnt.div(per.replace(0, np.nan), axis=0) * 100).round(2)


def trend_ranking(hn: pd.DataFrame, min_posts: int = 200, early=("2018-01-01", "2019-12-31"),
                  late=("2022-01-01", "2023-12-31")) -> pd.DataFrame:
    """So sánh tỉ lệ nhắc tới mỗi kỹ năng giữa giai đoạn đầu và cuối + độ dốc xu hướng (điểm %/năm)."""
    e = hn[["month", "skills"]].assign(skill=hn["skills"].str.split(";")).explode("skill")
    e = e[e["skill"].notna() & (e["skill"] != "")]
    counts = e["skill"].value_counts()
    skills = counts[counts >= min_posts].index
    tot_m = hn.groupby("month").size()
    rows = []
    a = hn["month"].between(*early)
    b = hn["month"].between(*late)
    na, nb = a.sum(), b.sum()
    for sk in skills:
        m = e[e["skill"] == sk].groupby("month").size().reindex(tot_m.index, fill_value=0) / tot_m * 100
        t = (m.index - m.index[0]).days / 365.25
        slope = np.polyfit(t, m.values, 1)[0] if len(m) > 2 else np.nan
        pa = e[(e["skill"] == sk) & e["month"].between(*early)].shape[0] / max(na, 1) * 100
        pb = e[(e["skill"] == sk) & e["month"].between(*late)].shape[0] / max(nb, 1) * 100
        rows.append({"Kỹ năng": sk, "Số tin": int(counts[sk]), "Tỉ lệ đầu kỳ (%)": round(pa, 2),
                     "Tỉ lệ cuối kỳ (%)": round(pb, 2), "Thay đổi (điểm %)": round(pb - pa, 2),
                     "Tăng trưởng tương đối (%)": round((pb / pa - 1) * 100, 1) if pa > 0 else np.nan,
                     "Độ dốc (điểm %/năm)": round(slope, 3)})
    return pd.DataFrame(rows).sort_values("Thay đổi (điểm %)", ascending=False).reset_index(drop=True)


# ============================ MÔ HÌNH ============================

def _future_index(s: pd.Series, h: int) -> pd.DatetimeIndex:
    return pd.date_range(s.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")


def _naive(train, h):
    return np.repeat(train.iloc[-1], h)


def _snaive(train, h):
    if len(train) < 12:
        return _naive(train, h)
    last = train.iloc[-12:].values
    return np.array([last[i % 12] for i in range(h)])


def _ma3(train, h):
    return np.repeat(train.iloc[-3:].mean(), h)


def _design(idx: pd.DatetimeIndex, t0: pd.Timestamp):
    t = ((idx.year - t0.year) * 12 + idx.month - t0.month).values.reshape(-1, 1)
    months = pd.get_dummies(pd.Categorical(idx.month, categories=range(1, 13)), drop_first=True).values
    return np.hstack([t, months])


def _linear(train, h):
    X = _design(train.index, train.index[0])
    m = LinearRegression().fit(X, train.values)
    return m.predict(_design(_future_index(train, h), train.index[0]))


def _holt(train, h):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    m = ExponentialSmoothing(train.values, trend="add", damped_trend=True, initialization_method="estimated").fit()
    return m.forecast(h)


def _hw(train, h):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    if len(train) < 24:
        return _holt(train, h)
    m = ExponentialSmoothing(train.values, trend="add", seasonal="add", seasonal_periods=12, damped_trend=True,
                             initialization_method="estimated").fit()
    return m.forecast(h)


LAGS = [1, 2, 3, 12]


def _ridge_lags(train, h):
    y = train.values.astype(float)
    rows, target = [], []
    for i in range(max(LAGS), len(y)):
        rows.append([y[i - l] for l in LAGS] + [np.sin(2 * np.pi * train.index[i].month / 12),
                                                 np.cos(2 * np.pi * train.index[i].month / 12)])
        target.append(y[i])
    if len(rows) < 6:
        return _naive(train, h)
    m = Ridge(alpha=1.0).fit(np.array(rows), np.array(target))
    hist = list(y)
    out = []
    for d in _future_index(train, h):
        x = [hist[-l] for l in LAGS] + [np.sin(2 * np.pi * d.month / 12), np.cos(2 * np.pi * d.month / 12)]
        p = float(m.predict(np.array([x]))[0])
        out.append(p)
        hist.append(p)
    return np.array(out)


def _arima(train, h):
    from src.analytics.forecasting.forecaster import fc_arima
    if len(train) < 20:
        raise ValueError("ARIMA cần ≥ 20 tháng huấn luyện")
    return np.asarray(fc_arima(train.values.astype(float), h, 0)[0])


def _sarima(train, h):
    from src.analytics.forecasting.forecaster import fc_arima
    if len(train) < 36:
        raise ValueError("SARIMA cần ≥ 36 tháng (3 chu kỳ mùa vụ)")
    return np.asarray(fc_arima(train.values.astype(float), h, 12)[0])


MODELS = {
    "Naive (baseline)": _naive,
    "Seasonal Naive": _snaive,
    "Trung bình trượt 3T": _ma3,
    "Hồi quy xu hướng + mùa vụ": _linear,
    "Holt (damped)": _holt,
    "Holt-Winters": _hw,
    "Ridge + biến trễ": _ridge_lags,
    "ARIMA (chọn theo AIC)": _arima,
    "SARIMA (mùa vụ 12)": _sarima,
}


def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_true - y_pred
    nz = y_true != 0
    return {"MAE": float(np.mean(np.abs(err))), "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "MAPE (%)": float(np.mean(np.abs(err[nz] / y_true[nz])) * 100) if nz.any() else np.nan}


def _score_models(train: pd.Series, holdout: pd.Series, models: list[str]):
    """Một lượt dự báo holdout: mô hình chỉ nhìn thấy phần train trước đó."""
    rows, preds, skipped = [], {"Thực tế": holdout.to_numpy()}, {}
    for name in models:
        try:
            p = np.asarray(MODELS[name](train, len(holdout)), dtype=float)
            if len(p) != len(holdout) or not np.isfinite(p).all():
                raise ValueError("Dự báo có độ dài hoặc giá trị không hợp lệ")
            p = np.clip(p, 0, None)
            preds[name] = p
            rows.append({"Mô hình": name, **metrics(holdout.values, p)})
        except Exception as exc:
            skipped[name] = str(exc)
    table = pd.DataFrame(rows)
    if not table.empty:
        base = table.loc[table["Mô hình"] == "Naive (baseline)", "RMSE"]
        if len(base) and base.iloc[0] > 0:
            table["Cải thiện RMSE so với baseline (%)"] = (1 - table["RMSE"] / base.iloc[0]) * 100
        table = table.sort_values("RMSE", kind="stable").reset_index(drop=True)
    return table, pd.DataFrame(preds, index=holdout.index), skipped


def evaluate_models(s: pd.Series, h: int = 6, models: list[str] | None = None,
                    validation_size: int | None = None) -> dict:
    """Chọn trên validation, khóa lựa chọn, CHỈ SAU ĐÓ đánh giá trên test.

    Cắt thời gian thành: train | validation | test, tất cả không chồng lấn.
    Các mô hình được fit trên train để chọn bằng validation RMSE; sau khi
    chọn xong, fit lại trên train+validation để đánh giá một lần ở test.
    Bảng test của tất cả mô hình chỉ để so sánh/báo cáo, KHÔNG dùng chọn best.
    """
    if not 1 <= h < len(s):
        raise ValueError("Số kỳ test không hợp lệ")
    val_h = validation_size if validation_size is not None else h
    if not isinstance(val_h, int) or val_h < 1 or len(s) - h - val_h < 3:
        raise ValueError("Cần tối thiểu 3 kỳ train, 1 kỳ validation và 1 kỳ test")
    names = list(dict.fromkeys(["Naive (baseline)"] + (list(MODELS) if models is None else list(models))))
    invalid = [x for x in names if x not in MODELS]
    if invalid:
        raise ValueError(f"Mô hình không hỗ trợ: {invalid}")
    train = s.iloc[:-(h + val_h)]
    validation = s.iloc[-(h + val_h):-h]
    test = s.iloc[-h:]
    val_table, val_preds, val_skipped = _score_models(train, validation, names)
    if val_table.empty:
        raise ValueError("Không mô hình nào dự báo được trên validation")
    # QUAN TRỌNG: best được tính TRƯỚC khi gọi đánh giá test.
    best = str(val_table.iloc[0]["Mô hình"])
    test_table, test_preds, test_skipped = _score_models(s.iloc[:-h], test, names)
    # Thứ tự bảng test theo VALIDATION (không sắp theo điểm test như trước).
    # Vẫn báo cáo đầy đủ MAE/RMSE thực tế của tất cả mô hình.
    if not test_table.empty:
        val_scores = val_table.set_index("Mô hình")
        test_table["Validation RMSE"] = test_table["Mô hình"].map(val_scores["RMSE"])
        test_table["Validation MAE"] = test_table["Mô hình"].map(val_scores["MAE"])
        test_table = test_table.sort_values("Validation RMSE", kind="stable").reset_index(drop=True)
    return {"train": train, "validation": validation, "test": test,
            "validation_size": val_h, "validation_table": val_table,
            "validation_predictions": val_preds, "test_table": test_table,
            "test_predictions": test_preds, "best": best,
            "validation_skipped": val_skipped, "test_skipped": test_skipped}


def backtest(s: pd.Series, h: int = 6, models: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tương thích API cũ: trả bảng test, test predictions; CHỌN bằng validation.

    Dùng evaluate_models() nếu cần tên mô hình tốt nhất và bảng validation.
    """
    out = evaluate_models(s, h, models)
    table, predictions = out["test_table"], out["test_predictions"]
    table.attrs["selected_on_validation"] = out["best"]
    table.attrs["validation_size"] = out["validation_size"]
    return table, predictions


def rolling_backtest(s: pd.Series, h: int = 3, n_origins: int = 6, models: list[str] | None = None) -> pd.DataFrame:
    """Đánh giá rolling-origin với test không tràn vào tương lai từng mốc.

    Đây là bảng mô tả độ ổn định, không phải bảng chọn mô hình cuối cùng.
    """
    out = []
    names = list(dict.fromkeys(["Naive (baseline)"] + (list(MODELS) if models is None else list(models))))
    for k in range(n_origins, 0, -1):
        end = len(s) - (k - 1) * h
        sub = s.iloc[:end]
        if len(sub) < 24 + h:
            continue
        r, _, _ = _score_models(sub.iloc[:-h], sub.iloc[-h:], names)
        if not r.empty:
            out.append(r.assign(origin=sub.index[-h]))
    if not out:
        return pd.DataFrame()
    allr = pd.concat(out)
    g = allr.groupby("Mô hình")[["MAE", "RMSE", "MAPE (%)"]].mean()
    g["Số lần đánh giá"] = allr.groupby("Mô hình").size()
    return g.sort_values("RMSE").reset_index()


def forecast(s: pd.Series, model: str, h: int, resid_std: float | None = None) -> pd.DataFrame:
    """Huấn luyện trên TOÀN BỘ chuỗi rồi dự báo h tháng tới, kèm khoảng tin cậy ~95% (±1.96·σ·√k)."""
    p = np.clip(MODELS[model](s, h), 0, None)
    idx = _future_index(s, h)
    sd = resid_std if resid_std is not None else float(s.diff().std())
    band = 1.96 * sd * np.sqrt(np.arange(1, h + 1))
    return pd.DataFrame({"Dự báo": p, "Cận dưới": np.clip(p - band, 0, None), "Cận trên": p + band}, index=idx)
