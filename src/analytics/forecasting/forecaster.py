"""Dự báo chuỗi thời gian: Naive, Moving Average, Linear Trend, ARIMA, SARIMA.

Nguyên tắc khoa học:
- Sắp xếp theo thời gian, gộp về tần suất đều, báo cáo số kỳ bị thiếu và cách điền.
- Chia train/test THEO THỨ TỰ THỜI GIAN (không xáo trộn) - test luôn là các kỳ cuối.
- Mọi mô hình được so sánh với baseline Naive; báo cáo MAE, RMSE, MAPE (khi không có giá trị 0).
- Từ chối mô hình khi không đủ điều kiện (số điểm tối thiểu, không có mùa vụ...) và nêu rõ lý do.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.analytics.config import MIN_POINTS, SARIMA_MIN_SEASONS
from src.analytics.data import schema
from src.analytics.analysis.eda import FREQ_LABEL, infer_frequency

try:
    from scipy import stats as _sstats
except Exception:  # pragma: no cover
    _sstats = None

SEASON_LENGTH = {"MS": 12, "QS": 4, "W": 52, "D": 7, "h": 24, "YS": 1}
MODEL_LABELS = {"naive": "Naive (baseline)", "moving_average": "Moving Average", "linear_trend": "Linear Trend",
                "holt": "Holt (damped)", "arima": "ARIMA", "sarima": "SARIMA"}


def has_statsmodels() -> bool:
    try:
        import statsmodels  # noqa: F401
        return True
    except Exception:
        return False


@dataclass
class SeriesInfo:
    series: pd.Series
    freq: str
    n_periods: int
    n_missing_periods: int
    fill_method: str
    partial_last: bool = False
    notes: list[str] = field(default_factory=list)


def prepare_series(df: pd.DataFrame, time_col: str, value_col: str | None = None, agg: str = "mean",
                   freq: str | None = None) -> SeriesInfo:
    """Gộp dữ liệu về chuỗi thời gian đều. value_col=None ⇒ đếm số bản ghi mỗi kỳ."""
    if time_col not in df.columns:
        raise ValueError("Dữ liệu không có cột thời gian đã chọn.")
    t = schema.to_time_index(df[time_col])
    valid_t = t.notna()
    if valid_t.sum() == 0:
        raise ValueError(f"Cột '{time_col}' không chuyển được sang kiểu thời gian - không thể dự báo.")
    notes = []
    if (~valid_t).sum():
        notes.append(f"Bỏ {int((~valid_t).sum()):,} dòng có thời gian không hợp lệ.")
    frame = pd.DataFrame({"t": t[valid_t].values})
    if value_col:
        frame["v"] = pd.to_numeric(df.loc[valid_t, value_col], errors="coerce").values
        if frame["v"].notna().sum() == 0:
            raise ValueError(f"Cột '{value_col}' không có giá trị số hợp lệ.")
    else:
        frame["v"] = 1.0
        agg = "sum"
    frame = frame.sort_values("t").set_index("t")
    freq = freq or infer_frequency(frame.index.to_series())
    counts = frame["v"].resample(freq).count()
    s = frame["v"].resample(freq).agg(agg).where(counts > 0)
    n_missing = int(s.isna().sum())
    fill = "không cần"
    if n_missing:
        if value_col is None:
            s = s.fillna(0)
            fill = "điền 0 (kỳ không có bản ghi)"
        else:
            # Bảo toàn tính nhân quả: không dùng dữ liệu tương lai để nội suy các kỳ trước.
            s = s.ffill()
            fill = "điền giá trị kỳ trước (forward fill, không nhìn tương lai)"
        notes.append(f"Có {n_missing} kỳ không có dữ liệu ⇒ {fill}.")
    partial = False
    if len(counts) >= 3:
        last, med = counts.iloc[-1], counts.iloc[:-1].median()
        if med > 0 and last < 0.5 * med:
            partial = True
            notes.append(f"Kỳ cuối ({_label(s.index[-1], freq)}) chỉ có {int(last):,} bản ghi so với trung vị {int(med):,}/kỳ "
                         "⇒ có thể là kỳ chưa thu thập đủ; nên cân nhắc loại khi dự báo số lượng.")
    s.name = value_col or "Số bản ghi"
    return SeriesInfo(s.astype(float), freq, len(s), n_missing, fill, partial, notes)


def _label(ts, freq: str) -> str:
    if freq == "YS":
        return str(ts.year)
    if freq in ("MS", "QS"):
        return ts.strftime("%m/%Y")
    return ts.strftime("%d/%m/%Y")


def check_eligibility(n: int, freq: str, test_size: int) -> dict[str, tuple[bool, str]]:
    n_train = n - test_size
    m = SEASON_LENGTH.get(freq, 1)
    out = {}
    for key in ("naive", "moving_average", "linear_trend", "holt", "arima"):
        need = MIN_POINTS[key]
        ok = n_train >= need
        out[key] = (ok, "Đủ điều kiện" if ok else f"Cần ≥ {need} kỳ huấn luyện, hiện có {n_train}.")
    if out["holt"][0] and not has_statsmodels():
        out["holt"] = (False, "Chưa cài statsmodels.")
    if out["arima"][0] and not has_statsmodels():
        out["arima"] = (False, "Chưa cài statsmodels.")
    if m <= 1:
        out["sarima"] = (False, f"Tần suất '{FREQ_LABEL.get(freq, freq)}' không có chu kỳ mùa vụ xác định.")
    elif n_train < SARIMA_MIN_SEASONS * m:
        out["sarima"] = (False, f"Cần ≥ {SARIMA_MIN_SEASONS} chu kỳ mùa vụ ({SARIMA_MIN_SEASONS * m} kỳ) trên tập train, hiện có {n_train}.")
    elif not has_statsmodels():
        out["sarima"] = (False, "Chưa cài statsmodels.")
    else:
        out["sarima"] = (True, "Đủ điều kiện")
    return out


# ------------------------------------------------------------------ Các mô hình
def _z(level: float = 0.95) -> float:
    return 1.959963984540054 if level == 0.95 else float(_sstats.norm.ppf(0.5 + level / 2)) if _sstats else 1.96


def fc_naive(train: np.ndarray, h: int):
    sigma = np.std(np.diff(train), ddof=1) if len(train) > 2 else 0.0
    pred = np.repeat(train[-1], h)
    width = _z() * sigma * np.sqrt(np.arange(1, h + 1))
    return pred, pred - width, pred + width, {}


def fc_moving_average(train: np.ndarray, h: int, window: int = 3):
    window = max(1, min(window, len(train)))
    level = float(np.mean(train[-window:]))
    resid = [train[i] - np.mean(train[i - window:i]) for i in range(window, len(train))]
    sigma = np.std(resid, ddof=1) if len(resid) > 1 else 0.0
    pred = np.repeat(level, h)
    width = _z() * sigma * np.sqrt(1 + np.arange(h) / window)
    return pred, pred - width, pred + width, {"window": window}


def fc_linear_trend(train: np.ndarray, h: int):
    n = len(train)
    x = np.arange(n, dtype=float)
    b, a = np.polyfit(x, train, 1)
    fitted = a + b * x
    resid = train - fitted
    dof = max(n - 2, 1)
    s = np.sqrt(np.sum(resid ** 2) / dof)
    xf = np.arange(n, n + h, dtype=float)
    pred = a + b * xf
    tcrit = float(_sstats.t.ppf(0.975, dof)) if _sstats else 1.96
    se = s * np.sqrt(1 + 1 / n + (xf - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    return pred, pred - tcrit * se, pred + tcrit * se, {"slope": float(b), "intercept": float(a)}


def fc_holt(train: np.ndarray, h: int):
    """Holt xu hướng tắt dần, khoảng dự báo xấp xỉ từ sai số fit train."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = ExponentialSmoothing(np.asarray(train, dtype=float), trend="add", damped_trend=True,
                                      initialization_method="estimated").fit(optimized=True)
    pred = np.asarray(fitted.forecast(h), dtype=float)
    err = np.asarray(train, dtype=float) - np.asarray(fitted.fittedvalues, dtype=float)
    sigma = float(np.std(err, ddof=1)) if len(err) > 1 else 0.0
    w = _z() * sigma * np.sqrt(np.arange(1, h + 1))
    return pred, pred - w, pred + w, {"method": "Holt damped trend", "train_residual_std": sigma}


def _fit_arima(train: np.ndarray, seasonal_m: int = 0):
    from statsmodels.tsa.arima.model import ARIMA

    best, best_aic, best_order = None, np.inf, None
    orders = [(p, d, q) for p in (0, 1, 2) for d in (0, 1) for q in (0, 1, 2)]
    seasonal = (1, 1, 1, seasonal_m) if seasonal_m > 1 else (0, 0, 0, 0)
    if seasonal_m > 1:
        orders = [(p, d, q) for p in (0, 1) for d in (0, 1) for q in (0, 1)]
    for order in orders:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ARIMA(train, order=order, seasonal_order=seasonal,
                            trend="ct" if order[1] == 0 and seasonal_m <= 1 else None).fit()
            if np.isfinite(res.aic) and res.aic < best_aic:
                best, best_aic, best_order = res, res.aic, order
        except Exception:
            continue
    if best is None:
        raise RuntimeError("Không mô hình ARIMA nào hội tụ trên dữ liệu này.")
    return best, best_order, seasonal, best_aic


def fc_arima(train: np.ndarray, h: int, seasonal_m: int = 0):
    res, order, seasonal, aic = _fit_arima(train, seasonal_m)
    f = res.get_forecast(h)
    ci = np.asarray(f.conf_int(alpha=0.05))
    params = {"order": order, "aic": float(aic)}
    if seasonal_m > 1:
        params["seasonal_order"] = seasonal
    return np.asarray(f.predicted_mean), ci[:, 0], ci[:, 1], params


FORECASTERS = {"naive": fc_naive, "moving_average": fc_moving_average, "linear_trend": fc_linear_trend,
               "holt": fc_holt, "arima": lambda tr, h, **k: fc_arima(tr, h, 0)}


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    err = actual - pred
    out = {"MAE": float(np.mean(np.abs(err))), "RMSE": float(np.sqrt(np.mean(err ** 2)))}
    out["MAPE (%)"] = float(np.mean(np.abs(err / actual)) * 100) if np.all(np.abs(actual) > 1e-9) else np.nan
    return out


def _forecast_fn(key: str, seasonal_length: int):
    return (lambda tr, h, **kwargs: fc_arima(tr, h, seasonal_length)) if key == "sarima" else FORECASTERS[key]


def run_forecast(info: SeriesInfo, models: list[str], horizon: int = 3, test_size: int | None = None,
                 ma_window: int = 3, exclude_partial: bool = False, validation_size: int | None = None) -> dict:
    """Chọn mô hình từ validation, đánh giá test sau khi đã khóa lựa chọn.

    train | validation | test được phân chia liên tục theo thời gian.
    Không dùng test để chọn tên mô hình, độ dài validation hay tham số.
    """
    s = info.series.copy()
    notes = list(info.notes)
    if exclude_partial and info.partial_last:
        s = s.iloc[:-1]
        notes.append("Đã loại kỳ cuối (chưa đầy đủ) khỏi dự báo.")
    n = len(s)
    if n < 4:
        return {"ok": False, "reason": f"Chuỗi chỉ có {n} kỳ - không đủ điều kiện để huấn luyện và kiểm định (cần ≥ 4 kỳ).",
                "notes": notes}
    test_size = test_size or max(1, min(int(round(n * 0.2)), 12))
    test_size = min(test_size, n - 3)
    val_size = validation_size if validation_size is not None else min(test_size, n - test_size - 2)
    if not isinstance(val_size, (int, np.integer)):
        return {"ok": False, "reason": "validation_size phải là số nguyên.", "notes": notes}
    if val_size < 1 or n - test_size - val_size < 2:
        return {"ok": False, "reason": "Cần ít nhất 2 kỳ train, 1 kỳ validation, 1 kỳ test.", "notes": notes}

    train = s.iloc[:-(test_size + val_size)]
    validation = s.iloc[-(test_size + val_size):-test_size]
    test = s.iloc[-test_size:]
    m = SEASON_LENGTH.get(info.freq, 1)
    eligible = check_eligibility(n - test_size, info.freq, val_size)
    if len(train) >= 2:
        # Naive vẫn có thể chạy trên chuỗi rất ngắn, đồng thời làm baseline.
        eligible["naive"] = (True, "Đủ điều kiện")
    run_list = list(dict.fromkeys(["naive"] + list(models)))
    validation_results, skipped = {}, {}
    for key in run_list:
        ok, why = eligible.get(key, (False, "Mô hình không xác định"))
        if not ok:
            skipped[key] = why
            continue
        try:
            fn = _forecast_fn(key, m)
            kwargs = {"window": ma_window} if key == "moving_average" else {}
            val_pred, _, _, _ = fn(train.to_numpy(), val_size, **kwargs)
            if len(val_pred) != val_size or not np.isfinite(val_pred).all():
                raise ValueError("Giá trị dự báo validation không hợp lệ")
            validation_results[key] = metrics(validation.to_numpy(), val_pred)
        except Exception as exc:
            skipped[key] = f"Lỗi khi chạy validation: {exc}"

    if not validation_results:
        return {"ok": False, "reason": "Không mô hình nào chạy được trên validation.", "skipped": skipped, "notes": notes}

    # Cố định best ở đây, TRƯỚC khi bất kỳ hàm nào đọc giá trị test.
    best_key = min(validation_results, key=lambda k: validation_results[k]["RMSE"])
    results = {}
    for key in validation_results:
        try:
            fn = _forecast_fn(key, m)
            kwargs = {"window": ma_window} if key == "moving_average" else {}
            p_te, lo_te, hi_te, _ = fn(s.iloc[:-test_size].to_numpy(), test_size, **kwargs)
            if len(p_te) != test_size or not np.isfinite(p_te).all():
                raise ValueError("Giá trị dự báo test không hợp lệ")
            met = metrics(test.to_numpy(), p_te)
            p_f, lo_f, hi_f, params_full = fn(s.to_numpy(), horizon, **kwargs)
            results[key] = {"label": MODEL_LABELS[key], "test_pred": p_te, "test_lo": lo_te, "test_hi": hi_te,
                            "metrics": met, "validation_metrics": validation_results[key],
                            "future": p_f, "future_lo": lo_f, "future_hi": hi_f, "params": params_full}
        except Exception as exc:
            skipped[key] = f"Lỗi khi đánh giá test / tái huấn luyện: {exc}"

    if best_key not in results:
        return {"ok": False, "reason": "Mô hình đã chọn trên validation không chạy được trên test; không chọn lại bằng kết quả test.",
                "skipped": skipped, "notes": notes, "selected": best_key}
    future_index = pd.date_range(s.index[-1], periods=horizon + 1, freq=info.freq)[1:]
    rows = []
    base_rmse = results["naive"]["metrics"]["RMSE"] if "naive" in results else np.nan
    for key, r in results.items():
        improve = (1 - r["metrics"]["RMSE"] / base_rmse) * 100 if np.isfinite(base_rmse) and base_rmse > 0 else np.nan
        rows.append({"Mô hình": r["label"], "MAE": r["metrics"]["MAE"], "RMSE": r["metrics"]["RMSE"],
                     "MAPE (%)": r["metrics"]["MAPE (%)"], "RMSE validation": r["validation_metrics"]["RMSE"],
                     "Cải thiện RMSE so với Naive (%)": improve})
    table = pd.DataFrame(rows).sort_values("RMSE validation", kind="stable").reset_index(drop=True)
    numeric = table.select_dtypes("number").columns
    table[numeric] = table[numeric].round(4)
    return {"ok": True, "series": s, "train": train, "validation": validation, "test": test,
            "validation_size": val_size, "validation_metrics": validation_results, "freq": info.freq,
            "test_size": test_size, "horizon": horizon, "future_index": future_index, "results": results,
            "table": table, "best": best_key, "selection_metric": "validation RMSE", "skipped": skipped, "notes": notes,
            "explanation": explain(results, best_key, table, s, info.freq, test_size, val_size)}


def explain(results: dict, best: str, table: pd.DataFrame, s: pd.Series, freq: str,
            test_size: int, validation_size: int = 0) -> list[str]:
    msgs = []
    r = results[best]
    label = r["label"]
    msgs.append(f"Chia train / validation / test theo thời gian; {validation_size} kỳ validation dùng CHỌN mô hình "
                f"theo RMSE, {test_size} kỳ test ({_label(s.index[-test_size], freq)} → {_label(s.index[-1], freq)}) "
                "chỉ dùng đánh giá sau khi đã chọn.")
    msgs.append(f"Đã chọn {label} (validation RMSE = {r['validation_metrics']['RMSE']:.4g}); "
                f"trên test đạt RMSE = {r['metrics']['RMSE']:.4g} và MAE = {r['metrics']['MAE']:.4g}.")
    if best == "naive":
        msgs.append("Naive có validation RMSE thấp nhất; không có cơ sở kết luận mô hình phức tạp hơn tốt hơn.")
    if "linear_trend" in results:
        slope = results["linear_trend"]["params"].get("slope", 0)
        msgs.append(f"Độ dốc xu hướng tuyến tính sau khi huấn luyện trên toàn chuỗi: {slope:+,.4g} đơn vị/kỳ.")
    if len(s) < 10:
        msgs.append("Chuỗi rất ngắn; tập validation và test nhỏ khiến sai số có độ bất định cao.")
    msgs.append(f"Dự báo kỳ kế tiếp theo {label}: {r['future'][0]:,.4g} "
                f"(khoảng 95%: {r['future_lo'][0]:,.4g} – {r['future_hi'][0]:,.4g}).")
    return msgs
