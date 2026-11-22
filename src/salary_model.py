"""Mô hình dự đoán mức lương (USD/năm) từ vị trí, cấp bậc, địa điểm, hình thức và kỹ năng (dữ liệu LinkedIn).

Bổ trợ cho mục tiêu "xu hướng mức lương theo vị trí" của đề tài: lượng hóa mỗi kỹ năng/cấp bậc làm lương thay đổi bao nhiêu.
So sánh với baseline (trung vị lương của tập train) bằng MAE, RMSE, R².
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .skills import skills_matrix

CAT_COLS = ["role", "experience", "state_grp", "work_type"]


def build_features(df: pd.DataFrame, top_states: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    if top_states is None:
        top_states = df["state"].value_counts().head(15).index.tolist()
    X = pd.DataFrame({
        "role": df["role"].values,
        "experience": df["experience"].values,
        "state_grp": np.where(df["state"].isin(top_states), df["state"], "Khác").astype(str),
        "work_type": df["work_type"].values,
        "remote": df["remote"].astype(int).values,
    }, index=df.index)
    sk = skills_matrix(df["skills"])
    X = pd.concat([X, sk.add_prefix("sk_")], axis=1)
    return X, top_states


def _make(model):
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CAT_COLS)],
                            remainder="passthrough", sparse_threshold=0)
    return Pipeline([("pre", pre), ("model", model)])


def salary_trend_table(li: pd.DataFrame, freq: str = "MS", min_posts: int = 5,
                       roles: list[str] | None = None) -> pd.DataFrame:
    """Trung vị lương (USD/năm) mỗi tháng/quý × vị trí, kèm cỡ mẫu.

    Không tự suy diễn hoặc nội suy kỳ thiếu; kỳ có < min_posts được loại
    để tránh trình bày một vài tin đăng như một xu hướng thị trường.
    """
    if freq not in ("MS", "QS"):
        raise ValueError("Chỉ hỗ trợ nhóm theo tháng (MS) hoặc quý (QS)")
    if min_posts < 1:
        raise ValueError("min_posts cần >= 1")
    cols = ["Kỳ", "Vị trí", "Lương trung vị (USD/năm)", "Số tin"]
    d = li.loc[:, ["listed_date", "role", "salary_usd"]].copy()
    d["listed_date"] = pd.to_datetime(d["listed_date"], errors="coerce")
    d["salary_usd"] = pd.to_numeric(d["salary_usd"], errors="coerce")
    d = d.dropna(subset=["listed_date", "role", "salary_usd"])
    d = d[d["salary_usd"] > 0]
    if roles is not None:
        d = d[d["role"].isin(roles)]
    if d.empty:
        return pd.DataFrame(columns=cols)
    out = (d.groupby([pd.Grouper(key="listed_date", freq=freq), "role"])["salary_usd"]
           .agg(["median", "size"]).reset_index()
           .rename(columns={"listed_date": "Kỳ", "role": "Vị trí",
                            "median": "Lương trung vị (USD/năm)", "size": "Số tin"}))
    return out[out["Số tin"] >= min_posts].sort_values(["Kỳ", "Vị trí"]).reset_index(drop=True)[cols]


def train_salary_models(li: pd.DataFrame, test_size: float = 0.2, seed: int = 42,
                        validation_size: float = 0.2) -> dict:
    """Chọn bằng validation MAE, test dùng báo cáo, không dùng chọn mô hình.

    Xây dựng top_states chỉ từ train để không rò rỉ phân phối tập giữ lại.
    Khi đã chọn, fit lại mọi mô hình trên train+validation và đánh giá test.
    """
    d = li.loc[li["salary_usd"].notna() & (li["salary_usd"] > 0)].copy()
    if len(d) < 20:
        raise ValueError("Cần >= 20 tin có mức lương hợp lệ")
    idx = np.arange(len(d))
    i_dev, i_test = train_test_split(idx, test_size=test_size, random_state=seed)
    i_train, i_val = train_test_split(i_dev, test_size=validation_size / (1 - test_size), random_state=seed)
    train_rows = d.iloc[i_train]
    top_states = train_rows["state"].value_counts().head(15).index.tolist()
    X, _ = build_features(d, top_states)
    y = d["salary_usd"].to_numpy(dtype=float)
    Xtr, Xva, Xte = X.iloc[i_train], X.iloc[i_val], X.iloc[i_test]
    ytr, yva, yte = y[i_train], y[i_val], y[i_test]
    cands = {
        "Baseline (trung vị)": DummyRegressor(strategy="median"),
        "Ridge (hồi quy tuyến tính)": Ridge(alpha=3.0),
        "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=seed),
        "Gradient Boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, early_stopping=True, random_state=seed),
    }
    validation_scores = {}
    for name, m in cands.items():
        pipe = _make(m)
        pipe.fit(Xtr, np.log(ytr))
        p = np.exp(pipe.predict(Xva))
        validation_scores[name] = {"MAE": float(mean_absolute_error(yva, p)),
                                   "R²": float(r2_score(yva, p))}
    # Lựa chọn hoàn toàn trước khi truy cập y_test / đánh giá test.
    candidates = [name for name in cands if not name.startswith("Baseline")]
    best = min(candidates, key=lambda name: validation_scores[name]["MAE"])

    rows, fitted, preds = [], {}, {}
    Xdev = X.iloc[i_dev]
    ydev = y[i_dev]
    for name, m in cands.items():
        t0 = time.time()
        pipe = _make(m)
        pipe.fit(Xdev, np.log(ydev))
        p = np.exp(pipe.predict(Xte))
        fitted[name], preds[name] = pipe, p
        rows.append({"Mô hình": name, "MAE (USD)": mean_absolute_error(yte, p),
                     "RMSE (USD)": float(np.sqrt(mean_squared_error(yte, p))), "R²": r2_score(yte, p),
                     "Validation MAE (USD)": validation_scores[name]["MAE"],
                     "Validation R²": validation_scores[name]["R²"],
                     "Thời gian (s)": time.time() - t0})
    res = pd.DataFrame(rows).sort_values("Validation MAE (USD)").reset_index(drop=True)
    base = res.loc[res["Mô hình"].str.startswith("Baseline"), "MAE (USD)"].iloc[0]
    res["Giảm MAE so với baseline (%)"] = (1 - res["MAE (USD)"] / base) * 100 if base > 0 else np.nan

    ridge = fitted["Ridge (hồi quy tuyến tính)"]
    names = ridge.named_steps["pre"].get_feature_names_out()
    coef = pd.Series(ridge.named_steps["model"].coef_, index=names)
    coef.index = coef.index.str.replace("cat__", "").str.replace("remainder__", "").str.replace("sk_", "Kỹ năng: ")
    effect = ((np.exp(coef) - 1) * 100).sort_values()
    return {"results": res, "best": best, "selection_metric": "validation MAE", "models": fitted,
            "validation_scores": validation_scores, "top_states": top_states, "y_test": yte,
            "pred_test": preds, "effect": effect, "n_train": len(Xtr), "n_validation": len(Xva),
            "n_test": len(Xte), "columns": list(X.columns)}


def predict_salary(bundle: dict, role: str, experience: str, state: str, work_type: str, remote: bool,
                   skills: list[str], model: str | None = None) -> float:
    row = pd.DataFrame([{"role": role, "experience": experience, "state": state, "work_type": work_type,
                         "remote": remote, "skills": ";".join(skills)}])
    X, _ = build_features(row, bundle["top_states"])
    X = X.reindex(columns=bundle["columns"], fill_value=0)
    m = bundle["models"][model or bundle["best"]]
    return float(np.exp(m.predict(X))[0])
