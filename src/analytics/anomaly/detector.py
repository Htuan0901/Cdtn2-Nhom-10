"""Phát hiện bất thường đa biến bằng Isolation Forest."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.analytics.config import RANDOM_STATE


def detect(df: pd.DataFrame, features: list[str], contamination: float | str = 0.05, n_estimators: int = 200) -> dict:
    if not features:
        raise ValueError("Cần chọn ít nhất 1 biến số.")
    data = df[features].apply(pd.to_numeric, errors="coerce").astype(float)
    data = data.loc[:, data.notna().any()]
    if data.empty or data.shape[1] == 0:
        raise ValueError("Các biến được chọn không có giá trị số hợp lệ.")
    if len(data) < 20:
        raise ValueError("Cần ít nhất 20 bản ghi để phát hiện bất thường.")
    if contamination != "auto":
        contamination = float(contamination)
        if not 0 < contamination <= 0.5:
            raise ValueError("contamination phải nằm trong (0, 0.5].")
    prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    X = prep.fit_transform(data)
    model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=RANDOM_STATE, n_jobs=-1)
    flags = model.fit_predict(X) == -1
    score = -model.score_samples(X)       # càng lớn càng bất thường
    out = df.copy()
    out["anomaly_score"] = np.round(score, 5)
    out["is_anomaly"] = flags
    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X) if X.shape[1] >= 2 else np.c_[X[:, 0], np.zeros(len(X))]
    proj = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "Trạng thái": np.where(flags, "Bất thường", "Bình thường"),
                         "Điểm": score}, index=df.index)
    comp = pd.DataFrame({
        "Trung bình (bình thường)": data[~flags].mean(), "Trung bình (bất thường)": data[flags].mean(),
        "Trung vị (bình thường)": data[~flags].median(), "Trung vị (bất thường)": data[flags].median(),
    }).round(3)
    n = int(flags.sum())
    return {"frame": out, "n_anomalies": n, "rate": n / len(out), "projection": proj, "comparison": comp,
            "features": list(data.columns), "contamination": contamination,
            "threshold": float(np.min(score[flags])) if n else float("nan")}


def describe(result: dict) -> list[str]:
    msgs = [f"Phát hiện {result['n_anomalies']:,} điểm bất thường ({result['rate'] * 100:.2f}% dữ liệu) với contamination = {result['contamination']}."]
    comp = result["comparison"]
    for feat, row in comp.iterrows():
        a, b = row["Trung vị (bất thường)"], row["Trung vị (bình thường)"]
        if pd.notna(a) and pd.notna(b) and b != 0 and abs(a - b) / abs(b) > 0.2:
            msgs.append(f"{feat}: trung vị nhóm bất thường {a:,.4g} so với {b:,.4g} ở nhóm bình thường ({(a - b) / abs(b) * 100:+.0f}%).")
    msgs.append("Bất thường không đồng nghĩa với lỗi: cần kiểm tra thủ công các bản ghi có điểm cao nhất trước khi loại bỏ.")
    return msgs
