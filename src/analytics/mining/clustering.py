"""Khai phá dữ liệu: K-Means, Elbow, Silhouette, PCA, hồ sơ cụm và mẫu đáng chú ý."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.analytics.config import MAX_CLUSTER_ROWS, RANDOM_STATE, SILHOUETTE_SAMPLE


def prepare_matrix(df: pd.DataFrame, features: list[str], max_rows: int = MAX_CLUSTER_ROWS):
    """Điền thiếu (median) + chuẩn hóa. Lấy mẫu ngẫu nhiên có cố định random_state nếu dữ liệu quá lớn."""
    if len(features) < 2:
        raise ValueError("Cần chọn ít nhất 2 biến số.")
    data = df[features].apply(pd.to_numeric, errors="coerce").astype(float)
    data = data.loc[:, data.notna().any()]
    if data.shape[1] < 2:
        raise ValueError("Các biến được chọn không có đủ giá trị số hợp lệ.")
    sampled = False
    if len(data) > max_rows:
        data = data.sample(max_rows, random_state=RANDOM_STATE)
        sampled = True
    prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    X = prep.fit_transform(data)
    return X, data, sampled


def elbow_silhouette(X: np.ndarray, k_min: int = 2, k_max: int = 10) -> pd.DataFrame:
    rows = []
    n = X.shape[0]
    k_max = min(k_max, n - 1)
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        sil = silhouette_score(X, km.labels_, sample_size=min(SILHOUETTE_SAMPLE, n), random_state=RANDOM_STATE) \
            if 1 < len(set(km.labels_)) < n else np.nan
        rows.append({"k": k, "Inertia (WCSS)": float(km.inertia_), "Silhouette": float(sil)})
    return pd.DataFrame(rows)


def suggest_k(table: pd.DataFrame) -> int:
    """Gợi ý k theo Silhouette cao nhất (tiêu chí định lượng), Elbow dùng để đối chiếu trực quan."""
    if table.empty or table["Silhouette"].isna().all():
        return 3
    return int(table.loc[table["Silhouette"].idxmax(), "k"])


def elbow_point(table: pd.DataFrame) -> int | None:
    """Điểm khuỷu: k có khoảng cách lớn nhất tới đường nối điểm đầu-cuối của đường inertia."""
    if len(table) < 3:
        return None
    x = table["k"].to_numpy(float)
    y = table["Inertia (WCSS)"].to_numpy(float)
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min() or 1)
    d = np.abs((yn[-1] - yn[0]) * xn - (xn[-1] - xn[0]) * yn + xn[-1] * yn[0] - yn[-1] * xn[0])
    return int(x[int(np.argmax(d))])


def run_kmeans(df: pd.DataFrame, features: list[str], k: int) -> dict:
    X, data, sampled = prepare_matrix(df, features)
    if k < 2 or k >= len(X):
        raise ValueError("Số cụm không hợp lệ.")
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
    labels = km.labels_
    sil = float(silhouette_score(X, labels, sample_size=min(SILHOUETTE_SAMPLE, len(X)), random_state=RANDOM_STATE))
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(X)
    coords = pca.transform(X)
    loadings = pd.DataFrame(pca.components_.T, index=data.columns, columns=["PC1", "PC2"]).round(4)
    profile = cluster_profiles(data, labels)
    proj = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "Cụm": [f"Cụm {l + 1}" for l in labels]}, index=data.index)
    return {
        "k": k, "features": list(data.columns), "labels": pd.Series(labels + 1, index=data.index, name="cluster"),
        "silhouette": sil, "inertia": float(km.inertia_), "projection": proj,
        "explained_variance": [float(v) for v in pca.explained_variance_ratio_], "loadings": loadings,
        "profile": profile, "descriptions": describe_clusters(profile, data), "sampled": sampled, "n": len(data),
    }


def cluster_profiles(data: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    frame = data.copy()
    frame["Cụm"] = [f"Cụm {l + 1}" for l in labels]
    prof = frame.groupby("Cụm").mean(numeric_only=True)
    prof.insert(0, "Số bản ghi", frame.groupby("Cụm").size())
    prof.insert(1, "Tỷ lệ (%)", (100 * prof["Số bản ghi"] / len(frame)).round(2))
    return prof.round(3)


def describe_clusters(profile: pd.DataFrame, data: pd.DataFrame) -> list[str]:
    """Mô tả cụm theo độ lệch chuẩn hóa (z) của trung bình cụm so với trung bình toàn bộ."""
    mu, sd = data.mean(), data.std().replace(0, np.nan)
    out = []
    for cl, row in profile.iterrows():
        z = ((row[data.columns] - mu) / sd).dropna()
        hi = z[z > 0.5].sort_values(ascending=False).head(3)
        lo = z[z < -0.5].sort_values().head(3)
        parts = [f"{c} cao (+{v:.2f}σ)" for c, v in hi.items()] + [f"{c} thấp ({v:.2f}σ)" for c, v in lo.items()]
        desc = "; ".join(parts) if parts else "các đặc trưng gần mức trung bình chung"
        out.append(f"**{cl}** ({int(row['Số bản ghi']):,} bản ghi, {row['Tỷ lệ (%)']:.1f}%): {desc}.")
    return out


def silhouette_label(s: float) -> str:
    if s >= 0.7:
        return "cấu trúc cụm mạnh"
    if s >= 0.5:
        return "cấu trúc cụm hợp lý"
    if s >= 0.25:
        return "cấu trúc cụm yếu, các cụm chồng lấn một phần"
    return "không có cấu trúc cụm rõ ràng"


def notable_patterns(df: pd.DataFrame, features: list[str], top: int = 5) -> list[str]:
    """Mẫu đáng chú ý: cặp tương quan mạnh, biến có tỷ lệ giá trị cực đoan cao."""
    from src.analytics.analysis.eda import correlation_matrix, top_correlations

    msgs = []
    corr = correlation_matrix(df, features, "spearman")
    for _, r in top_correlations(corr, top).iterrows():
        if abs(r["Hệ số"]) >= 0.3:
            msgs.append(f"{r['Biến 1']} ↔ {r['Biến 2']}: Spearman ρ = {r['Hệ số']:.3f} ({r['Mức độ'].lower()}).")
    for c in features:
        x = pd.to_numeric(df[c], errors="coerce").astype(float).dropna()
        if len(x) < 10:
            continue
        q1, q3 = x.quantile([.25, .75])
        iqr = q3 - q1
        if iqr > 0:
            share = ((x < q1 - 3 * iqr) | (x > q3 + 3 * iqr)).mean()
            if share > 0.01:
                msgs.append(f"{c}: {share * 100:.1f}% giá trị là ngoại lệ cực đoan (ngoài 3×IQR).")
    return msgs
