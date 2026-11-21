"""Biểu đồ tĩnh (PNG) bằng Matplotlib/Seaborn để nhúng vào báo cáo PDF và xuất riêng."""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

NAVY, BLUE, TEAL, GREY = "#0B1F3A", "#1D4ED8", "#14B8A6", "#94A3B8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 11, "axes.edgecolor": "#CBD5E1"})


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    return buf.getvalue()


def missing_bar(df: pd.DataFrame, top: int = 15) -> bytes | None:
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0].head(top)
    if miss.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(miss) + 1.2))
    ax.barh(miss.index[::-1].astype(str), miss.values[::-1], color=BLUE)
    ax.set_xlabel("% giá trị thiếu")
    ax.set_title("Tỷ lệ giá trị thiếu theo cột")
    return _png(fig)


def corr_heatmap(corr: pd.DataFrame) -> bytes | None:
    if corr is None or corr.empty:
        return None
    n = len(corr)
    fig, ax = plt.subplots(figsize=(min(1 + 0.6 * n, 8), min(0.8 + 0.5 * n, 7)))
    sns.heatmap(corr, annot=n <= 12, fmt=".2f", cmap="RdBu", vmin=-1, vmax=1, center=0, ax=ax, cbar_kws={"shrink": .7},
                annot_kws={"size": 7})
    ax.set_title("Ma trận tương quan")
    return _png(fig)


def histogram(df: pd.DataFrame, col: str) -> bytes | None:
    x = pd.to_numeric(df[col], errors="coerce").astype(float).dropna()
    if x.empty:
        return None
    hi = x.quantile(0.99)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(x[x <= hi], bins=40, color=BLUE, edgecolor="white")
    ax.axvline(x.median(), color=TEAL, ls="--", label=f"Median = {x.median():,.3g}")
    ax.axvline(x.mean(), color=NAVY, ls=":", label=f"Mean = {x.mean():,.3g}")
    ax.legend(frameon=False)
    ax.set_title(f"Phân phối {col} (ẩn 1% giá trị lớn nhất)")
    return _png(fig)


def barh(labels, values, title: str, xlabel: str = "", color: str = BLUE) -> bytes | None:
    if len(labels) == 0:
        return None
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(labels) + 1.2))
    ax.barh([str(l) for l in labels][::-1], list(values)[::-1], color=color)
    for i, v in enumerate(list(values)[::-1]):
        ax.text(v, i, f" {v:,.3g}", va="center", fontsize=7, color="#334155")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    return _png(fig)


def skill_trend(share: pd.DataFrame, skills: list[str]) -> bytes | None:
    if share is None or not skills:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    pal = sns.color_palette("tab10", len(skills))
    for s, c in zip(skills, pal):
        ax.plot(share.index.astype(str), share[s], marker="o", label=s, color=c, lw=2)
    ax.set_ylabel("% số tin")
    ax.set_title("Tỷ lệ tin yêu cầu kỹ năng theo kỳ")
    ax.legend(ncol=3, frameon=False, fontsize=7)
    return _png(fig)


def model_comparison(comp: pd.DataFrame, metric: str) -> bytes | None:
    d = comp.dropna(subset=[metric])
    if d.empty:
        return None
    colors = [GREY if "Baseline" in m else BLUE for m in d["Mô hình"]]
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(d) + 1.2))
    ax.barh(d["Mô hình"][::-1], d[metric][::-1], color=colors[::-1])
    for i, v in enumerate(d[metric][::-1]):
        ax.text(v, i, f" {v:.4g}", va="center", fontsize=7)
    ax.set_title(f"So sánh mô hình theo {metric} (tập test)")
    return _png(fig)


def forecast_plot(res: dict) -> bytes | None:
    if not res:
        return None
    s, test = res["series"], res["test"]
    best = res["results"][res["best"]]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(s.index, s.values, color=NAVY, marker="o", lw=2, label="Thực tế")
    ax.plot(test.index, best["test_pred"], color=TEAL, ls="--", marker="x", label=f"{best['label']} (kiểm định)")
    if "naive" in res["results"] and res["best"] != "naive":
        ax.plot(test.index, res["results"]["naive"]["test_pred"], color=GREY, ls=":", label="Naive (kiểm định)")
    fx = res["future_index"]
    ax.plot([s.index[-1], *fx], [s.values[-1], *best["future"]], color=BLUE, marker="o", lw=2, label="Dự báo")
    ax.fill_between(fx, best["future_lo"], best["future_hi"], color=BLUE, alpha=0.15, label="Khoảng 95%")
    ax.set_title(f"Dự báo {s.name}")
    ax.legend(frameon=False, fontsize=7)
    fig.autofmt_xdate()
    return _png(fig)


def pca_clusters(proj: pd.DataFrame, ev: list[float]) -> bytes | None:
    if proj is None or proj.empty:
        return None
    d = proj.sample(min(len(proj), 4000), random_state=42)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for (name, g), c in zip(d.groupby("Cụm"), sns.color_palette("tab10")):
        ax.scatter(g["PC1"], g["PC2"], s=8, alpha=0.6, label=name, color=c)
    ax.set_xlabel(f"PC1 ({ev[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1] * 100:.1f}%)")
    ax.set_title("Các cụm K-Means trên không gian PCA")
    ax.legend(frameon=False, fontsize=7, markerscale=2)
    return _png(fig)


def anomaly_scatter(proj: pd.DataFrame) -> bytes | None:
    if proj is None or proj.empty:
        return None
    d = proj.sample(min(len(proj), 5000), random_state=42)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    norm, ab = d[d["Trạng thái"] == "Bình thường"], d[d["Trạng thái"] == "Bất thường"]
    ax.scatter(norm["PC1"], norm["PC2"], s=6, alpha=0.4, color="#93C5FD", label="Bình thường")
    ax.scatter(ab["PC1"], ab["PC2"], s=14, alpha=0.9, color="#DC2626", label="Bất thường")
    ax.set_title("Isolation Forest - điểm bất thường (PCA 2D)")
    ax.legend(frameon=False)
    return _png(fig)
