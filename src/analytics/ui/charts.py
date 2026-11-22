"""Biểu đồ Plotly tương tác với template thống nhất. Tất cả nhận dữ liệu thật đã tính sẵn."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.analytics.ui.theme import BLUE, BORDER, DANGER, DIVERGING, MUTED, NAVY, PALETTE, SEQUENTIAL, TEAL, TEXT


def style(fig: go.Figure, title: str | None = None, height: int = 380, legend: bool = True) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=TEXT), x=0.01, xanchor="left") if title else None,
        height=height, margin=dict(l=10, r=10, t=48 if title else 16, b=10),
        paper_bgcolor="white", plot_bgcolor="white", font=dict(family="Inter, Segoe UI, sans-serif", color=TEXT, size=12),
        colorway=PALETTE, showlegend=legend, hoverlabel=dict(bgcolor="white", font_size=12, bordercolor=BORDER),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=False, linecolor=BORDER, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor="#EEF2F7", zeroline=False, linecolor=BORDER, tickfont=dict(color=MUTED))
    return fig


def histogram(df: pd.DataFrame, col: str, bins: int = 40, title: str | None = None) -> go.Figure:
    x = pd.to_numeric(df[col], errors="coerce").astype(float).dropna()
    fig = go.Figure(go.Histogram(x=x, nbinsx=bins, marker=dict(color=BLUE, line=dict(color="white", width=0.5)), opacity=0.9,
                                 hovertemplate="Khoảng: %{x}<br>Tần số: %{y}<extra></extra>"))
    if len(x):
        fig.add_vline(x=x.mean(), line=dict(color=TEAL, dash="dash"), annotation_text=f"Mean {x.mean():,.3g}", annotation_position="top right")
        fig.add_vline(x=x.median(), line=dict(color=NAVY, dash="dot"), annotation_text=f"Median {x.median():,.3g}", annotation_position="top left")
    fig.update_layout(bargap=0.04)
    return style(fig, title or f"Phân phối {col}", legend=False)


def box(df: pd.DataFrame, cols: list[str], title: str = "Phân phối (Boxplot)", normalize: bool = False) -> go.Figure:
    fig = go.Figure()
    for i, c in enumerate(cols):
        x = pd.to_numeric(df[c], errors="coerce").astype(float).dropna()
        if normalize and x.std() > 0:
            x = (x - x.mean()) / x.std()
        fig.add_trace(go.Box(y=x, name=c, marker_color=PALETTE[i % len(PALETTE)], boxmean=True))
    return style(fig, title + (" - chuẩn hóa z-score" if normalize else ""), legend=False)


def box_by_group(df: pd.DataFrame, value: str, group: str, top: int = 12, title: str | None = None) -> go.Figure:
    keep = df[group].value_counts().head(top).index
    d = df[df[group].isin(keep)]
    order = d.groupby(group)[value].median().sort_values(ascending=False).index.tolist()
    fig = px.box(d, x=group, y=value, color=group, category_orders={group: order}, color_discrete_sequence=PALETTE)
    return style(fig, title or f"{value} theo {group}", legend=False, height=420)


def bar(df: pd.DataFrame, x: str, y: str, title: str | None = None, horizontal: bool = False, color: str | None = None,
        height: int = 380, text: bool = True) -> go.Figure:
    if horizontal:
        d = df.iloc[::-1]
        fig = px.bar(d, x=y, y=x, orientation="h", color=color, color_discrete_sequence=PALETTE, text=y if text else None)
    else:
        fig = px.bar(df, x=x, y=y, color=color, color_discrete_sequence=PALETTE, text=y if text else None)
    if not color:
        fig.update_traces(marker_color=BLUE)
    fig.update_traces(texttemplate="%{text:,.4~g}" if text else None, textposition="outside", cliponaxis=False)
    return style(fig, title, height=height, legend=bool(color))


def line(df: pd.DataFrame, x: str, ys: list[str] | str, title: str | None = None, markers: bool = True, height: int = 380) -> go.Figure:
    ys = [ys] if isinstance(ys, str) else ys
    fig = go.Figure()
    for i, y in enumerate(ys):
        fig.add_trace(go.Scatter(x=df[x], y=df[y], mode="lines+markers" if markers else "lines", name=str(y),
                                 line=dict(color=PALETTE[i % len(PALETTE)], width=2.4), marker=dict(size=7)))
    return style(fig, title, height=height, legend=len(ys) > 1)


def heatmap(mat: pd.DataFrame, title: str = "Ma trận tương quan", diverging: bool = True, fmt: str = ".2f",
            height: int | None = None) -> go.Figure:
    n = len(mat)
    fig = go.Figure(go.Heatmap(z=mat.values, x=[str(c) for c in mat.columns], y=[str(i) for i in mat.index],
                               colorscale=DIVERGING if diverging else SEQUENTIAL, zmid=0 if diverging else None,
                               zmin=-1 if diverging else None, zmax=1 if diverging else None,
                               text=mat.values, texttemplate="%{text:" + fmt + "}" if n <= 16 else None,
                               hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>", colorbar=dict(thickness=12)))
    fig.update_yaxes(autorange="reversed")
    return style(fig, title, height=height or max(360, 34 * n + 120), legend=False)


def scatter(df: pd.DataFrame, x: str, y: str, color: str | None = None, title: str | None = None, trend: bool = True,
            max_points: int = 5000) -> go.Figure:
    d = df[[c for c in {x, y, color} if c]].dropna(subset=[x, y])
    if len(d) > max_points:
        d = d.sample(max_points, random_state=42)
    fig = px.scatter(d, x=x, y=y, color=color, opacity=0.65, color_discrete_sequence=PALETTE)
    if trend and len(d) > 2 and pd.api.types.is_numeric_dtype(d[x]) and pd.api.types.is_numeric_dtype(d[y]):
        b, a = np.polyfit(d[x].astype(float), d[y].astype(float), 1)
        xs = np.linspace(d[x].min(), d[x].max(), 50)
        r = np.corrcoef(d[x].astype(float), d[y].astype(float))[0, 1]
        fig.add_trace(go.Scatter(x=xs, y=a + b * xs, mode="lines", name=f"OLS (r = {r:.3f})", line=dict(color=DANGER, width=2)))
    return style(fig, title or f"{y} theo {x}", height=420)


def pie(labels, values, title: str | None = None, height: int = 360) -> go.Figure:
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.55, marker=dict(colors=PALETTE), textinfo="percent",
                           hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>"))
    return style(fig, title, height=height)


def elbow_chart(tab: pd.DataFrame, suggested: int | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tab["k"], y=tab["Inertia (WCSS)"], name="Inertia (WCSS)", mode="lines+markers",
                             line=dict(color=BLUE, width=2.4)))
    fig.add_trace(go.Scatter(x=tab["k"], y=tab["Silhouette"], name="Silhouette", mode="lines+markers", yaxis="y2",
                             line=dict(color=TEAL, width=2.4, dash="dot")))
    if suggested:
        fig.add_vline(x=suggested, line=dict(color=NAVY, dash="dash"), annotation_text=f"k gợi ý = {suggested}")
    fig.update_layout(yaxis=dict(title="Inertia"), yaxis2=dict(title="Silhouette", overlaying="y", side="right", showgrid=False),
                      xaxis=dict(title="Số cụm k", dtick=1))
    return style(fig, "Elbow Method & Silhouette Score", height=380)


def pca_scatter(proj: pd.DataFrame, color_col: str, title: str, ev: list[float] | None = None, max_points: int = 6000,
                color_map: dict | None = None) -> go.Figure:
    d = proj.sample(max_points, random_state=42) if len(proj) > max_points else proj
    fig = px.scatter(d, x="PC1", y="PC2", color=color_col, opacity=0.7, color_discrete_sequence=PALETTE,
                     color_discrete_map=color_map)
    if ev:
        fig.update_xaxes(title=f"PC1 ({ev[0] * 100:.1f}% phương sai)")
        fig.update_yaxes(title=f"PC2 ({ev[1] * 100:.1f}% phương sai)")
    fig.update_traces(marker=dict(size=6))
    return style(fig, title, height=440)


def confusion(cm: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=cm.values, x=[f"Dự đoán: {c}" for c in cm.columns], y=[f"Thực tế: {i}" for i in cm.index],
                               colorscale=SEQUENTIAL, text=cm.values, texttemplate="%{text}", showscale=False))
    fig.update_yaxes(autorange="reversed")
    return style(fig, "Confusion Matrix (tập test)", height=max(320, 60 * len(cm) + 120), legend=False)


def roc_curve_fig(y_true, y_score, pos_label) -> go.Figure:
    from sklearn.metrics import auc, roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {auc(fpr, tpr):.3f})", line=dict(color=BLUE, width=2.5)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ngẫu nhiên", line=dict(color=MUTED, dash="dash")))
    fig.update_xaxes(title="False Positive Rate")
    fig.update_yaxes(title="True Positive Rate")
    return style(fig, "ROC Curve", height=380)


def actual_vs_pred(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(df, x="Thực tế", y="Dự đoán", opacity=0.55, color_discrete_sequence=[BLUE])
    lo, hi = float(np.nanmin(df.values)), float(np.nanmax(df.values))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y = x", line=dict(color=TEAL, dash="dash")))
    return style(fig, "Giá trị thực tế và dự đoán (tập test)", height=400)


def forecast_chart(res: dict, models: list[str]) -> go.Figure:
    s, train, test = res["series"], res["train"], res["test"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train.values, name="Train (thực tế)", mode="lines+markers", line=dict(color=NAVY, width=2.4)))
    if "validation" in res:
        validation = res["validation"]
        fig.add_trace(go.Scatter(x=validation.index, y=validation.values, name="Validation (thực tế)",
                                 mode="lines+markers", line=dict(color=MUTED, width=2)))
    fig.add_trace(go.Scatter(x=test.index, y=test.values, name="Test (thực tế)", mode="lines+markers",
                             line=dict(color=TEAL, width=2.4), marker=dict(size=8, symbol="diamond")))
    for i, key in enumerate(models):
        r = res["results"].get(key)
        if not r:
            continue
        c = PALETTE[(i + 3) % len(PALETTE)] if key != "naive" else MUTED
        fig.add_trace(go.Scatter(x=test.index, y=r["test_pred"], name=f"{r['label']} - dự đoán test", mode="lines",
                                 line=dict(color=c, dash="dot", width=2)))
        fx = list(res["future_index"])
        fig.add_trace(go.Scatter(x=[s.index[-1]] + fx, y=[s.values[-1]] + list(r["future"]), name=f"{r['label']} - dự báo",
                                 mode="lines+markers", line=dict(color=c, width=2.4)))
        if key == res["best"]:
            fig.add_trace(go.Scatter(x=fx + fx[::-1], y=list(r["future_hi"]) + list(r["future_lo"])[::-1], fill="toself",
                                     fillcolor="rgba(20,184,166,0.15)", line=dict(width=0), name="Khoảng dự báo 95%", hoverinfo="skip"))
    fig.add_vrect(x0=test.index[0], x1=test.index[-1], fillcolor="rgba(29,78,216,0.05)", line_width=0,
                  annotation_text="Vùng kiểm định", annotation_position="top left")
    return style(fig, f"Thực tế và dự báo: {s.name}", height=460)
