"""Tạo báo cáo PDF (ReportLab) từ kết quả THỰC TẾ của phiên làm việc. Phần nào chưa chạy sẽ ghi rõ 'chưa thực hiện'."""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

from src.analytics.analysis import eda
from src.analytics.config import APP_NAME, APP_VERSION, COURSE, FONTS_DIR, PROJECT_TITLE_EN, PROJECT_TITLE_VI
from src.analytics.data import schema
from src.analytics.data.cleaning import compare_quality, quality_summary
from src.analytics.reporting import static_charts as sc

NAVY = colors.HexColor("#0B1F3A")
BLUE = colors.HexColor("#1D4ED8")
TEAL = colors.HexColor("#14B8A6")
LIGHT = colors.HexColor("#EEF2F7")

SECTIONS = {
    "dataset": "Thông tin dataset", "overview": "Tổng quan dữ liệu", "cleaning": "Kết quả làm sạch", "eda": "Kết quả EDA",
    "insights": "Những insight quan trọng", "jobs": "Thị trường tuyển dụng & kỹ năng", "mining": "Kết quả Data Mining",
    "ml": "Kết quả Machine Learning & so sánh mô hình", "forecast": "Kết quả dự báo", "anomaly": "Phát hiện bất thường",
    "kaggle": "Kết quả trên dữ liệu Kaggle (dự báo theo tháng, mô hình lương)",
    "conclusion": "Nhận xét và kết luận",
}


def _register_fonts() -> tuple[str, str]:
    candidates = [
        (FONTS_DIR / "DejaVuSans.ttf", FONTS_DIR / "DejaVuSans-Bold.ttf"),
        (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
        (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
    ]
    for reg, bold in candidates:
        if reg.exists() and bold.exists():
            try:
                pdfmetrics.registerFont(TTFont("VN", str(reg)))
                pdfmetrics.registerFont(TTFont("VN-Bold", str(bold)))
                return "VN", "VN-Bold"
            except Exception:
                continue
    return "Helvetica", "Helvetica-Bold"   # dự phòng (không hiển thị đủ dấu tiếng Việt)


def _styles(font: str, bold: str):
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], fontName=bold, fontSize=17, leading=22, textColor=NAVY, alignment=TA_CENTER),
        "sub": ParagraphStyle("s", parent=ss["Normal"], fontName=font, fontSize=10.5, leading=14, alignment=TA_CENTER, textColor=colors.HexColor("#475569")),
        "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontName=bold, fontSize=13.5, leading=18, textColor=NAVY, spaceBefore=10, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName=bold, fontSize=11, leading=15, textColor=BLUE, spaceBefore=6, spaceAfter=4),
        "body": ParagraphStyle("b", parent=ss["Normal"], fontName=font, fontSize=9.5, leading=13.5),
        "small": ParagraphStyle("sm", parent=ss["Normal"], fontName=font, fontSize=8, leading=10.5, textColor=colors.HexColor("#64748B")),
        "cell": ParagraphStyle("c", parent=ss["Normal"], fontName=font, fontSize=7.5, leading=9.5),
        "cellb": ParagraphStyle("cb", parent=ss["Normal"], fontName=bold, fontSize=7.5, leading=9.5, textColor=colors.white),
    }


def _esc(t) -> str:
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("**", "")


def _table(df: pd.DataFrame, st, max_rows: int = 20, col_widths=None) -> Table:
    d = df.head(max_rows).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:,.4g}")
    data = [[Paragraph(_esc(c), st["cellb"]) for c in d.columns]]
    for row in d.astype(str).itertuples(index=False):
        data.append([Paragraph(_esc(v)[:120], st["cell"]) for v in row])
    width = 17.5 * cm
    t = Table(data, colWidths=col_widths or [width / len(d.columns)] * len(d.columns), repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                           ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return t


def _img(png: bytes | None, width_cm: float = 16):
    if not png:
        return None
    img = Image(io.BytesIO(png))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width_cm * cm
    img.drawHeight = width_cm * cm * ratio
    return img


def build_report(ctx: dict, sections: list[str] | None = None, author: str = "") -> bytes:
    """ctx: {df, raw_df, meta, clean_log, time_col, results: {eda, jobs/skills, mining, ml, forecast, anomaly}}"""
    font, bold = _register_fonts()
    st = _styles(font, bold)
    sections = sections or list(SECTIONS)
    df, raw, meta, res = ctx["df"], ctx.get("raw_df"), ctx.get("meta") or {}, ctx.get("results") or {}
    story = []

    def h1(num, key):
        story.append(Paragraph(f"{num}. {SECTIONS[key]}", st["h1"]))

    def para(t):
        story.append(Paragraph(_esc(t), st["body"]))

    def not_done(module):
        story.append(Paragraph(f"<i>Chưa thực hiện trong phiên làm việc này (module {module}).</i>", st["small"]))

    def add_img(png, w=16):
        im = _img(png, w)
        if im:
            story.append(im)
            story.append(Spacer(1, 6))

    # ---------------- Trang bìa
    story += [Spacer(1, 2.2 * cm), Paragraph("BÁO CÁO PHÂN TÍCH DỮ LIỆU", st["sub"]), Spacer(1, 8),
              Paragraph(_esc(PROJECT_TITLE_VI.upper()), st["title"]), Spacer(1, 6), Paragraph(_esc(PROJECT_TITLE_EN), st["sub"]),
              Spacer(1, 1.2 * cm), Paragraph(_esc(COURSE), st["sub"])]
    if author:
        story.append(Paragraph(_esc(author), st["sub"]))
    story += [Spacer(1, 4), Paragraph(f"Hệ thống: {APP_NAME} v{APP_VERSION}", st["sub"]),
              Paragraph(f"Ngày tạo: {datetime.now():%d/%m/%Y %H:%M}", st["sub"]), PageBreak()]

    n = 0
    q = quality_summary(df)
    if "dataset" in sections:
        n += 1
        h1(n, "dataset")
        info = pd.DataFrame([
            ("Tên file", meta.get("name", "")), ("Kích thước", f"{(meta.get('size_bytes') or 0) / 1e6:,.2f} MB"),
            ("Encoding / phân cách", f"{meta.get('encoding')} / {repr(meta.get('delimiter'))}"),
            ("Số dòng × cột lúc tải lên", f"{meta.get('n_rows', 0):,} × {meta.get('n_cols', 0)}"),
            ("Số dòng × cột hiện tại", f"{len(df):,} × {df.shape[1]}"), ("Cột thời gian", ctx.get("time_col") or "Không có"),
        ], columns=["Thuộc tính", "Giá trị"])
        story.append(_table(info, st, col_widths=[5 * cm, 12.5 * cm]))
    if "overview" in sections:
        n += 1
        h1(n, "overview")
        prof = schema.profile_frame(df)[["name", "dtype", "role_label", "missing_pct", "n_unique"]]
        prof.columns = ["Cột", "Kiểu", "Vai trò", "Thiếu (%)", "Số giá trị khác nhau"]
        story.append(_table(prof, st, 40))
        story.append(Spacer(1, 6))
        para(f"Tỷ lệ ô hợp lệ: {q['valid_pct']}%; số dòng trùng lặp (bỏ qua cột ID): {q['duplicates']:,}; số biến số: "
             f"{len(schema.numeric_columns(df))}.")
        add_img(sc.missing_bar(df), 14)
    if "cleaning" in sections:
        n += 1
        h1(n, "cleaning")
        if raw is not None:
            story.append(_table(compare_quality(raw, df), st))
        log = ctx.get("ws_clean_log") or []
        if log:
            lg = pd.DataFrame(log)[["action", "detail", "rows_before", "rows_after"]]
            lg.columns = ["Thao tác", "Chi tiết", "Dòng trước", "Dòng sau"]
            story.append(Spacer(1, 6))
            story.append(_table(lg, st, 25, [4 * cm, 9.5 * cm, 2 * cm, 2 * cm]))
        else:
            not_done("Làm sạch dữ liệu")
    nums = schema.numeric_columns(df)
    if "eda" in sections:
        n += 1
        h1(n, "eda")
        if nums:
            full = eda.descriptive_stats(df, nums)
            stats = full[["Biến", "N", "Mean", "Median", "Std", "Min", "Q1 (25%)", "Q3 (75%)", "Max", "Skewness"]]
            story.append(_table(stats, st, 15))
            for t in eda.describe_numeric_text(full)[:6]:
                para("• " + t)
            add_img(sc.histogram(df, "salary_mid" if "salary_mid" in nums else nums[0]), 15)
            corr = eda.correlation_matrix(df, nums[:12])
            if not corr.empty:
                add_img(sc.corr_heatmap(corr), 12)
                for t in eda.describe_correlation_text(eda.top_correlations(corr, 5)):
                    para("• " + t)
        else:
            para("Dataset không có biến số để thống kê mô tả.")
    if "insights" in sections:
        n += 1
        h1(n, "insights")
        for t in eda.key_findings(df):
            para("• " + t)
    sk = res.get("skills")
    if "jobs" in sections:
        n += 1
        h1(n, "jobs")
        if "salary_mid" in df:
            s = df["salary_mid"].dropna()
            para(f"Số tin phân tích: {len(df):,}; {len(s):,} tin có lương; lương trung vị {s.median():.1f} triệu VND/tháng "
                 f"(IQR {s.quantile(.25):.1f} – {s.quantile(.75):.1f}).")
            if "job_group" in df:
                g = df.dropna(subset=["salary_mid"]).groupby("job_group")["salary_mid"].agg(["count", "median"]).query("count >= 20")
                g = g.sort_values("median", ascending=False).head(12)
                add_img(sc.barh(g.index, g["median"], "Lương trung vị theo nhóm vị trí (triệu VND/tháng)"), 14)
        if sk:
            c = sk["counts"].head(20)
            add_img(sc.barh(c["Kỹ năng"], c["Tỷ lệ tin (%)"], "Top 20 kỹ năng được yêu cầu (% số tin)", color="#14B8A6"), 14)
            if sk.get("share") is not None:
                add_img(sc.skill_trend(sk["share"], c["Kỹ năng"].head(6).tolist()), 15)
            fc = sk.get("forecast")
            if fc and fc.get("ok"):
                para(f"Backtest dự báo tỷ lệ kỹ năng (kỳ {fc['test_period']}): MAE Linear Trend = {fc['mae_linear']} điểm %, "
                     f"MAE Naive = {fc['mae_naive']} điểm % ⇒ mô hình tốt hơn: {fc['better']}.")
                story.append(_table(fc["table"][["Kỹ năng", "Tỷ lệ kỳ cuối (%)", "Độ dốc (điểm %/kỳ)", "Dự báo kỳ tới (%)", "Xu hướng"]], st, 15))
        else:
            not_done("Thị trường tuyển dụng CNTT")
    if "mining" in sections:
        n += 1
        h1(n, "mining")
        m = res.get("mining")
        if m:
            para(f"K-Means với k = {m['k']} trên các biến {', '.join(m['features'])}; Silhouette = {m['silhouette']:.3f}; "
                 f"PCA 2 thành phần giữ {sum(m['explained_variance']) * 100:.1f}% phương sai.")
            add_img(sc.pca_clusters(m["projection"], m["explained_variance"]), 13)
            story.append(_table(m["profile"].reset_index(), st))
            for t in m["descriptions"]:
                para("• " + t)
        else:
            not_done("Khai phá dữ liệu")
    if "ml" in sections:
        n += 1
        h1(n, "ml")
        r = res.get("ml")
        if r:
            para(f"Bài toán: {'hồi quy' if r.problem == 'regression' else 'phân loại'}; biến mục tiêu: {r.target}; "
                 f"{len(r.features)} đặc trưng; train {r.n_train:,} / test {r.n_test:,} dòng.")
            cols = [c for c in r.comparison.columns if c not in ("Thời gian (s)",)]
            story.append(_table(r.comparison[cols], st))
            add_img(sc.model_comparison(r.comparison, "RMSE" if r.problem == "regression" else "F1 (macro)"), 14)
            para(f"Mô hình được chọn: {r.best_name}. {r.best_reason}")
            for t in r.notes:
                para("• " + t)
            imp = r.details[r.best_name]["importance"]
            if not imp.empty:
                add_img(sc.barh(imp["Đặc trưng"].head(12), imp["Mức quan trọng"].head(12), f"Độ quan trọng đặc trưng - {r.best_name}"), 14)
        else:
            not_done("Machine Learning")
    if "forecast" in sections:
        n += 1
        h1(n, "forecast")
        f = res.get("forecast")
        if f:
            story.append(_table(f["table"], st))
            add_img(sc.forecast_plot(f), 15)
            for t in f["explanation"]:
                para("• " + t)
            if f["skipped"]:
                para("Mô hình không chạy: " + "; ".join(f"{k}: {v}" for k, v in f["skipped"].items()))
        else:
            not_done("Dự báo")
    if "anomaly" in sections:
        n += 1
        h1(n, "anomaly")
        a = res.get("anomaly")
        if a:
            from src.analytics.anomaly.detector import describe
            add_img(sc.anomaly_scatter(a["projection"]), 13)
            story.append(_table(a["comparison"].reset_index().rename(columns={"index": "Biến"}), st))
            for t in describe(a):
                para("• " + t)
        else:
            not_done("Phát hiện bất thường")
    if "kaggle" in sections:
        n += 1
        h1(n, "kaggle")
        kg = ctx.get("kaggle") or {}
        plog = kg.get("pipeline_log")
        if plog is not None and len(plog):
            para("Nhật ký pipeline làm sạch dữ liệu Kaggle (LinkedIn + Hacker News):")
            cols = [c for c in ["Nguồn", "Bước", "Trước", "Sau", "Bị loại"] if c in plog.columns]
            story.append(_table(plog[cols], st, 20))
        fcr = kg.get("fc_result")
        if fcr:
            para(f"Dự báo chuỗi '{fcr['target']}' ({fcr['unit']}), kiểm định trên {fcr['h']} tháng cuối. Mô hình tốt nhất: {fcr['best']}.")
            story.append(_table(fcr["results"], st))
        sb = kg.get("salary_bundle")
        if sb:
            para(f"Mô hình dự đoán lương (LinkedIn, USD/năm): tốt nhất {sb['best']}; train {sb['n_train']:,} / test {sb['n_test']:,}.")
            story.append(_table(sb["results"], st))
        if not (fcr or sb or (plog is not None and len(plog))):
            not_done("Xu hướng & dự báo / Dự đoán mức lương")
    if "conclusion" in sections:
        n += 1
        h1(n, "conclusion")
        for t in conclusions(df, res):
            para("• " + t)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.7 * cm, rightMargin=1.7 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                            title=PROJECT_TITLE_VI, author=author or APP_NAME)

    def footer(canvas, d):
        canvas.saveState()
        canvas.setFont(font, 7.5)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawString(1.7 * cm, 1 * cm, f"{APP_NAME} - Báo cáo tự động từ dữ liệu thực tế")
        canvas.drawRightString(A4[0] - 1.7 * cm, 1 * cm, f"Trang {d.page}")
        canvas.setStrokeColor(TEAL)
        canvas.line(1.7 * cm, A4[1] - 1.2 * cm, A4[0] - 1.7 * cm, A4[1] - 1.2 * cm)
        canvas.restoreState()

    doc.build(story, onFirstPage=lambda c, d: None, onLaterPages=footer)
    return buf.getvalue()


def conclusions(df: pd.DataFrame, res: dict) -> list[str]:
    """Kết luận chỉ được rút ra từ kết quả đã tính; không thêm nhận định không có căn cứ."""
    out = []
    sk = res.get("skills")
    if sk and not sk["counts"].empty:
        top = sk["counts"].head(5)
        out.append("Năm kỹ năng/công nghệ được yêu cầu nhiều nhất: " + ", ".join(f"{r['Kỹ năng']} ({r['Tỷ lệ tin (%)']}%)" for _, r in top.iterrows()) + ".")
        fc = sk.get("forecast")
        if fc and fc.get("ok"):
            up = fc["table"].sort_values("Độ dốc (điểm %/kỳ)", ascending=False).head(3)["Kỹ năng"].tolist()
            out.append(f"Các kỹ năng có độ dốc tăng lớn nhất: {', '.join(up)}. Trên kỳ kiểm định, {fc['better']} cho sai số thấp hơn "
                       f"(MAE {fc['mae_linear']} so với {fc['mae_naive']} điểm %), nên dự báo chỉ mang tính định hướng.")
    if "salary_mid" in df and "exp_bucket" in df:
        g = df.groupby("exp_bucket")["salary_mid"].median().dropna()
        if len(g) >= 2:
            out.append("Lương trung vị theo kinh nghiệm: " + ", ".join(f"{k}: {v:.1f} tr" for k, v in g.items()) + ".")
    r = res.get("ml")
    if r:
        out.append(f"Mô hình {r.best_name} tốt nhất trên tập test; {r.best_reason}")
    f = res.get("forecast")
    if f:
        out.append(f["explanation"][1] if len(f["explanation"]) > 1 else f["explanation"][0])
    a = res.get("anomaly")
    if a:
        out.append(f"Isolation Forest đánh dấu {a['n_anomalies']:,} bản ghi bất thường ({a['rate'] * 100:.2f}%), cần kiểm tra thủ công trước khi loại bỏ.")
    out.append("Hạn chế: dữ liệu là ảnh chụp (snapshot) từ các nguồn đăng tin, cơ cấu nguồn thay đổi theo năm; số kỳ thời gian ít "
               "làm giảm độ tin cậy của dự báo; trích xuất kỹ năng dựa trên từ điển nên có thể bỏ sót kỹ năng mới.")
    return out
