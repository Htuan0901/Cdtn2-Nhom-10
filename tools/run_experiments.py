"""Chạy lại thực nghiệm cho Bảng 4.2 (dự báo) và Bảng 4.3 (lương) + xu hướng lương, KHÔNG chọn mô hình trên test.

Chạy từ thư mục gốc dự án:
    python tools/run_experiments.py                # ghi vào experiments/ và docs/KET_QUA_THUC_NGHIEM_4_2_4_3.md
    python tools/run_experiments.py --out tmp_exp  # ghi vào thư mục khác (không ghi đè kết quả cũ)
    python tools/run_experiments.py --allow-missing-statsmodels   # chạy thử khi máy chưa cài statsmodels

Đầu ra (trong --out):
    bang_4_2_du_bao.csv           kết quả test của 9 mô hình, xếp theo RMSE validation, cột "Được chọn"
    bang_4_2_validation.csv       RMSE/MAE trung bình rolling-origin trên validation (tiêu chí chọn)
    bang_4_2_validation_tung_moc.csv   chi tiết từng mốc rolling-origin
    du_bao_test_chi_tiet.csv      giá trị dự báo từng tháng test
    bang_4_3_luong.csv            Baseline / Ridge / Random Forest / Gradient Boosting
    xu_huong_luong_hn_theo_nam.csv, xu_huong_luong_thay_doi.csv, xu_huong_luong_linkedin.csv
    hinh_xu_huong_luong.png, hinh_du_bao_test.png   hình chèn vào báo cáo
    experiment_config.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src import forecast as fc  # noqa: E402
from src.pipeline import load_processed  # noqa: E402
from src.salary_model import OVERALL_LABEL, salary_trend_change, salary_trend_table, train_salary_models  # noqa: E402

TEST_MONTHS = 6
TREND_ROLES = ["Software Engineer", "Backend / Full-stack", "Frontend / Web", "DevOps / Cloud / SRE"]
TREND_MIN_POSTS = 10


def _has_statsmodels() -> bool:
    try:
        import statsmodels  # noqa: F401
        return True
    except ImportError:
        return False


def _num(x, nd=2):
    return f"{float(x):,.{nd}f}" if pd.notna(x) else "—"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="experiments", help="thư mục ghi kết quả (tương đối với thư mục dự án)")
    ap.add_argument("--doc", default="docs/KET_QUA_THUC_NGHIEM_4_2_4_3.md", help="file Markdown tổng hợp kết quả")
    ap.add_argument("--allow-missing-statsmodels", action="store_true",
                    help="vẫn chạy khi thiếu statsmodels (Holt, Holt-Winters, ARIMA, SARIMA sẽ bị bỏ qua)")
    args = ap.parse_args()

    if not _has_statsmodels() and not args.allow_missing_statsmodels:
        sys.exit("Chưa cài statsmodels nên không chạy được Holt, Holt-Winters, ARIMA, SARIMA.\n"
                 "Cài bằng: pip install -r requirements.txt  (hoặc thêm --allow-missing-statsmodels để chạy thử)")

    data = load_processed()
    if data is None:
        raise FileNotFoundError("Thiếu data/processed/linkedin_it.csv.gz hoặc hn_jobs.csv.gz")
    linkedin, hn, _ = data
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ Bảng 4.2
    print("[1/4] Dự báo số tin: rolling-origin trên validation → chốt mô hình → đánh giá test...", flush=True)
    series = fc.monthly_series(hn)
    r = fc.evaluate_models(series, h=TEST_MONTHS, models=list(fc.MODELS))
    fc_table = r["test_table"].copy()
    fc_table.to_csv(out / "bang_4_2_du_bao.csv", index=False, encoding="utf-8-sig")
    r["validation_table"].to_csv(out / "bang_4_2_validation.csv", index=False, encoding="utf-8-sig")
    r["validation_detail"].to_csv(out / "bang_4_2_validation_tung_moc.csv", index=False, encoding="utf-8-sig")
    r["test_predictions"].to_csv(out / "du_bao_test_chi_tiet.csv", encoding="utf-8-sig")
    print("   Mô hình được chọn:", r["best"], "–", r["selection_rule"], flush=True)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(series.index, series.values, color="#1b1f29", lw=1.6, label="Thực tế")
    tp = r["test_predictions"]
    for name in dict.fromkeys([r["best"], "Naive (baseline)"]):
        if name in tp:
            ax.plot(tp.index, tp[name], lw=2.6 if name == r["best"] else 1.6,
                    ls="-" if name == r["best"] else "--", label=f"Dự báo test – {name}")
    half = pd.Timedelta(days=15)
    ax.axvspan(r["validation"].index[0] - half, r["validation"].index[-1] + half, color="#f6c87a", alpha=0.25, label="Validation (chọn mô hình)")
    ax.axvspan(r["test"].index[0] - half, r["test"].index[-1] + half, color="#9cc3f0", alpha=0.3, label="Test")
    ax.set_title("Số tin tuyển dụng mỗi tháng (Hacker News) – validation, test và dự báo")
    ax.set_ylabel("Số tin")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "hinh_du_bao_test.png", dpi=180)
    plt.close(fig)

    # ------------------------------------------------------------------ Bảng 4.3
    print("[2/4] Huấn luyện mô hình lương (Baseline / Ridge / Random Forest / Gradient Boosting)...", flush=True)
    salary = train_salary_models(linkedin, seed=42)
    salary_table = salary["results"].drop(columns=["Thời gian (s)"]).copy()
    salary_table["Được chọn"] = salary_table["Mô hình"].eq(salary["best"])
    salary_table.to_csv(out / "bang_4_3_luong.csv", index=False, encoding="utf-8-sig")
    print("   Mô hình lương được chọn:", salary["best"], flush=True)

    # ------------------------------------------------------------------ Xu hướng lương
    print("[3/4] Xu hướng lương trung vị theo năm (Hacker News 2018–2023)...", flush=True)
    hn_sal = hn[hn["salary_usd"].notna()]
    trend = salary_trend_table(hn_sal, "YS", TREND_MIN_POSTS, TREND_ROLES, date_col="month", include_overall=True)
    change = salary_trend_change(trend)
    trend.to_csv(out / "xu_huong_luong_hn_theo_nam.csv", index=False, encoding="utf-8-sig")
    change.to_csv(out / "xu_huong_luong_thay_doi.csv", index=False, encoding="utf-8-sig")
    salary_trend_table(linkedin, "MS", 5).to_csv(out / "xu_huong_luong_linkedin.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9, 4.6))
    for role, g in trend.groupby("Vị trí"):
        overall = role == OVERALL_LABEL
        ax.plot(g["Kỳ"].dt.year, g["Lương trung vị (USD/năm)"] / 1000, marker="o",
                lw=3 if overall else 1.6, ls="--" if overall else "-", color="#1b1f29" if overall else None, label=role)
    ax.set_title(f"Lương trung vị theo năm và nhóm vị trí – Hacker News (nhóm × năm ≥ {TREND_MIN_POSTS} tin)")
    ax.set_ylabel("Nghìn USD/năm")
    ax.set_xlabel("Năm (2023: 6 tháng đầu)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "hinh_xu_huong_luong.png", dpi=180)
    plt.close(fig)

    # ------------------------------------------------------------------ Tổng hợp
    print("[4/4] Ghi experiment_config.json và file Markdown tổng hợp...", flush=True)
    cfg = {
        "statsmodels_installed": _has_statsmodels(),
        "dataset": {"hn_rows": int(len(hn)), "hn_first_month": str(series.index.min().date()),
                    "hn_last_month": str(series.index.max().date()), "linkedin_rows": int(len(linkedin)),
                    "linkedin_salary_rows": int(linkedin["salary_usd"].notna().sum()),
                    "hn_salary_rows": int(len(hn_sal))},
        "forecast": {"selected_by": "RMSE trung bình rolling-origin trên validation + quy tắc tiết kiệm",
                     "naive_margin": fc.NAIVE_MARGIN, "best": r["best"], "selection_rule": r["selection_rule"],
                     "n_origins": r["n_origins"], "origin_step_months": r["origin_step"],
                     "train_months_first_origin": int(len(r["train"])),
                     "validation_months": int(len(r["validation"])), "test_months": int(len(r["test"])),
                     "validation_start": str(r["validation"].index.min().date()),
                     "validation_end": str(r["validation"].index.max().date()),
                     "test_start": str(r["test"].index.min().date()), "test_end": str(r["test"].index.max().date()),
                     "skipped_validation": r["validation_skipped"], "skipped_test": r["test_skipped"]},
        "salary": {"selected_by": "validation MAE", "best": salary["best"], "n_train": salary["n_train"],
                   "n_validation": salary["n_validation"], "n_test": salary["n_test"], "seed": 42},
    }
    (out / "experiment_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    v = r["validation"]
    m = [
        "# KẾT QUẢ THỰC NGHIỆM – ITJobTrend (Bảng 4.2, 4.3 và xu hướng lương)", "",
        "*Sinh tự động bởi `tools/run_experiments.py` từ dữ liệu trong `data/processed/`.*", "",
    ]
    if not _has_statsmodels():
        m += ["> **CẢNH BÁO:** máy chạy chưa cài statsmodels nên Holt, Holt-Winters, ARIMA, SARIMA bị bỏ qua. "
              "KHÔNG đưa bảng này vào báo cáo; cài statsmodels rồi chạy lại.", ""]
    m += [
        "## 1. Quy trình đánh giá dự báo", "",
        f"- Hacker News: {len(hn):,} tin, {len(series)} tháng ({series.index.min():%m/%Y}–{series.index.max():%m/%Y}).",
        f"- Validation {v.index[0]:%m/%Y}–{v.index[-1]:%m/%Y}: rolling-origin {r['n_origins']} mốc × {r['origin_step']} tháng. "
        "Ở mỗi mốc, mô hình học toàn bộ dữ liệu trước mốc rồi dự báo các tháng tiếp theo; tiêu chí chọn là RMSE trung bình.",
        f"- Quy tắc tiết kiệm: mô hình khác Naive phải giảm RMSE validation ≥ {fc.NAIVE_MARGIN * 100:.0f}% so với Naive mới được chọn.",
        f"- Test {r['test'].index[0]:%m/%Y}–{r['test'].index[-1]:%m/%Y} ({len(r['test'])} tháng): mọi mô hình học lại trên train+validation, "
        "dự báo một lần, chỉ để báo cáo. Không chọn lại theo test.",
        "- Bậc ARIMA chọn theo AIC trên dữ liệu huấn luyện của từng lượt, không dùng điểm test.", "",
        f"**Mô hình được chọn:** {r['best']} – {r['selection_rule']}.", "",
        "## Bảng 4.2 – So sánh mô hình dự báo số tin tuyển dụng", "",
        "| Mô hình | RMSE validation (TB) | MAE test | RMSE test | Cải thiện RMSE test so với Naive |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in fc_table.iterrows():
        tag = " **(được chọn)**" if row["Được chọn"] else ""
        m.append(f"| {row['Mô hình']}{tag} | {_num(row['Validation RMSE'])} | {_num(row['MAE'])} | {_num(row['RMSE'])} | "
                 f"{_num(row.get('Cải thiện RMSE so với baseline (%)'), 1)}% |")
    if r["validation_skipped"]:
        m += ["", "Không xét: " + "; ".join(f"{k} ({w})" for k, w in r["validation_skipped"].items())]
    m += ["", "## Bảng 4.3 – So sánh mô hình dự đoán lương (LinkedIn)", "",
          f"{int(linkedin['salary_usd'].notna().sum()):,} tin có lương. Chia ngẫu nhiên cố định (seed 42): "
          f"train {salary['n_train']}, validation {salary['n_validation']}, test {salary['n_test']}. "
          "Chọn theo MAE validation; mô hình cuối học lại trên train+validation rồi đánh giá test.", "",
          "| Mô hình | MAE validation (USD) | MAE test (USD) | RMSE test (USD) | R² test | Giảm MAE so với baseline |",
          "|---|---:|---:|---:|---:|---:|"]
    for _, row in salary_table.iterrows():
        tag = " **(được chọn)**" if row["Được chọn"] else ""
        m.append(f"| {row['Mô hình']}{tag} | {_num(row['Validation MAE (USD)'], 0)} | {_num(row['MAE (USD)'], 0)} | "
                 f"{_num(row['RMSE (USD)'], 0)} | {float(row['R²']):.3f} | {_num(row['Giảm MAE so với baseline (%)'], 1)}% |")
    m += ["", "## Xu hướng mức lương (Hacker News, theo năm)", "",
          f"{len(hn_sal):,} tin có ghi khoảng lương; lương = trung điểm khoảng. Chỉ giữ nhóm × năm có ≥ {TREND_MIN_POSTS} tin. "
          "Năm 2023 chỉ có 6 tháng đầu.", "",
          "| Nhóm vị trí | Kỳ đầu | Lương kỳ đầu (USD) | Kỳ cuối | Lương kỳ cuối (USD) | Thay đổi | Số tin |",
          "|---|---|---:|---|---:|---:|---:|"]
    for _, row in change.iterrows():
        m.append(f"| {row['Vị trí']} | {row['Kỳ đầu']:%Y} | {_num(row['Lương kỳ đầu'], 0)} | {row['Kỳ cuối']:%Y} | "
                 f"{_num(row['Lương kỳ cuối'], 0)} | {row['Thay đổi (%)']:+.1f}% | {int(row['Tổng số tin'])} |")
    m += ["", "LinkedIn chỉ có lương trong 01–04/2024 (gần hết ở 04/2024) nên chỉ dùng cho mô hình lương, không dùng cho xu hướng.",
          "", "## Hình cho báo cáo", "",
          f"- `{args.out}/hinh_du_bao_test.png` – chuỗi số tin, vùng validation/test và dự báo của mô hình được chọn.",
          f"- `{args.out}/hinh_xu_huong_luong.png` – lương trung vị theo năm và nhóm vị trí.", "",
          "## Tái lập", "", "```bash", "pip install -r requirements.txt", "python tools/run_experiments.py", "pytest -q", "```", ""]
    doc = ROOT / args.doc
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("\n".join(m), encoding="utf-8")
    print(f"Đã lưu kết quả vào {out} và {doc}", flush=True)


if __name__ == "__main__":
    main()
