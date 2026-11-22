"""Chạy lại Bảng 4.2 / 4.3 bằng dữ liệu mẫu hiện có, không chọn mô hình trên test.

Run from project root: python tools/run_experiments.py
Output: experiments/bang_4_2_du_bao.csv, bang_4_3_luong.csv,
        experiments/xu_huong_luong.csv, docs/KET_QUA_THUC_NGHIEM_4_2_4_3.md
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src import forecast as fc
from src.pipeline import load_processed
from src.salary_model import salary_trend_table, train_salary_models


def main() -> None:
    data = load_processed()
    if data is None:
        raise FileNotFoundError("Thiếu data/processed/linkedin_it.csv.gz hoặc hn_jobs.csv.gz")
    linkedin, hn, _ = data
    out = ROOT / "experiments"
    out.mkdir(exist_ok=True)
    print("[1/3] Dự báo số tin tuyển dụng: train / validation / test theo thời gian...", flush=True)
    series = fc.monthly_series(hn)
    names = ["Naive (baseline)", "Trung bình trượt 3T", "Holt (damped)",
             "ARIMA (chọn theo AIC)", "SARIMA (mùa vụ 12)"]
    r = fc.evaluate_models(series, h=6, validation_size=6, models=names)
    fc_table = r["test_table"].copy()
    fc_table["Được chọn từ validation"] = fc_table["Mô hình"].eq(r["best"])
    fc_table.to_csv(out / "bang_4_2_du_bao.csv", index=False, encoding="utf-8-sig")
    r["validation_table"].to_csv(out / "bang_4_2_validation.csv", index=False, encoding="utf-8-sig")
    r["test_predictions"].to_csv(out / "du_bao_test_chi_tiet.csv", encoding="utf-8-sig")
    print("Selected forecasting:", r["best"], flush=True)

    print("[2/3] Huấn luyện Ridge / Random Forest / Gradient Boosting...", flush=True)
    salary = train_salary_models(linkedin, seed=42)
    salary_table = salary["results"].drop(columns=["Thời gian (s)"]).copy()
    salary_table["Được chọn từ validation"] = salary_table["Mô hình"].eq(salary["best"])
    salary_table.to_csv(out / "bang_4_3_luong.csv", index=False, encoding="utf-8-sig")
    print("Selected salary:", salary["best"], flush=True)

    print("[3/3] Xuất dữ liệu lương trung vị theo thời gian...", flush=True)
    trend = salary_trend_table(linkedin, "MS", min_posts=5)
    trend.to_csv(out / "xu_huong_luong.csv", index=False, encoding="utf-8-sig")
    out_json = {
        "dataset": {"hn_rows": int(len(hn)), "hn_first_month": str(series.index.min().date()),
                    "hn_last_month": str(series.index.max().date()),
                    "linkedin_rows": int(len(linkedin)),
                    "linkedin_salary_rows": int(linkedin["salary_usd"].notna().sum())},
        "forecast": {"selected_by": "validation RMSE", "best": r["best"],
                     "train_months": int(len(r["train"])), "validation_months": int(len(r["validation"])),
                     "test_months": int(len(r["test"])),
                     "train_end": str(r["train"].index.max().date()),
                     "validation_start": str(r["validation"].index.min().date()),
                     "validation_end": str(r["validation"].index.max().date()),
                     "test_start": str(r["test"].index.min().date()),
                     "test_end": str(r["test"].index.max().date()),
                     "skipped_validation": r["validation_skipped"], "skipped_test": r["test_skipped"]},
        "salary": {"selected_by": "validation MAE", "best": salary["best"],
                   "n_train": salary["n_train"], "n_validation": salary["n_validation"],
                   "n_test": salary["n_test"], "seed": 42}
    }
    (out / "experiment_config.json").write_text(json.dumps(out_json, indent=2, ensure_ascii=False), encoding="utf-8")

    num = lambda x: f"{float(x):,.2f}" if pd.notna(x) else "—"
    m = [
        "# KẾT QUẢ THỰC NGHIỆM – ITJobTrend (Bảng 4.2 và 4.3)",
        "", "*Kết quả tính trực tiếp từ dữ liệu có trong `data/processed/`, không phải số liệu giả định.*", "",
        "## 1. Quy trình đánh giá", "",
        f"- Hacker News: {len(hn):,} tin, {len(series)} tháng ({series.index.min():%m/%Y}–{series.index.max():%m/%Y}).",
        f"- Train: {r['train'].index.min():%m/%Y}–{r['train'].index.max():%m/%Y} ({len(r['train'])} kỳ).",
        f"- Validation: {r['validation'].index.min():%m/%Y}–{r['validation'].index.max():%m/%Y} ({len(r['validation'])} kỳ), chọn mô hình theo RMSE.",
        f"- Test: {r['test'].index.min():%m/%Y}–{r['test'].index.max():%m/%Y} ({len(r['test'])} kỳ), chỉ đánh giá sau khi chốt model.",
        "- Tham số ARIMA/SARIMA được xác định bằng AIC **trên tập huấn luyện của mỗi lượt chạy**, không sử dụng điểm test.",
        "- Sau khi chọn theo validation, các mô hình fit lại trên train+validation để báo cáo MAE/RMSE ở test.",
        "", "## Bảng 4.2 – So sánh mô hình dự báo số lượng tin tuyển dụng (test)", "",
        "| Mô hình | Validation RMSE | MAE test | RMSE test | Cải thiện RMSE so với Naive |", "|---|---:|---:|---:|---:|",
    ]
    for _, row in fc_table.iterrows():
        m.append(f"| {row['Mô hình']} {'**(chọn)**' if row['Được chọn từ validation'] else ''} | {num(row['Validation RMSE'])} | {num(row['MAE'])} | {num(row['RMSE'])} | {num(row.get('Cải thiện RMSE so với baseline (%)'))}% |")
    m += ["", f"**Mô hình chọn bằng validation:** {r['best']}. *Không chọn lại bằng điểm test.*", "",
          "## Bảng 4.3 – So sánh mô hình dự đoán lương (test)", "",
          f"Dữ liệu LinkedIn: {len(linkedin):,} tin CNTT; {linkedin['salary_usd'].notna().sum():,} tin có công bố lương.",
          f"Chia ngẫu nhiên nhưng cố định `random_state=42`: train {salary['n_train']}, validation {salary['n_validation']}, test {salary['n_test']}. ",
          "Chọn mô hình hồi quy theo MAE của validation; test không tham gia lựa chọn.", "",
          "| Mô hình | MAE validation (USD) | MAE test (USD) | RMSE test (USD) | R² test |", "|---|---:|---:|---:|---:|",
          ]
    for _, row in salary_table.iterrows():
        m.append(f"| {row['Mô hình']} {'**(chọn)**' if row['Được chọn từ validation'] else ''} | {num(row['Validation MAE (USD)'])} | {num(row['MAE (USD)'])} | {num(row['RMSE (USD)'])} | {float(row['R²']):.4f} |")
    m += ["", f"**Mô hình lương chọn bằng validation:** {salary['best']}.", "",
          "## Giới hạn dữ liệu – Xu hướng mức lương", "",
          f"- LinkedIn có lương trong {linkedin['listed_date'].min():%m/%Y}–{linkedin['listed_date'].max():%m/%Y}.",
          "- Gần như toàn bộ lương quan sát tập trung ở **04/2024**, còn 01/2024 chỉ có 1 mẫu; không đủ để khẳng định xu hướng dài hạn.",
          "- Trang 'Dự đoán mức lương' hiển thị trung vị **theo nhóm vị trí × tháng/quý**, kèm **số tin** và ngưỡng mẫu tối thiểu.",
          "- Biểu đồ chỉ phản ánh lương **được công bố trong mẫu LinkedIn**, không phải mức lương thị trường chung.",
          "", "## Tái lập thực nghiệm", "", "```bash", "pip install -r requirements.txt", "python tools/run_experiments.py", "pytest -q", "```", "",
          "Số gốc và bảng chi tiết: `experiments/bang_4_2_du_bao.csv`, `experiments/bang_4_2_validation.csv`, `experiments/bang_4_3_luong.csv`.",
          "Khi thay đổi dữ liệu hoặc tham số, phải chạy lại script; không sao chép số liệu này sang dữ liệu mới.", ""]
    (ROOT / "docs" / "KET_QUA_THUC_NGHIEM_4_2_4_3.md").write_text("\n".join(m), encoding="utf-8")
    print("Saved experiments and docs/KET_QUA_THUC_NGHIEM_4_2_4_3.md", flush=True)


if __name__ == "__main__":
    main()
