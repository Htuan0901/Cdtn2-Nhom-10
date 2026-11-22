"""Chuẩn bị dữ liệu từ file Parquet gốc (606.878 tin) → CSV để tải vào ứng dụng.

Cách dùng (trong thư mục dự án, đã kích hoạt môi trường ảo):
    python tools/prepare_vn_dataset.py duong_dan/data.parquet                 # chỉ giữ tin CNTT → data/vn/it_jobs_full.csv
    python tools/prepare_vn_dataset.py duong_dan/data.parquet --all           # giữ toàn bộ tin
    python tools/prepare_vn_dataset.py duong_dan/data.parquet --sample 20000  # lấy mẫu phân tầng theo năm

Quy trình minh bạch: đọc Parquet → (tùy chọn) lọc tin CNTT bằng cùng bộ luật với ứng dụng → ghi CSV UTF-8 BOM.
Không làm sạch tự động ở bước này: việc làm sạch được thực hiện có kiểm soát trong module Làm sạch dữ liệu.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analytics.domain.jobs import auto_map_columns, is_it_posting  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("--all", action="store_true", help="Giữ toàn bộ tin (không lọc CNTT)")
    ap.add_argument("--sample", type=int, default=0, help="Số dòng lấy mẫu phân tầng theo năm (0 = không lấy mẫu)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    df = pd.read_parquet(a.parquet)
    print(f"Đọc {len(df):,} dòng × {df.shape[1]} cột")
    m = auto_map_columns(df)
    if not a.all:
        pairs = df[[m["title"], m["industry"]]].drop_duplicates() if m.get("industry") else df[[m["title"]]].drop_duplicates()
        pairs["_it"] = [is_it_posting(*row) for row in pairs.itertuples(index=False)]
        df = df.merge(pairs, how="left").query("_it").drop(columns="_it")
        print(f"Giữ {len(df):,} tin CNTT")
    if a.sample and a.sample < len(df):
        tcol = m.get("time")
        frac = a.sample / len(df)
        df = (pd.concat([g.sample(frac=frac, random_state=42) for _, g in df.groupby(tcol)]) if tcol
              else df.sample(a.sample, random_state=42))
        print(f"Lấy mẫu {len(df):,} dòng (random_state = 42)")
    out = Path(a.out) if a.out else ROOT / "data" / "vn" / ("jobs_full.csv" if a.all else "it_jobs_full.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Đã ghi {out} ({out.stat().st_size / 1e6:,.1f} MB)")


if __name__ == "__main__":
    main()
