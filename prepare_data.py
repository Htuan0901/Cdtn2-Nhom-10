"""Chạy pipeline làm sạch từ dữ liệu Kaggle gốc rồi lưu vào data/processed/.

Cách dùng (trong thư mục dự án):
    python prepare_data.py --linkedin "đường_dẫn/archive.zip" --hn "đường_dẫn/archive (1).zip"
Có thể truyền file .zip tải từ Kaggle hoặc thư mục đã giải nén.
Nếu bỏ trống, script tìm trong data/raw/linkedin và data/raw/hn.
"""
import argparse

import pandas as pd

from src.config import RAW_DIR
from src.pipeline import clean_hn, clean_linkedin, save_processed

ap = argparse.ArgumentParser()
ap.add_argument("--linkedin", default=str(RAW_DIR / "linkedin"))
ap.add_argument("--hn", default=str(RAW_DIR / "hn"))
a = ap.parse_args()

print("[1/3] Làm sạch LinkedIn…")
li, log1 = clean_linkedin(a.linkedin)
print("[2/3] Làm sạch Hacker News…")
hn, log2 = clean_hn(a.hn)
print("[3/3] Lưu kết quả…")
save_processed(li, hn, log1 + log2)
print(pd.DataFrame(log1 + log2).to_string(index=False))
print(f"Xong: LinkedIn {len(li):,} tin CNTT, Hacker News {len(hn):,} tin.")
