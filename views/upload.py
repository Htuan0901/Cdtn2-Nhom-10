import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import RAW_DIR
from src.pipeline import HN_FILE, LI_FILE, clean_hn, clean_linkedin, save_processed
from src.state import get_data, set_data
from src.ui import fmt_int, kpi_cards, page_header

page_header("📥", "Nhập dữ liệu", "Nạp 2 bộ dữ liệu Kaggle và chạy pipeline làm sạch, chuẩn hóa, trích xuất kỹ năng.")

li, hn, log = get_data(stop=False)
if li is not None:
    st.success(f"Đang dùng: **{st.session_state.get('data_source')}**, gồm {fmt_int(len(li))} tin LinkedIn (CNTT) "
               f"và {fmt_int(len(hn))} tin Hacker News.")
else:
    st.warning("Chưa có dữ liệu đã xử lý. Hãy chạy pipeline ở bên dưới.")


def _run(li_src, hn_src, label):
    bar = st.progress(0.0, text="Bắt đầu…")
    t0 = time.time()
    try:
        li_df, log1 = clean_linkedin(li_src, progress=lambda p, m: bar.progress(p * 0.8, text=m))
        hn_df, log2 = clean_hn(hn_src, progress=lambda p, m: bar.progress(min(0.8 + (p - 0.8), 0.99), text=m))
    except Exception as e:  # noqa: BLE001
        bar.empty()
        st.error(f"Không xử lý được dữ liệu: {e}")
        return
    bar.progress(1.0, text="Đang lưu…")
    save_processed(li_df, hn_df, log1 + log2)
    set_data(li_df, hn_df, pd.DataFrame(log1 + log2), label)
    st.cache_data.clear()
    bar.empty()
    st.success(f"Xong sau {time.time() - t0:.0f} giây: {fmt_int(len(li_df))} tin LinkedIn (CNTT), {fmt_int(len(hn_df))} tin HN. "
               "Kết quả đã lưu vào data/processed để lần sau mở app là có ngay.")


tab_path, tab_up = st.tabs(["📂 Đọc từ đường dẫn trên máy (khuyên dùng)", "⬆️ Tải file zip lên"])
with tab_path:
    st.markdown("Nhập đường dẫn tới **file zip tải từ Kaggle** hoặc **thư mục đã giải nén**. "
                f"Mặc định app tìm trong `{RAW_DIR.relative_to(RAW_DIR.parent.parent)}/linkedin` và `.../hn`.")
    c1, c2 = st.columns(2)
    p_li = c1.text_input("LinkedIn Job Postings (chứa postings.csv)", value=str(RAW_DIR / "linkedin"),
                         help=r"Ví dụ: C:\Users\ban\Downloads\archive.zip")
    p_hn = c2.text_input("Hacker News Who is hiring (chứa jobs_2018_2023.csv)", value=str(RAW_DIR / "hn"),
                         help=r"Ví dụ: C:\Users\ban\Downloads\archive (1).zip")
    if st.button("⚙️ Chạy pipeline", type="primary"):
        a, b = Path(p_li.strip().strip('"')), Path(p_hn.strip().strip('"'))
        if not a.exists() or not b.exists():
            st.error("Không tìm thấy đường dẫn. Kiểm tra lại (có thể bấm chuột phải vào file → Copy as path).")
        else:
            with st.spinner("Đang xử lý (LinkedIn ~124 nghìn tin nên mất khoảng 3–6 phút)…"):
                _run(a, b, f"Pipeline từ {a.name} + {b.name}")

with tab_up:
    st.markdown("Tải lên 2 file zip gốc từ Kaggle (LinkedIn ~166 MB, Hacker News ~30 MB).")
    c1, c2 = st.columns(2)
    u_li = c1.file_uploader("archive.zip – LinkedIn", type=["zip"], key="u_li")
    u_hn = c2.file_uploader("archive (1).zip – Hacker News", type=["zip"], key="u_hn")
    if st.button("⚙️ Chạy pipeline với file tải lên", type="primary", disabled=not (u_li and u_hn)):
        with st.spinner("Đang xử lý…"):
            _run(u_li, u_hn, f"Tải lên: {u_li.name} + {u_hn.name}")

li, hn, log = get_data(stop=False)
if li is not None:
    st.markdown("#### 👀 Xem nhanh dữ liệu đã xử lý")
    kpi_cards([
        ("LinkedIn – số tin CNTT", fmt_int(len(li))),
        ("LinkedIn – số công ty", fmt_int(li["company_name"].nunique())),
        ("HN – số tin", fmt_int(len(hn))),
        ("HN – số công ty", fmt_int(hn["company_name"].nunique())),
        ("Dung lượng file xử lý", f"{(LI_FILE.stat().st_size + HN_FILE.stat().st_size) / 1024**2:.1f} MB"
         if LI_FILE.exists() else "–"),
    ])
    t1, t2 = st.tabs(["LinkedIn (CNTT)", "Hacker News"])
    with t1:
        st.dataframe(li.head(200), width="stretch", height=360)
        st.caption("Các cột: " + ", ".join(li.columns))
    with t2:
        st.dataframe(hn.head(200), width="stretch", height=360)
        st.caption("Các cột: " + ", ".join(hn.columns))
