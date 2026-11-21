# Kiến trúc hệ thống ITJobTrend 2.0

## 1. Tổng quan

Ứng dụng Streamlit nhiều trang, kiến trúc **phân lớp + module hóa**. Hai nhánh xử lý dùng chung giao diện:

```
                    ┌──────────────────────────── Trình duyệt (Streamlit) ────────────────────────────┐
                    │ app.py: st.navigation · 20 trang trong views/ · icon Material · Plotly tương tác │
                    └───────────────┬───────────────────────────────────────────────┬──────────────────┘
                                    │                                               │
 NHÁNH KAGGLE (gốc)                 │                    NHÁNH PHÂN TÍCH TỔNG QUÁT (src/analytics)
 src/pipeline.py   làm sạch LinkedIn + HN         views/dataset.py → loader + schema (CSV/Parquet bất kỳ,
 src/skills.py     ~80 kỹ năng (tiếng Anh)                           LinkedIn, HN, dữ liệu Việt Nam)
 src/forecast.py   9 mô hình theo tháng           cleaning · eda · domain (VN) · mining · ml · forecasting
                   (gồm ARIMA/SARIMA)             anomaly · reporting (PDF) · db (SQLite)
 src/salary_model.py mô hình lương USD            ai/ (Data Analysis Agent)
 src/chatbot.py    9 công cụ Gemini (Kaggle)                 │
                    └──────────────┬──────────────────────────┘
                                   ▼
                     views/chat.py - Trợ lý AI hợp nhất (định tuyến câu hỏi)
                     views/export.py - CSV / Excel / PDF
```

## 2. Dữ liệu

| Bộ dữ liệu | Nơi lưu | Độ chi tiết thời gian | Dùng cho |
|---|---|---|---|
| LinkedIn CNTT (Mỹ) | `data/processed/linkedin_it.csv.gz` | ngày đăng (3–4/2024) | EDA, lương USD, mô hình lương, dashboard |
| Hacker News | `data/processed/hn_jobs.csv.gz` | **tháng**, 66 kỳ | Xu hướng kỹ năng, dự báo (ARIMA/SARIMA đủ điều kiện) |
| Việt Nam | `data/vn/sample_it_jobs.csv` | năm, 5 kỳ | Lương VND, tỉnh/thành, kỹ năng tiếng Việt |
| CSV/Parquet người dùng | tải lên | tự nhận diện | Mọi module tổng quát |

## 3. Trạng thái phiên (session_state)

| Nhánh | Khóa |
|---|---|
| Kaggle (src/state.py) | `li`, `hn`, `clean_log` (nhật ký pipeline), `fc_result`, `fc_future`, `salary_bundle`, `messages` |
| Tổng quát (src/analytics/state.py) | `raw_df`, `df`, `meta`, `dataset_id`, `time_col`, `target_col`, `ws_clean_log`, `undo_stack`, `results{key: {dataset_id, value}}` |

Khi chọn bộ dữ liệu phân tích mới, `set_dataset()` xóa toàn bộ kết quả cũ; `get_result()` chỉ trả kết quả khớp `dataset_id` hiện tại.

## 4. Trợ lý AI hợp nhất (views/chat.py)

```
Câu hỏi ─► router.route(câu hỏi, bộ dữ liệu phân tích)
   ├─ "data" và không nhắc LinkedIn/HN ─► DataTools (pandas tính) ─► JSON ≤ 12.000 ký tự ─► Gemini diễn giải
   │                                       (không có key ⇒ trả trực tiếp kết quả tính toán)
   └─ còn lại ─► chatbot.ask_gemini: system prompt có bối cảnh Kaggle + 9 công cụ function calling
                 hoặc Google Search (bật trong thanh bên)
```

## 5. Lưu trữ, logging, lỗi

- SQLite `storage/app.db`: `datasets`, `activity_log`, `training_history`, `error_log`; xem tại trang Thông tin hệ thống.
- File log `storage/logs/app.log` (xoay vòng), bộ lọc che API key.
- Mỗi trang mở rộng chạy qua `src/analytics/ui/runner.py`: lỗi được ghi log, hiển thị thân thiện, không làm sập ứng dụng.

## 6. Kiểm thử

`tests/` - 117 test (pytest). Xem bảng trong README mục 4b.
