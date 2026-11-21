# ITJobTrend — Bản hoàn thiện phần việc của Nhân

## Nội dung đã hoàn thiện

1. `src/forecast.py` và `src/analytics/forecasting/forecaster.py`: tách train / validation / test theo thời gian. Chọn mô hình bằng RMSE **validation**; test độc lập chỉ dùng để báo cáo MAE/RMSE, không xếp hạng lại mô hình dựa trên test. Bổ sung Holt vào dự báo tổng quát; xử lý kỳ thiếu theo hướng không nhìn tương lai.
2. `src/salary_model.py` và `views/salary.py`: biểu đồ **trung vị lương theo tháng/quý và nhóm vị trí**, kèm số tin, bộ lọc và cảnh báo cỡ mẫu. Các mô hình Ridge, Random Forest, Gradient Boosting cũng được chọn bằng **validation MAE**, trước khi xem test.
3. `tools/run_experiments.py` và `experiments/`: kết quả thực nghiệm trên dữ liệu đóng gói, gồm Bảng 4.2 (Naive/MA/Holt/ARIMA/SARIMA), Bảng 4.3 (Baseline/Ridge/Random Forest/Gradient Boosting), bảng validation, dự báo từng tháng và thống kê lương.
4. `tests/test_itjobtrend.py`: kiểm thử chống rò rỉ (đổi giá trị test không thay đổi lựa chọn mô hình), kiểm thử Holt, số liệu lương trung vị và xử lý chuỗi thiếu kỳ.
5. Tài liệu hướng dẫn trình bày và giải thích mô hình: `docs/HUONG_DAN_GIAI_THICH_MODULE_NHAN.md`.

Các module ML tổng quát, K-Means/PCA, Isolation Forest và các hàm phân tích của trợ lý AI được **giữ lại** vì nằm trong phần cần giải thích khi bảo vệ.

## Cách chạy trên Windows (VS Code → Terminal)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Lưu ý: luôn mở terminal **trong thư mục `ITJobTrend`** (chứa `app.py`). Không cần Gemini API Key để chạy phần thống kê, dự báo và các kiểm thử.

## Chạy lại thí nghiệm và kiểm thử

```powershell
.\.venv\Scripts\python.exe tools/run_experiments.py
.\.venv\Scripts\python.exe -m pytest -q
```

Kết quả để sao chép vào báo cáo:

- `docs/KET_QUA_THUC_NGHIEM_4_2_4_3.md`
- `experiments/bang_4_2_du_bao.csv`
- `experiments/bang_4_3_luong.csv`

**Quan trọng:** ZIP dự án nguồn không bao gồm bản báo cáo Word gốc, nên không thể sửa trực tiếp Bảng 4.2/4.3 trong tệp Word của nhóm. Nội dung bảng đã được tạo ở các tệp trên để chèn vào báo cáo. Phân tích xu hướng lương chỉ có vài kỳ quan sát, không được diễn giải thành tăng/giảm lương dài hạn toàn thị trường.

Không đóng gói `.env` để tránh lộ cấu hình hoặc API Key cá nhân; nếu muốn dùng trợ lý AI Gemini, tạo `.env` bằng cách sao chép `.env.example` rồi điền key riêng.
