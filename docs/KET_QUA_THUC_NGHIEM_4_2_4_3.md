# KẾT QUẢ THỰC NGHIỆM – ITJobTrend (Bảng 4.2, 4.3 và xu hướng lương)

*Sinh tự động bởi `tools/run_experiments.py` từ dữ liệu trong `data/processed/`.*

## 1. Quy trình đánh giá dự báo

- Hacker News: 43,160 tin, 66 tháng (01/2018–06/2023).
- Validation 01/2022–12/2022: rolling-origin 4 mốc × 3 tháng. Ở mỗi mốc, mô hình học toàn bộ dữ liệu trước mốc rồi dự báo các tháng tiếp theo; tiêu chí chọn là RMSE trung bình.
- Quy tắc tiết kiệm: mô hình khác Naive phải giảm RMSE validation ≥ 5% so với Naive mới được chọn.
- Test 01/2023–06/2023 (6 tháng): mọi mô hình học lại trên train+validation, dự báo một lần, chỉ để báo cáo. Không chọn lại theo test.
- Bậc ARIMA chọn theo AIC trên dữ liệu huấn luyện của từng lượt, không dùng điểm test.

**Mô hình được chọn:** Naive (baseline) – mô hình tốt nhất khác Naive (Holt (damped)) còn kém Naive 9.4% RMSE validation, chưa đạt ngưỡng giảm 5% ⇒ giữ Naive.

## Bảng 4.2 – So sánh mô hình dự báo số tin tuyển dụng

| Mô hình | RMSE validation (TB) | MAE test | RMSE test | Cải thiện RMSE test so với Naive |
|---|---:|---:|---:|---:|
| Naive (baseline) **(được chọn)** | 69.19 | 87.00 | 95.19 | 0.0% |
| Holt (damped) | 75.68 | 44.31 | 49.91 | 47.6% |
| Ridge + biến trễ | 78.85 | 138.32 | 146.71 | -54.1% |
| Holt-Winters | 81.24 | 68.04 | 73.11 | 23.2% |
| ARIMA (chọn theo AIC) | 86.57 | 102.79 | 110.26 | -15.8% |
| Trung bình trượt 3T | 90.06 | 84.33 | 92.76 | 2.6% |
| SARIMA (mùa vụ 12) | 112.08 | 90.39 | 96.44 | -1.3% |
| Hồi quy xu hướng + mùa vụ | 139.16 | 304.15 | 305.91 | -221.4% |
| Seasonal Naive | 248.31 | 346.17 | 347.01 | -264.5% |

## Bảng 4.3 – So sánh mô hình dự đoán lương (LinkedIn)

3,560 tin có lương. Chia ngẫu nhiên cố định (seed 42): train 2136, validation 712, test 712. Chọn theo MAE validation; mô hình cuối học lại trên train+validation rồi đánh giá test.

| Mô hình | MAE validation (USD) | MAE test (USD) | RMSE test (USD) | R² test | Giảm MAE so với baseline |
|---|---:|---:|---:|---:|---:|
| Random Forest **(được chọn)** | 31,849 | 28,957 | 39,205 | 0.415 | 26.1% |
| Gradient Boosting | 31,915 | 29,256 | 40,462 | 0.377 | 25.3% |
| Ridge (hồi quy tuyến tính) | 33,127 | 30,553 | 41,407 | 0.347 | 22.0% |
| Baseline (trung vị) | 41,809 | 39,185 | 51,492 | -0.010 | 0.0% |

## Xu hướng mức lương (Hacker News, theo năm)

1,067 tin có ghi khoảng lương; lương = trung điểm khoảng. Chỉ giữ nhóm × năm có ≥ 10 tin. Năm 2023 chỉ có 6 tháng đầu.

| Nhóm vị trí | Kỳ đầu | Lương kỳ đầu (USD) | Kỳ cuối | Lương kỳ cuối (USD) | Thay đổi | Số tin |
|---|---|---:|---|---:|---:|---:|
| Backend / Full-stack | 2018 | 125,000 | 2023 | 170,000 | +36.0% | 242 |
| DevOps / Cloud / SRE | 2018 | 125,000 | 2022 | 143,750 | +15.0% | 63 |
| Frontend / Web | 2018 | 122,500 | 2022 | 145,000 | +18.4% | 105 |
| Software Engineer | 2018 | 125,000 | 2023 | 177,500 | +42.0% | 283 |
| Tất cả vị trí | 2018 | 125,000 | 2023 | 165,000 | +32.0% | 1067 |

LinkedIn chỉ có lương trong 01–04/2024 (gần hết ở 04/2024) nên chỉ dùng cho mô hình lương, không dùng cho xu hướng.

## Hình cho báo cáo

- `experiments/hinh_du_bao_test.png` – chuỗi số tin, vùng validation/test và dự báo của mô hình được chọn.
- `experiments/hinh_xu_huong_luong.png` – lương trung vị theo năm và nhóm vị trí.

## Tái lập

```bash
pip install -r requirements.txt
python tools/run_experiments.py
pytest -q
```
