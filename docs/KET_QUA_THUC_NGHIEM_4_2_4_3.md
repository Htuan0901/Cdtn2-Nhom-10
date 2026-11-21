# KẾT QUẢ THỰC NGHIỆM – ITJobTrend (Bảng 4.2 và 4.3)

*Kết quả tính trực tiếp từ dữ liệu có trong `data/processed/`, không phải số liệu giả định.*

## 1. Quy trình đánh giá

- Hacker News: 43,160 tin, 66 tháng (01/2018–06/2023).
- Train: 01/2018–06/2022 (54 kỳ).
- Validation: 07/2022–12/2022 (6 kỳ), chọn mô hình theo RMSE.
- Test: 01/2023–06/2023 (6 kỳ), chỉ đánh giá sau khi chốt model.
- Tham số ARIMA/SARIMA được xác định bằng AIC **trên tập huấn luyện của mỗi lượt chạy**, không sử dụng điểm test.
- Sau khi chọn theo validation, các mô hình fit lại trên train+validation để báo cáo MAE/RMSE ở test.

## Bảng 4.2 – So sánh mô hình dự báo số lượng tin tuyển dụng (test)

| Mô hình | Validation RMSE | MAE test | RMSE test | Cải thiện RMSE so với Naive |
|---|---:|---:|---:|---:|
| SARIMA (mùa vụ 12) **(chọn)** | 175.84 | 90.39 | 96.44 | -1.31% |
| Naive (baseline)  | 184.37 | 87.00 | 95.19 | 0.00% |
| Holt (damped)  | 186.97 | 44.31 | 49.91 | 47.56% |
| ARIMA (chọn theo AIC)  | 196.27 | 102.79 | 110.26 | -15.83% |
| Trung bình trượt 3T  | 216.22 | 84.33 | 92.76 | 2.55% |

**Mô hình chọn bằng validation:** SARIMA (mùa vụ 12). *Không chọn lại bằng điểm test.*

## Bảng 4.3 – So sánh mô hình dự đoán lương (test)

Dữ liệu LinkedIn: 12,099 tin CNTT; 3,560 tin có công bố lương.
Chia ngẫu nhiên nhưng cố định `random_state=42`: train 2136, validation 712, test 712. 
Chọn mô hình hồi quy theo MAE của validation; test không tham gia lựa chọn.

| Mô hình | MAE validation (USD) | MAE test (USD) | RMSE test (USD) | R² test |
|---|---:|---:|---:|---:|
| Random Forest **(chọn)** | 31,853.90 | 28,961.09 | 39,211.02 | 0.4145 |
| Gradient Boosting  | 31,915.14 | 29,256.25 | 40,462.28 | 0.3766 |
| Ridge (hồi quy tuyến tính)  | 33,126.86 | 30,552.93 | 41,407.32 | 0.3471 |
| Baseline (trung vị)  | 41,808.83 | 39,184.79 | 51,492.40 | -0.0096 |

**Mô hình lương chọn bằng validation:** Random Forest.

## Giới hạn dữ liệu – Xu hướng mức lương

- LinkedIn có lương trong 01/2024–04/2024.
- Gần như toàn bộ lương quan sát tập trung ở **04/2024**, còn 01/2024 chỉ có 1 mẫu; không đủ để khẳng định xu hướng dài hạn.
- Trang 'Dự đoán mức lương' hiển thị trung vị **theo nhóm vị trí × tháng/quý**, kèm **số tin** và ngưỡng mẫu tối thiểu.
- Biểu đồ chỉ phản ánh lương **được công bố trong mẫu LinkedIn**, không phải mức lương thị trường chung.

## Tái lập thực nghiệm

```bash
pip install -r requirements.txt
python tools/run_experiments.py
pytest -q
```

Số gốc và bảng chi tiết: `experiments/bang_4_2_du_bao.csv`, `experiments/bang_4_2_validation.csv`, `experiments/bang_4_3_luong.csv`.
Khi thay đổi dữ liệu hoặc tham số, phải chạy lại script; không sao chép số liệu này sang dữ liệu mới.
