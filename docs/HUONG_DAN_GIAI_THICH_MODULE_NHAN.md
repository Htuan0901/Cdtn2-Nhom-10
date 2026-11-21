# Hướng dẫn đọc mã nguồn và trả lời phản biện – Phần Nhân

> Tài liệu dành cho việc bảo vệ/demo. Các kết quả định lượng chính thức xem `KET_QUA_THUC_NGHIEM_4_2_4_3.md`.

## 1) Những gì đã sửa và lý do

**Rò rỉ dữ liệu khi chọn mô hình:** Một sai lầm phổ biến là chạy Naive, Moving Average, Holt, ARIMA/SARIMA trên tập test rồi lấy mô hình có RMSE test nhỏ nhất. Khi đó test gián tiếp đã tham gia tối ưu, kết quả không còn là đánh giá độc lập. Bản sửa tách **train → validation → test** theo thời gian. Từng mô hình học train, dự báo validation; **chọn theo RMSE validation**. Sau khi khóa mô hình, fit lại trên **train+validation**, đánh giá các mô hình trên test và báo cáo kết quả. Sau khi đánh giá, có thể học lại toàn bộ chuỗi để dự báo các tháng chưa quan sát. Test không dùng chọn model, không dùng chọn siêu tham số hay thời lượng validation.

**Lưu ý về rò rỉ gián tiếp:** Cột số ở các kỳ không có quan sát của module dự báo tổng quát đã chuyển từ nội suy nhìn cả hai phía sang `forward fill`, chỉ lấy giá trị trước đó. Các thủ tục làm sạch mang tính thống kê (như biến đổi/chuẩn hóa theo toàn dữ liệu) cần được đưa vào pipeline fit trên train khi dùng ở bài toán ML có nhãn. Bản forecast hiện xử lý kỳ thiếu một cách nhân quả, không được dùng kết quả làm sạch bằng toàn bộ dữ liệu trước thời điểm test mà không kiểm soát.

**Biểu đồ xu hướng lương:** Tab `Xu hướng lương` trên trang `Dự đoán mức lương` nhóm theo **ngày đăng → tháng hoặc quý** và **role**, tính `median(salary_usd)` (USD/năm). Chỉ dùng tin có lương dương hợp lệ, hiển thị số tin và lọc theo `min_posts`. Tập dữ liệu có số kỳ rất hạn chế (chủ yếu 03–04/2024) nên không được khẳng định lương đang tăng/giảm theo thời gian cho toàn thị trường.

## 2) Dự báo nhu cầu tuyển dụng – 2 module

- `src/forecast.py`: module cho **Hacker News đã làm sạch**. `monthly_series()` đếm tin theo tháng; `evaluate_models()` trả train, validation, test, bảng RMSE/MAE, tên model được chọn; `backtest()` giữ giao diện API cũ; `forecast()` dự báo tương lai bằng toàn bộ chuỗi sau khi kết thúc đánh giá. `views/forecast.py` hiển thị tách validation/test.
- `src/analytics/forecasting/forecaster.py`: module **tổng quát** cho dữ liệu được người dùng tải lên; `prepare_series()` chọn cột thời gian, gộp theo tần suất, bù kỳ thiếu; `check_eligibility()` kiểm tra tối thiểu số kỳ; `run_forecast()` chọn bằng validation và báo cáo test; `fc_naive`, `fc_moving_average`, `fc_holt`, `fc_arima` cài thuật toán. Trang: `src/analytics/ui/pages/forecast.py`.

| Thuật toán | Nguyên lý ngắn gọn | Điểm cần lưu ý |
|---|---|---|
| Naive | Lấy giá trị cuối của train làm dự báo h kỳ | Baseline; không học xu hướng/mùa vụ |
| Moving Average | Dùng trung bình các quan sát gần nhất (mặc định 3) | Làm mượt nhiễu, dự báo dạng hằng |
| Holt (damped) | Làm trơn mũ cho **mức** và **xu hướng** tắt dần | Không mô hình hóa mùa vụ rõ rệt |
| ARIMA(p,d,q) | Tự hồi quy, sai phân để làm dừng, trung bình trượt sai số | Bậc (p,d,q) lựa chọn bằng AIC **trên dữ liệu fit**, không dựa test |
| SARIMA | ARIMA thêm thành phần mùa vụ chu kỳ 12 tháng (nếu MS) | Chỉ chạy khi ≥ 3 chu kỳ mùa vụ trong train; chi phí lớn |
| Seasonal Naive | Lấy quan sát cùng tháng năm trước | Cần ít nhất 12 tháng lịch sử |
| Holt-Winters | Làm trơn mũ có mức, xu hướng và mùa vụ | Hữu ích khi có mùa vụ rõ |
| Ridge + lags | Dự báo từ trễ 1/2/3/12 và biến tháng, rồi lặp đệ quy | Cần chuỗi đủ dài để lập lags |

**Thước đo:** `MAE = mean(|y−ŷ|)` (dễ đọc theo số tin), `RMSE = sqrt(mean((y−ŷ)²))` (nhạy với lỗi lớn); `MAPE` là sai số % và không phù hợp nếu y=0. Baseline Naive làm mốc so sánh. **Một mô hình có RMSE test thấp hơn nhưng chưa chắc được chọn** vì thứ tự đã khóa bằng validation; ví dụ Bảng 4.2 chọn SARIMA, mặc dù Holt có RMSE test thấp hơn.

## 3) Dự đoán lương – `src/salary_model.py`

- `build_features()` biến đặc trưng gồm `role`, `experience`, `state_grp`, `work_type`, `remote` và kỹ năng đa nhãn; danh sách top bang lấy từ **train**.
- Biến mục tiêu: `salary_usd` (>0), chuẩn hóa về USD/năm từ pipeline LinkedIn. Đầu vào phân loại được mã hóa OneHot trong `ColumnTransformer` của sklearn, fit trên tập train tương ứng.
- `train_salary_models()` chia dữ liệu thành **60% train / 20% validation / 20% test** với seed=42. Học `log(salary)` để giảm lệch; `exp(prediction)` trả lại đơn vị USD/năm.
- So sánh `DummyRegressor(median)` (baseline), `Ridge(alpha=3.0)` (hồi quy tuyến tính L2), `RandomForestRegressor(200 cây, min_samples_leaf=3)` (bagging), và `HistGradientBoostingRegressor(max_iter=300)` (boosting, tối ưu tuần tự).
- `best` xác định từ **MAE validation** của 3 thuật toán chính (không chọn baseline); tất cả mô hình fit lại trên train+validation để báo cáo test. `predict_salary()` dùng mô hình đã học để thử hồ sơ mới.
- `MAE` đo sai số USD trung bình, `RMSE` phạt sai số lớn mạnh hơn, `R²` đo mức biến thiên được giải thích so với trung bình (có thể âm). Dữ liệu đăng tuyển là lương **công bố**, không phải lương thực nhận.
- Phần `effect` của Ridge có phép `exp(coef)-1`, diễn giải sự thay đổi tương đối gắn với đặc trưng, **không khẳng định quan hệ nhân quả**.

## 4) Machine Learning tổng quát – `src/analytics/ml/trainer.py`

- `detect_problem_type()` kiểm tra biến mục tiêu liên tục hay phân loại; `usable_features()` đề xuất cột và bỏ cột không phù hợp.
- `leakage_check()` phát hiện đặc trưng gần trùng với mục tiêu hoặc nguy cơ rò rỉ; vẫn cần người làm kiểm tra logic nghiệp vụ.
- `build_preprocessor()` điền thiếu, mã hóa biến phân loại, chuẩn hóa biến số cho các mô hình cần scale; biến đổi nằm trong Pipeline học từ train.
- `train_models()` chia dữ liệu, so sánh baseline với nhiều thuật toán hồi quy/phân loại; đánh giá bằng RMSE/R² hoặc Accuracy/Precision/Recall/F1 tùy bài toán; có validation/CV phù hợp. `save_model()` và `predict_one()` hỗ trợ lưu và dự đoán.
- Câu hỏi thường gặp: **Vì sao không chọn Accuracy khi mất cân bằng lớp?** Vì có thể đoán tất cả về lớp đa số vẫn Accuracy cao; nên nhìn Recall/F1, confusion matrix và độ cân bằng.

## 5) Phân cụm K-Means và PCA – `src/analytics/mining/clustering.py`

- `prepare_matrix()` tạo ma trận số sạch, xử lý thiếu và scale. `elbow_silhouette()` thử nhiều K, đo inertia và Silhouette Score; `suggest_k()` chọn số cụm theo tiêu chí nội tại (chỉ mang tính gợi ý).
- `run_kmeans()` fit K-Means, gán nhãn cụm; `cluster_profiles()` mô tả cụm theo trung bình các biến; `describe_clusters()` diễn giải khác biệt. **K-Means là học không giám sát**: nhãn cụm không phải nhãn công việc thật.
- PCA giúp chiếu dữ liệu nhiều chiều về 2 chiều để trực quan hóa; `explained_variance_ratio` cho biết bao nhiêu phương sai giữ lại. PCA không tự phân cụm và không chứng minh các nhóm có ý nghĩa nghiệp vụ.

## 6) Isolation Forest – `src/analytics/anomaly/detector.py`

- `detect()` học nhiều cây phân tách ngẫu nhiên; những điểm dễ bị cô lập sau ít bước chia được xem là bất thường. Tham số `contamination` quy định tỷ lệ bất thường dự kiến, **không phải xác suất gian lận**.
- `describe()` tóm tắt số điểm bất thường. Kết quả nên kiểm tra thủ công vì điểm rất hiếm có thể vẫn là hợp lệ.

## 7) Trợ lý AI và các hàm phân tích thực tế

**Hai luồng:**

1. Trợ lý cho LinkedIn/HN: `src/chatbot.py` thiết lập dữ liệu ở `set_data`, tính bằng Python các công cụ `thong_tin_du_lieu`, `top_ky_nang`, `luong_theo_nhom`, `luong_theo_ky_nang`, `xu_huong_ky_nang`, `ky_nang_tang_giam`, `tim_tin`, `dem_theo_cot`, `ket_qua_mo_hinh`. Phần diễn giải ngôn ngữ dùng Gemini tùy câu hỏi.
2. Agent phân tích tổng quát: `src/analytics/ai/router.py` nhận diện ý định; `src/analytics/ai/tools.py` chứa các hàm Pandas/thống kê; `src/analytics/ai/agent.py` kết hợp dữ liệu có cấu trúc và kết quả công cụ; `gemini_client.py` xử lý gọi mô hình. **Số liệu nên do Python/SQL tính**, không yêu cầu LLM tự tưởng tượng thống kê.

Ví dụ demo: “Top 5 kỹ năng trên LinkedIn?” → lọc chuỗi kỹ năng, đếm/tỷ lệ bằng Pandas rồi trả bảng; “Xu hướng Python theo năm?” → nhóm thời gian, tính tỉ lệ theo năm. Có thể kiểm chứng bằng EDA trước khi bật AI; API key không cần cho phần tính toán cơ bản.

## 8) Câu hỏi phản biện nhanh

**Q1. Vì sao chia theo thời gian mà không random?** Vì test phải là tương lai so với train; random làm thông tin tương lai xuất hiện trong tập học.

**Q2. Validation và test khác gì?** Validation dùng chọn mô hình/siêu tham số; test đánh giá cuối khi lựa chọn đã khóa. Không được xem test để thay lựa chọn.

**Q3. Tại sao SARIMA được chọn nhưng Holt lại test tốt hơn?** SARIMA có RMSE validation nhỏ nhất; test chỉ là kiểm tra độc lập. Chọn lại Holt bằng test sẽ gây selection leakage, mất tính khách quan.

**Q4. MAPE có điểm yếu gì?** Chia cho y thực tế; khi y bằng hoặc gần 0 thì sai số tỷ lệ không ổn định hoặc vô nghĩa.

**Q5. Vì sao dùng lương trung vị thay trung bình?** Lương thường lệch phải và có outlier; trung vị bền vững hơn với vài tin trả lương rất cao.

**Q6. Biểu đồ lương nói gì?** Trung vị lương trong **tập tin công bố lương**, từng vị trí/kỳ. Dữ liệu chỉ vài kỳ và rất lệch về tháng 04/2024 nên không khẳng định xu hướng thị trường nhiều năm.

**Q7. Ridge/RF/Gradient Boosting khác nhau thế nào?** Ridge giả định quan hệ tuyến tính và giảm overfit bằng L2; RF trung bình nhiều cây độc lập; GB học cây nối tiếp để sửa lỗi mô hình trước.

**Q8. Vì sao có baseline?** Nếu không vượt được Naive hoặc trung vị, mô hình phức tạp chưa chứng minh lợi ích.

**Q9. Isolation Forest có phải classification có nhãn?** Không. Nó là phát hiện bất thường không giám sát, dựa vào mức dễ cô lập.

**Q10. API AI gọi dữ liệu thật như thế nào?** Router nhận câu hỏi → gọi đúng công cụ Pandas → nhận kết quả JSON có nguồn dữ liệu → mô hình ngôn ngữ diễn giải; công thức và con số ở Python.

## 9) Các lệnh phải nhớ khi demo trên Windows

```powershell
# Từ thư mục ITJobTrend (chứa app.py)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py

# Chạy lại số thực nghiệm (phát sinh file CSV trong experiments/)
python tools/run_experiments.py

# Chạy tất cả test và test riêng phần Nhân
python -m pytest -q
python -m pytest tests/test_itjobtrend.py -v
```

Các màn hình demo: `Xu hướng & dự báo` → so sánh validation/test, `Dự báo tổng quát` → Naive/MA/Holt/ARIMA/SARIMA, `Dự đoán mức lương` → xu hướng lương theo vị trí và đánh giá Ridge/RF/GB, `Khai phá dữ liệu` → KMeans/PCA, `Phát hiện bất thường` → Isolation Forest, `Trợ lý AI` → công cụ tính theo dữ liệu.
