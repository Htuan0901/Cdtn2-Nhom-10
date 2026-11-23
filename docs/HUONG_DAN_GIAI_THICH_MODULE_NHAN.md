# Hướng dẫn đọc mã nguồn và trả lời phản biện – Phần Nhân

> Tài liệu dành cho việc bảo vệ/demo. Các kết quả định lượng chính thức xem `KET_QUA_THUC_NGHIEM_4_2_4_3.md`.

## 1) Những gì đã sửa và lý do

**Rò rỉ dữ liệu khi chọn mô hình:** Một sai lầm phổ biến là chạy Naive, Moving Average, Holt, ARIMA/SARIMA trên tập test rồi lấy mô hình có RMSE test nhỏ nhất. Khi đó test gián tiếp đã tham gia tối ưu, kết quả không còn là đánh giá độc lập. Bản sửa tách **train → validation → test** theo thời gian. Vùng validation (12 tháng 01–12/2022 trên Hacker News) được đánh giá bằng **rolling-origin 4 mốc × 3 tháng**: ở mỗi mốc mô hình học toàn bộ dữ liệu trước mốc rồi dự báo 3 tháng kế tiếp; **chọn theo RMSE trung bình** của 4 mốc. Thêm **quy tắc tiết kiệm**: mô hình khác Naive phải giảm RMSE validation ≥ 5% (hằng `NAIVE_MARGIN`) mới được chọn, nếu không giữ Naive. Số mốc, độ dài mốc và ngưỡng 5% được cố định trong code (`VALIDATION_MONTHS`, `ORIGIN_STEP`, `NAIVE_MARGIN`) trước khi xem test. Lý do dùng nhiều mốc: với một cửa sổ 6 tháng duy nhất, lựa chọn phụ thuộc vào may rủi của đúng 6 tháng đó (bản trước chọn SARIMA nhưng trên test SARIMA còn kém Naive). Sau khi khóa mô hình, fit lại trên **train+validation**, đánh giá các mô hình trên test và báo cáo kết quả. Sau khi đánh giá, có thể học lại toàn bộ chuỗi để dự báo các tháng chưa quan sát. Test không dùng chọn model, không dùng chọn siêu tham số hay thời lượng validation.

**Lưu ý về rò rỉ gián tiếp:** Cột số ở các kỳ không có quan sát của module dự báo tổng quát đã chuyển từ nội suy nhìn cả hai phía sang `forward fill`, chỉ lấy giá trị trước đó. Các thủ tục làm sạch mang tính thống kê (như biến đổi/chuẩn hóa theo toàn dữ liệu) cần được đưa vào pipeline fit trên train khi dùng ở bài toán ML có nhãn. Bản forecast hiện xử lý kỳ thiếu một cách nhân quả, không được dùng kết quả làm sạch bằng toàn bộ dữ liệu trước thời điểm test mà không kiểm soát.

**Biểu đồ xu hướng lương:** Tab `Xu hướng lương` trên trang `Dự đoán mức lương` có 2 nguồn. **Mặc định là Hacker News 2018–2023** (1.067 tin ghi khoảng lương; lương = trung điểm khoảng, ví dụ $120k–$160k → 140k), gộp theo **năm / nửa năm / quý** và **nhóm vị trí**, tính `median(salary_usd)`, kèm đường "Tất cả vị trí" và bảng % thay đổi kỳ đầu → kỳ cuối. Nhóm × kỳ có ít hơn ngưỡng (mặc định 10 tin) bị loại. Kết quả: trung vị tổng tăng từ 125.000 USD (2018) lên 165.000 USD (6 tháng đầu 2023), +32%. Nguồn LinkedIn vẫn giữ để đối chiếu nhưng gần như chỉ có tin 04/2024 nên không dùng kết luận xu hướng. Hàm: `salary_trend_table(df, freq, min_posts, roles, date_col, include_overall)` và `salary_trend_change(trend)`.

## 2) Dự báo nhu cầu tuyển dụng – 2 module

- `src/forecast.py`: module cho **Hacker News đã làm sạch**. `monthly_series()` đếm tin theo tháng; `evaluate_models()` chạy rolling-origin trên validation, chốt mô hình (`best`, `selection_rule`), rồi mới đánh giá test; trả về `validation_table` (RMSE TB), `validation_detail` (từng mốc), `test_table` (có cột `Được chọn`); `rolling_backtest(..., exclude_last=h)` là bảng tham khảo độ ổn định, luôn cắt bỏ tập test; `backtest()` giữ giao diện API cũ; `forecast()` dự báo tương lai bằng toàn bộ chuỗi sau khi kết thúc đánh giá. `views/forecast.py` hiển thị tách validation/test.
- `src/analytics/forecasting/forecaster.py`: module **tổng quát** cho dữ liệu được người dùng tải lên; `prepare_series()` chọn cột thời gian, gộp theo tần suất, bù kỳ thiếu; `check_eligibility()` kiểm tra tối thiểu số kỳ; `run_forecast()` chọn bằng một cửa sổ validation rồi báo cáo test (dữ liệu tải lên thường ngắn, không đủ cho nhiều mốc rolling-origin); `fc_naive`, `fc_moving_average`, `fc_holt`, `fc_arima` cài thuật toán. Trang: `src/analytics/ui/pages/forecast.py`.

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

**Thước đo:** `MAE = mean(|y−ŷ|)` (dễ đọc theo số tin), `RMSE = sqrt(mean((y−ŷ)²))` (nhạy với lỗi lớn); `MAPE` là sai số % và không phù hợp nếu y=0. Baseline Naive làm mốc so sánh. **Một mô hình có RMSE test thấp hơn nhưng chưa chắc được chọn** vì lựa chọn đã khóa bằng validation. Lưu ý: kết quả test của từng mô hình KHÔNG phụ thuộc cách chọn (mô hình nào cũng học trên train+validation rồi dự báo 6 tháng test), nên số MAE/RMSE test của Holt, ARIMA, SARIMA giữ nguyên giữa bản cũ và bản mới; chỉ tên mô hình được chọn thay đổi.

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

**Q3. Nếu mô hình khác có RMSE test thấp hơn mô hình được chọn thì sao?** Đó là chuyện bình thường: test chỉ đo sai số thật khi triển khai. Nếu đổi lựa chọn theo test thì test đã tham gia chọn mô hình (selection leakage) và con số báo cáo sẽ lạc quan giả tạo. Cách đúng là cải thiện quy trình chọn trên validation (bản này đã chuyển từ 1 cửa sổ sang rolling-origin 4 mốc) chứ không nhìn test.

**Q3b. Vì sao ngưỡng 5% với Naive?** Nguyên tắc tiết kiệm: mô hình phức tạp hơn phải tốt hơn rõ rệt mới đáng dùng, vì chênh lệch nhỏ trên 12 tháng validation có thể chỉ là nhiễu. Ngưỡng đặt trước và cố định trong code, không chỉnh theo kết quả test.

**Q4. MAPE có điểm yếu gì?** Chia cho y thực tế; khi y bằng hoặc gần 0 thì sai số tỷ lệ không ổn định hoặc vô nghĩa.

**Q5. Vì sao dùng lương trung vị thay trung bình?** Lương thường lệch phải và có outlier; trung vị bền vững hơn với vài tin trả lương rất cao.

**Q6. Biểu đồ lương nói gì?** Trung vị lương trong **các tin Hacker News có ghi khoảng lương**, theo năm và nhóm vị trí, 2018–2023: tăng khoảng 32% (125k → 165k USD). Giới hạn: chỉ 1.067 tin có lương (khoảng 2,5% số tin), là lương công bố chứ không phải lương thực nhận, 2023 chỉ có 6 tháng, một số nhóm/năm ít mẫu nên dao động (ví dụ Frontend 2020). Không dùng LinkedIn cho xu hướng vì gần như chỉ có 04/2024.

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

# Chạy lại số thực nghiệm (CSV + hình PNG trong experiments/, bảng tổng hợp trong docs/KET_QUA_THUC_NGHIEM_4_2_4_3.md)
# Bắt buộc đã cài statsmodels, nếu không script sẽ dừng để tránh thiếu Holt/ARIMA/SARIMA
python tools/run_experiments.py

# Chạy tất cả test và test riêng phần Nhân
python -m pytest -q
python -m pytest tests/test_itjobtrend.py -v
```

Các màn hình demo: `Xu hướng & dự báo` → so sánh validation/test, `Dự báo tổng quát` → Naive/MA/Holt/ARIMA/SARIMA, `Dự đoán mức lương` → tab Xu hướng lương (Hacker News theo năm) và đánh giá Ridge/RF/GB, `Khai phá dữ liệu` → KMeans/PCA, `Phát hiện bất thường` → Isolation Forest, `Trợ lý AI` → công cụ tính theo dữ liệu.
