# Cơ sở khoa học của các thuật toán

Tài liệu này được sinh từ `src/analysis/algorithms_doc.py` - cùng nguồn với phần "Cơ sở khoa học" hiển thị trong ứng dụng.

## Phát hiện ngoại lệ IQR & Z-score

| Mục | Nội dung |
|---|---|
| Mục đích | Xác định các giá trị nằm xa phần lớn dữ liệu, có thể do lỗi nhập liệu hoặc hiện tượng hiếm. |
| Nguyên lý hoạt động | IQR: giá trị ngoài [Q1 − k·IQR, Q3 + k·IQR] (k = 1.5). Z-score: |x − μ| / σ > ngưỡng (thường 3). |
| Đầu vào | Một hoặc nhiều cột số. |
| Đầu ra | Cận dưới/cận trên, số lượng và tỷ lệ ngoại lệ mỗi cột. |
| Tham số chính | k (IQR), ngưỡng z; hành động: giữ, loại bỏ, giới hạn (capping). |
| Chỉ số đánh giá | Tỷ lệ ngoại lệ; so sánh thống kê trước/sau xử lý. |
| Ưu điểm | Đơn giản, dễ giải thích. IQR bền vững với phân phối lệch. |
| Hạn chế | Z-score giả định phân phối gần chuẩn; cả hai là đơn biến, không phát hiện bất thường đa biến. |

## K-Means Clustering

| Mục | Nội dung |
|---|---|
| Mục đích | Phân nhóm các bản ghi tương đồng mà không cần nhãn (học không giám sát). |
| Nguyên lý hoạt động | Lặp: gán mỗi điểm vào tâm cụm gần nhất (khoảng cách Euclid) rồi cập nhật tâm = trung bình cụm, tối thiểu hóa tổng bình phương khoảng cách nội cụm (inertia/WCSS). |
| Đầu vào | Các biến số đã chuẩn hóa (StandardScaler) để các biến có thang đo ngang nhau. |
| Đầu ra | Nhãn cụm cho từng bản ghi, tâm cụm, hồ sơ đặc trưng từng cụm. |
| Tham số chính | k (số cụm), n_init = 10, random_state = 42. |
| Chỉ số đánh giá | Elbow (inertia theo k), Silhouette Score ∈ [−1, 1] (càng cao càng tách biệt). |
| Ưu điểm | Nhanh, mở rộng tốt với dữ liệu lớn, kết quả dễ diễn giải qua tâm cụm. |
| Hạn chế | Phải chọn k; giả định cụm hình cầu, kích thước tương đương; nhạy với ngoại lệ và thang đo. |

## Principal Component Analysis (PCA)

| Mục | Nội dung |
|---|---|
| Mục đích | Giảm chiều dữ liệu, trực quan hóa dữ liệu nhiều chiều trên mặt phẳng 2D. |
| Nguyên lý hoạt động | Tìm các trục trực giao (vector riêng của ma trận hiệp phương sai) theo thứ tự giữ lại phương sai lớn nhất. |
| Đầu vào | Ma trận biến số đã chuẩn hóa. |
| Đầu ra | Tọa độ thành phần chính, tỷ lệ phương sai giải thích, hệ số tải (loadings). |
| Tham số chính | n_components. |
| Chỉ số đánh giá | Explained variance ratio (tỷ lệ phương sai giải thích tích lũy). |
| Ưu điểm | Khử tương quan giữa các biến, giảm nhiễu, hỗ trợ trực quan hóa. |
| Hạn chế | Chỉ nắm bắt quan hệ tuyến tính; thành phần chính khó diễn giải trực tiếp. |

## Linear Regression

| Mục | Nội dung |
|---|---|
| Mục đích | Dự đoán biến mục tiêu liên tục bằng tổ hợp tuyến tính của các đặc trưng. |
| Nguyên lý hoạt động | Ước lượng hệ số bằng bình phương tối thiểu (OLS): cực tiểu tổng bình phương phần dư. |
| Đầu vào | Đặc trưng số (chuẩn hóa) + đặc trưng phân loại (One-Hot). |
| Đầu ra | Giá trị dự đoán, hệ số của từng đặc trưng. |
| Tham số chính | Không có siêu tham số chính (fit_intercept). |
| Chỉ số đánh giá | MAE, MSE, RMSE, R². |
| Ưu điểm | Nhanh, dễ giải thích, là baseline tốt. |
| Hạn chế | Không nắm bắt quan hệ phi tuyến; nhạy với đa cộng tuyến và ngoại lệ. |

## Logistic Regression

| Mục | Nội dung |
|---|---|
| Mục đích | Phân loại nhị phân/đa lớp, ước lượng xác suất thuộc lớp. |
| Nguyên lý hoạt động | Mô hình hóa log-odds là hàm tuyến tính của đặc trưng; tối ưu hàm log-loss có điều chuẩn L2. |
| Đầu vào | Đặc trưng đã chuẩn hóa + One-Hot. |
| Đầu ra | Xác suất và nhãn lớp. |
| Tham số chính | C (nghịch đảo độ điều chuẩn), class_weight, max_iter. |
| Chỉ số đánh giá | Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix. |
| Ưu điểm | Xác suất có ý nghĩa, dễ diễn giải, ít overfit. |
| Hạn chế | Biên quyết định tuyến tính. |

## Decision Tree

| Mục | Nội dung |
|---|---|
| Mục đích | Hồi quy/phân loại bằng chuỗi luật if-else. |
| Nguyên lý hoạt động | Chia đệ quy không gian đặc trưng theo ngưỡng làm giảm nhiều nhất độ hỗn tạp (Gini/Entropy) hoặc MSE. |
| Đầu vào | Đặc trưng số + One-Hot (không cần chuẩn hóa). |
| Đầu ra | Dự đoán, độ quan trọng đặc trưng. |
| Tham số chính | max_depth, min_samples_leaf. |
| Chỉ số đánh giá | Theo bài toán (RMSE/R² hoặc F1/Accuracy...). |
| Ưu điểm | Dễ giải thích, xử lý được phi tuyến và tương tác. |
| Hạn chế | Dễ overfit, phương sai cao. |

## Random Forest

| Mục | Nội dung |
|---|---|
| Mục đích | Tăng độ chính xác và ổn định so với một cây đơn. |
| Nguyên lý hoạt động | Bagging: huấn luyện nhiều cây trên mẫu bootstrap với tập đặc trưng ngẫu nhiên, lấy trung bình/biểu quyết. |
| Đầu vào | Đặc trưng số + One-Hot. |
| Đầu ra | Dự đoán, độ quan trọng đặc trưng. |
| Tham số chính | n_estimators, max_depth, min_samples_leaf, class_weight. |
| Chỉ số đánh giá | Theo bài toán. |
| Ưu điểm | Mạnh, ít overfit hơn cây đơn, ít cần tinh chỉnh. |
| Hạn chế | Chậm hơn, khó diễn giải từng dự đoán, mô hình lớn. |

## Gradient Boosting / XGBoost

| Mục | Nội dung |
|---|---|
| Mục đích | Mô hình dạng cây có độ chính xác cao cho dữ liệu bảng. |
| Nguyên lý hoạt động | Boosting: thêm tuần tự các cây nông, mỗi cây khớp với gradient của hàm mất mát (phần dư) của mô hình trước. |
| Đầu vào | Đặc trưng số + One-Hot. |
| Đầu ra | Dự đoán, độ quan trọng đặc trưng. |
| Tham số chính | n_estimators, learning_rate, max_depth (XGBoost: thêm subsample, regularization). |
| Chỉ số đánh giá | Theo bài toán. |
| Ưu điểm | Thường cho kết quả tốt nhất trên dữ liệu bảng. |
| Hạn chế | Dễ overfit nếu không điều chỉnh, huấn luyện tuần tự nên chậm hơn. |

## Naive Forecast (Baseline)

| Mục | Nội dung |
|---|---|
| Mục đích | Mốc so sánh tối thiểu: mô hình phức tạp phải tốt hơn baseline mới có giá trị. |
| Nguyên lý hoạt động | Giá trị dự báo = giá trị quan sát cuối cùng: ŷ(t+h) = y(t). |
| Đầu vào | Chuỗi thời gian đã sắp xếp. |
| Đầu ra | Dự báo hằng; khoảng dự báo theo độ lệch chuẩn của sai phân. |
| Tham số chính | Không có. |
| Chỉ số đánh giá | MAE, RMSE, MAPE. |
| Ưu điểm | Đơn giản, thường khó bị đánh bại với chuỗi ngắn/ngẫu nhiên. |
| Hạn chế | Không mô hình hóa xu hướng hay mùa vụ. |

## Moving Average

| Mục | Nội dung |
|---|---|
| Mục đích | Dự báo bằng mức trung bình gần đây, giảm nhiễu. |
| Nguyên lý hoạt động | ŷ(t+h) = trung bình w quan sát gần nhất. |
| Đầu vào | Chuỗi thời gian. |
| Đầu ra | Dự báo hằng. |
| Tham số chính | Cửa sổ w. |
| Chỉ số đánh giá | MAE, RMSE, MAPE. |
| Ưu điểm | Làm mượt nhiễu, đơn giản. |
| Hạn chế | Trễ khi có xu hướng; không ngoại suy xu hướng. |

## Linear Trend

| Mục | Nội dung |
|---|---|
| Mục đích | Ngoại suy xu hướng tuyến tính dài hạn. |
| Nguyên lý hoạt động | Hồi quy OLS y = a + b·t; khoảng dự báo dựa trên phân phối t của phần dư. |
| Đầu vào | Chuỗi thời gian. |
| Đầu ra | Dự báo và khoảng dự báo 95%. |
| Tham số chính | Không có. |
| Chỉ số đánh giá | MAE, RMSE, MAPE; độ dốc b. |
| Ưu điểm | Dễ giải thích, phù hợp chuỗi ngắn có xu hướng rõ. |
| Hạn chế | Giả định xu hướng không đổi; không có mùa vụ; sai khi có điểm gãy cấu trúc. |

## ARIMA(p,d,q)

| Mục | Nội dung |
|---|---|
| Mục đích | Mô hình hóa tự tương quan của chuỗi thời gian. |
| Nguyên lý hoạt động | Sai phân d lần để chuỗi dừng, sau đó kết hợp tự hồi quy bậc p và trung bình trượt sai số bậc q. Chọn (p,d,q) theo AIC nhỏ nhất trên tập huấn luyện. |
| Đầu vào | Chuỗi thời gian đều (≥ 20 điểm). |
| Đầu ra | Dự báo và khoảng tin cậy. |
| Tham số chính | p, d, q (tìm kiếm lưới p,q ∈ {0,1,2}, d ∈ {0,1}). |
| Chỉ số đánh giá | AIC; MAE, RMSE, MAPE trên tập test. |
| Ưu điểm | Nền tảng thống kê vững, có khoảng tin cậy. |
| Hạn chế | Cần chuỗi đủ dài, giả định tuyến tính và dừng sau sai phân. |

## SARIMA(p,d,q)(P,D,Q,s)

| Mục | Nội dung |
|---|---|
| Mục đích | ARIMA có thành phần mùa vụ chu kỳ s. |
| Nguyên lý hoạt động | Bổ sung các hạng tự hồi quy/trung bình trượt/sai phân ở độ trễ mùa vụ s. |
| Đầu vào | Chuỗi có ≥ 3 chu kỳ mùa vụ đầy đủ. |
| Đầu ra | Dự báo và khoảng tin cậy. |
| Tham số chính | (p,d,q), (P,D,Q), s. |
| Chỉ số đánh giá | AIC; MAE, RMSE, MAPE. |
| Ưu điểm | Mô hình hóa được mùa vụ. |
| Hạn chế | Cần nhiều dữ liệu; tốn tài nguyên. |

## Isolation Forest

| Mục | Nội dung |
|---|---|
| Mục đích | Phát hiện bất thường đa biến, không cần nhãn. |
| Nguyên lý hoạt động | Xây nhiều cây cô lập bằng cách chọn ngẫu nhiên đặc trưng và ngưỡng chia; điểm bất thường bị cô lập sau ít lần chia hơn (độ dài đường đi trung bình ngắn). |
| Đầu vào | Các biến số (đã điền thiếu bằng trung vị và chuẩn hóa). |
| Đầu ra | Điểm bất thường (anomaly score) và nhãn bất thường/bình thường. |
| Tham số chính | contamination (tỷ lệ bất thường kỳ vọng), n_estimators = 200, random_state = 42. |
| Chỉ số đánh giá | Số lượng/tỷ lệ bất thường; phân phối điểm; kiểm tra thủ công các điểm đầu bảng. |
| Ưu điểm | Hiệu quả với dữ liệu lớn, nhiều chiều; không giả định phân phối. |
| Hạn chế | Phải chọn contamination; khó giải thích vì sao một điểm bất thường. |

## Trích xuất kỹ năng (Text mining dựa trên từ điển)

| Mục | Nội dung |
|---|---|
| Mục đích | Chuyển văn bản mô tả công việc thành biến kỹ năng có cấu trúc để đếm, so sánh, dự báo. |
| Nguyên lý hoạt động | Chuẩn hóa văn bản (chữ thường, khoảng trắng) → so khớp biểu thức chính quy với ranh giới từ trên từ điển ~75 kỹ năng (nhiều biến thể/kỹ năng) → ánh xạ về tên chuẩn. Từ viết tắt dễ nhầm (AI, ML, JS) chỉ khớp khi viết hoa. |
| Đầu vào | Tiêu đề, mô tả, yêu cầu công việc. |
| Đầu ra | Tập kỹ năng của mỗi tin; tần suất, tỷ lệ theo năm, đồng xuất hiện, chênh lệch lương. |
| Tham số chính | Từ điển kỹ năng, cột văn bản đầu vào. |
| Chỉ số đánh giá | Tỷ lệ tin trích xuất được ≥ 1 kỹ năng; kiểm tra thủ công mẫu. |
| Ưu điểm | Minh bạch, nhanh, kiểm soát được, hỗ trợ tiếng Việt. |
| Hạn chế | Bỏ sót kỹ năng ngoài từ điển; không hiểu ngữ cảnh phủ định ('không yêu cầu Java'). |

## Chỉ số đánh giá

| Chỉ số | Công thức / ý nghĩa | Dùng cho |
|---|---|---|
| MAE | (1/n)·Σ|y − ŷ| - sai số tuyệt đối trung bình, cùng đơn vị với y | Hồi quy, dự báo |
| MSE | (1/n)·Σ(y − ŷ)² - phạt nặng sai số lớn | Hồi quy |
| RMSE | √MSE - cùng đơn vị với y | Hồi quy, dự báo (tiêu chí chọn mô hình) |
| R² | 1 − SS_res/SS_tot - tỷ lệ phương sai được giải thích | Hồi quy |
| MAPE | (100/n)·Σ|(y − ŷ)/y| - chỉ tính khi không có y = 0 | Dự báo |
| Accuracy | Tỷ lệ dự đoán đúng - dễ gây hiểu nhầm khi lớp mất cân bằng | Phân loại |
| Precision / Recall | TP/(TP+FP) và TP/(TP+FN), trung bình có trọng số | Phân loại |
| F1-macro | Trung bình F1 các lớp (mỗi lớp trọng số như nhau) - tiêu chí chọn mô hình phân loại | Phân loại |
| ROC-AUC | Diện tích dưới đường ROC; 0.5 = ngẫu nhiên | Phân loại (có xác suất) |
| Silhouette | (b − a)/max(a, b) ∈ [−1, 1] | Phân cụm |

## Nguyên tắc chống sai lệch khoa học được cài đặt

1. **Không rò rỉ dữ liệu**: chia train/test trước; Imputer/Scaler/One-Hot nằm trong `Pipeline` và chỉ học trên train; tự cảnh báo cột có |r| ≥ 0,95 với mục tiêu, cột cùng gốc tên, hoặc một biến tự phân loại đúng ≥ 99%.
2. **Luôn có baseline**: DummyRegressor (trung bình) / DummyClassifier (lớp đa số) cho ML; Naive cho dự báo; Naive cho dự báo kỹ năng.
3. **Không kết luận bằng một chỉ số**: báo cáo đồng thời nhiều chỉ số, chênh lệch train–test, độ lệch chuẩn qua 5-fold CV.
4. **Chuỗi thời gian**: chia theo thứ tự thời gian; từ chối ARIMA khi < 20 kỳ huấn luyện, SARIMA khi < 3 chu kỳ mùa vụ; cảnh báo kỳ cuối chưa đầy đủ.
5. **Tái lập**: `random_state = 42` cho mọi bước ngẫu nhiên (chia dữ liệu, lấy mẫu, mô hình).
6. **Mất cân bằng lớp**: tự dùng `class_weight='balanced'` khi tỷ lệ lớp nhỏ/lớn < 0,5; chọn mô hình theo F1-macro.
7. **Không tạo số liệu giả**: mọi KPI, biểu đồ, báo cáo và câu trả lời chatbot đều tính trực tiếp từ dữ liệu; phần chưa chạy ghi rõ "chưa thực hiện".
