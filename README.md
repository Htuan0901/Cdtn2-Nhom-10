# 📈 ITJobTrend: Phân tích & dự báo xu hướng tuyển dụng ngành CNTT

**Đồ án Chuyên đề tốt nghiệp 2, chuyên ngành Khoa học dữ liệu** · phiên bản 2.0 (gộp thêm các module phân tích tổng quát)

Ứng dụng web (Streamlit) xử lý trọn quy trình khoa học dữ liệu trên dữ liệu tin tuyển dụng CNTT công khai:
thu thập (Kaggle) → làm sạch & chuẩn hóa → trích xuất kỹ năng → EDA → phân tích xu hướng & dự báo theo tháng → mô hình lương → dashboard → trợ lý AI Gemini.

## 0. Dữ liệu

| Bộ dữ liệu | Link Kaggle | Quy mô | Dùng cho |
|---|---|---|---|
| LinkedIn Job Postings (2023–2024) | https://www.kaggle.com/datasets/arshkon/linkedin-job-postings | 123.849 tin (Mỹ, chủ yếu 3–4/2024) → **12.099 tin CNTT** sau làm sạch | EDA kỹ năng, lương theo vị trí/cấp bậc/bang, mô hình lương |
| Tin tuyển dụng Việt Nam 2022–2026 (`data.parquet`) | file của nhóm | 606.878 tin → mẫu **9.999 tin** (`data/vn/sample_it_jobs.csv`, phân tầng theo năm, random_state = 42) | Lương VND, tỉnh/thành, kinh nghiệm, kỹ năng tiếng Việt (chỉ có mốc **năm**) |
| Hacker News “Who is hiring” 2018–2023 | https://www.kaggle.com/datasets/kidrulit/hackernews-who-is-hiring-listings-from-2018 | 47.206 tin → **43.160 tin**, 66 tháng (01/2018–06/2023) | Chuỗi thời gian, xu hướng kỹ năng, dự báo |

Thư mục `data/processed/` **đã có sẵn dữ liệu đã xử lý** (~18 MB), nên mở app là dùng được ngay, không cần tải file Kaggle.
Muốn chạy lại pipeline từ dữ liệu gốc: giải nén 2 file zip Kaggle vào `data/raw/linkedin/` và `data/raw/hn/`, rồi vào trang **Nhập dữ liệu** bấm
**Chạy pipeline** (hoặc chạy `python prepare_data.py --linkedin "đường_dẫn\archive.zip" --hn "đường_dẫn\archive (1).zip"`). Mất khoảng 3 phút.

Muốn dùng **toàn bộ** dữ liệu Việt Nam: `python tools/prepare_vn_dataset.py "đường_dẫn\data.parquet"` → tạo `data/vn/it_jobs_full.csv` (~14.000 tin CNTT), rồi tải file này lên trang **Bộ dữ liệu phân tích** (trang này cũng nhận trực tiếp file `.parquet`).

## 1. Cấu trúc thư mục

```
ITJobTrend/
├── app.py                  # File chạy chính (điều hướng các trang)
├── prepare_data.py         # Chạy pipeline từ dữ liệu Kaggle gốc (dòng lệnh)
├── .env.example            # Mẫu cấu hình; sao chép sang .env trên máy riêng
├── requirements.txt        # Danh sách thư viện
├── data/
│   ├── raw/                # Đặt dữ liệu Kaggle gốc (không bắt buộc)
│   └── processed/          # Dữ liệu đã làm sạch + nhật ký làm sạch (có sẵn)
├── src/
│   ├── pipeline.py         # Làm sạch, lọc tin CNTT, chuẩn hóa lương/địa điểm/tháng
│   ├── skills.py           # Từ điển ~80 kỹ năng + regex trích xuất, phân loại vị trí
│   ├── forecast.py         # Chuỗi thời gian, 9 mô hình dự báo (gồm ARIMA, SARIMA), backtest, rolling origin
│   ├── salary_model.py     # Ridge / Random Forest / Gradient Boosting dự đoán lương
│   ├── chatbot.py          # Trợ lý Gemini + 9 công cụ truy vấn dữ liệu
│   ├── state.py, config.py, ui.py
│   └── analytics/          # ★ Các module phân tích tổng quát (phiên bản 2.0)
│       ├── data/           #   loader (CSV/Parquet, encoding, dấu phân cách) · schema (nhận diện vai trò cột) · cleaning
│       ├── domain/         #   jobs (lương VND, kinh nghiệm, 63 tỉnh, lọc CNTT) · skills (từ điển kỹ năng tiếng Việt)
│       ├── analysis/       #   eda (thống kê, tương quan, diễn giải tự động) · algorithms_doc (cơ sở khoa học)
│       ├── mining/         #   K-Means, Elbow, Silhouette, PCA
│       ├── ml/             #   huấn luyện hồi quy/phân loại, chống rò rỉ, CV, baseline, lưu mô hình
│       ├── forecasting/    #   Naive, MA, Linear Trend, Holt, ARIMA, SARIMA (có điều kiện áp dụng)
│       ├── anomaly/        #   Isolation Forest
│       ├── ai/             #   Data Analysis Agent: router → tools (pandas) → Gemini
│       ├── db/             #   SQLite: lịch sử hoạt động, huấn luyện, nhật ký lỗi
│       ├── reporting/      #   báo cáo PDF (ReportLab) + biểu đồ PNG
│       └── ui/             #   các trang mở rộng + thành phần giao diện
├── views/                  # 20 trang giao diện (10 trang gốc + 10 trang mở rộng)
├── tests/                  # pytest tự động: có test chống rò rỉ dữ liệu
├── tools/prepare_vn_dataset.py   # Parquet Việt Nam → CSV
├── data/vn/                # Dữ liệu Việt Nam (mẫu)
├── storage/                # Tự tạo khi chạy: app.db (SQLite), models/, logs/
└── docs/                   # ARCHITECTURE.md, ALGORITHMS.md, screenshots/
```

## 2. Hướng dẫn chạy trên Visual Studio Code (Windows)

1. Cài **Python 3.10+** (tick **Add python.exe to PATH**) và **VS Code** + extension **Python**.
2. Giải nén `ITJobTrend.zip`, trong VS Code chọn **File → Open Folder…** thư mục `ITJobTrend`, mở Terminal (``Ctrl+` ``).
3. Tạo môi trường ảo và cài thư viện:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
   (Nếu PowerShell báo *running scripts is disabled*: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`.)
4. (Tùy chọn) Cấu hình Gemini - xem mục **2b** bên dưới.
5. Chạy: `streamlit run app.py` (hoặc nhấn **F5**, hoặc nhấp đúp `run_windows.bat`). Trình duyệt mở http://localhost:8501.

## 2b. Cấu hình Gemini API

1. Lấy key miễn phí tại https://aistudio.google.com/app/apikey → **Create API key** → Copy.
2. Mở file `.env` (cùng thư mục với `app.py`; nếu chưa có thì copy từ `.env.example`), điền:
   ```
   GEMINI_API_KEY=AIza...key_cua_ban...
   GEMINI_MODEL=gemini-2.5-flash
   ```
   Không để dấu cách hay dấu nháy. Lưu file rồi F5. Có thể dán key tạm trong thanh bên của trang Trợ lý AI (chỉ lưu trong phiên).
3. Kiểm tra: trang **Thông tin hệ thống → Kiểm tra kết nối**.

Key không bao giờ được ghi trong mã nguồn hay file log (bộ lọc log tự che chuỗi dạng `AIza…`). **Chưa có key** thì câu hỏi định lượng
về bộ dữ liệu phân tích vẫn được trả lời bằng số liệu tính trực tiếp (chế độ offline).

## 3. Kịch bản demo

| Bước | Trang | Nội dung |
|---|---|---|
| 1 | 🏠 Trang chủ | Bài toán, mục tiêu, yêu cầu chức năng, công nghệ, nguồn dữ liệu |
| 2 | 📥 Nhập dữ liệu | Dữ liệu đã nạp sẵn; có thể chạy lại pipeline từ file zip Kaggle |
| 3 | 🧹 Làm sạch dữ liệu | Biểu đồ thác nước số dòng qua từng bước, nhật ký, kiểm tra chất lượng |
| 4 | 🔎 EDA | Insight tự động; kỹ năng hot, đồng xuất hiện; lương theo vị trí/cấp bậc/kỹ năng; bản đồ theo bang; HN theo tháng |
| 5 | 🧩 Trích xuất kỹ năng | Thử trích xuất trên tin thật hoặc văn bản tự nhập (tô đậm kỹ năng), từ điển, bảng đặc trưng |
| 6 | 📊 Dashboard | Lọc theo vị trí, cấp bậc, bang, kỹ năng, remote, giai đoạn |
| 7 | 📈 Xu hướng & dự báo | Top kỹ năng tăng/giảm 2018–19 → 2022–23; chọn chuỗi, backtest 9 mô hình (gồm ARIMA, SARIMA) (MAE/RMSE/MAPE so với baseline), dự báo 3–18 tháng |
| 8 | 💵 Dự đoán mức lương | So sánh 3 mô hình với baseline; **xu hướng lương trung vị theo tháng/quý và vị trí**; yếu tố ảnh hưởng; dự đoán thử |
| 9 | ✨ Trợ lý AI | Hỏi “Kỹ năng nào tăng mạnh nhất?”, “Lương Data Engineer ở CA?”… |
| 10 | 📤 Xuất kết quả & báo cáo | CSV dữ liệu sạch, bảng đặc trưng, kết quả dự báo, báo cáo Excel nhiều sheet, **báo cáo PDF** |

**Các trang mở rộng (phiên bản 2.0)** - làm việc trên *bộ dữ liệu phân tích* do bạn chọn:

| Bước | Trang | Nội dung |
|---|---|---|
| 11 | Bộ dữ liệu phân tích | Chọn LinkedIn / Hacker News / **Việt Nam** hoặc tải CSV/Parquet bất kỳ; tự nhận diện encoding (UTF-8, BOM, CP1258), dấu phân cách, kiểu dữ liệu; 20 dòng đầu; chọn cột thời gian & biến mục tiêu |
| 12 | Dashboard tổng quan dữ liệu | 8 thẻ KPI (bản ghi, thuộc tính, ô thiếu, tỷ lệ hợp lệ, trùng lặp, biến số, số mô hình đã huấn luyện, thời điểm cập nhật) + 6 nhóm biểu đồ + phát hiện quan trọng |
| 13 | Làm sạch tương tác | Thiếu (Mean/Median/Mode/hằng số/ffill/xóa), trùng lặp (bỏ qua ID), ngoại lệ IQR/Z-score (giữ/loại/capping), chuẩn hóa, chuyển kiểu, ngày tháng, giá trị không hợp lệ; xem trước tác động, nhật ký, hoàn tác, tải CSV |
| 14 | EDA tự động | Thống kê mô tả đầy đủ, tần suất, histogram/boxplot/scatter, Pearson/Spearman, groupby, thời gian, khu vực + diễn giải tự động |
| 15 | Thị trường tuyển dụng (VN) | Với dữ liệu Việt Nam: chuẩn hóa lương VND, kinh nghiệm, tỉnh/vùng, lọc tin CNTT, trích xuất ~75 kỹ năng tiếng Việt, xu hướng & dự báo kỹ năng theo năm (Linear vs Naive) |
| 16 | Khai phá dữ liệu | K-Means, Elbow, Silhouette, PCA, hồ sơ cụm, mẫu đáng chú ý |
| 17 | Machine Learning | Hồi quy/phân loại tự nhận diện; 5 mô hình + XGBoost; cảnh báo rò rỉ; cân bằng lớp; CV 5-fold; baseline; lưu mô hình; dự đoán thử |
| 18 | Dự báo tổng quát | Chọn cột thời gian/biến/tần suất; Naive, MA, Linear Trend, **Holt, ARIMA, SARIMA**; từ chối mô hình khi không đủ điều kiện kèm lý do |
| 19 | Phát hiện bất thường | Isolation Forest, contamination tùy chỉnh, PCA, xuất CSV |
| 20 | Thông tin hệ thống | Phiên bản, thư viện, trạng thái Gemini, trạng thái module, lịch sử hoạt động/huấn luyện, nhật ký lỗi (SQLite) |

Trang **Xu hướng & dự báo** (HN theo tháng) nay có thêm **ARIMA (chọn bậc theo AIC)** và **SARIMA (mùa vụ 12)** - tổng 9 mô hình.
Trang **Trợ lý AI**: câu hỏi định lượng về bộ dữ liệu phân tích đi qua **Data Analysis Agent** (Python tính → Gemini diễn giải,
có mục *Minh bạch tính toán*); câu hỏi về LinkedIn/HN, kiến thức chung, thời sự dùng Gemini + công cụ Kaggle / Google Search như cũ.

## 4. Kết quả chính (để đưa vào báo cáo)

- **Làm sạch:** LinkedIn 123.849 → 12.099 tin CNTT (xóa 12.650 tin lặp, lọc tin ngoài ngành); 3.560 tin có lương (trung vị ~132k USD/năm). HN 47.206 → 43.160 tin.
- **Kỹ năng hot (LinkedIn):** Python, Agile/Scrum, SQL, AWS, Azure, Java. **Lương cao nhất:** nhóm Data Scientist/AI-ML; kỹ năng LLM/GenAI gắn với lương trung vị cao nhất.
- **Xu hướng (HN, 2018–19 → 2022–23):** TypeScript tăng mạnh nhất (+17 điểm %), tiếp theo Rust, Terraform, GraphQL; JavaScript, Java, Docker, Angular, Python giảm tỉ lệ.
  Số tin/tháng giảm từ ~750 (2018) xuống ~420 (12 tháng gần nhất), phản ánh làn sóng cắt giảm nhân sự công nghệ 2022–2023.
- **Dự báo số tin/tháng (train 54 / validation 6 / test 6):** SARIMA được chọn theo RMSE validation; trên test SARIMA RMSE 96,44 so với Naive 95,19. Holt trên test đạt RMSE 49,91 nhưng không được chọn lại dựa vào test. Xem số liệu đầy đủ trong `docs/KET_QUA_THUC_NGHIEM_4_2_4_3.md`.
- **Mô hình lương (train 2136 / validation 712 / test 712):** Random Forest được chọn theo MAE validation, có MAE test 28.961 USD và R² test 0,4145 (Naive baseline trung vị MAE 39.185 USD).

## 4a. Thay đổi hoàn thiện yêu cầu (11/2026)

- **Không rò rỉ dữ liệu khi chọn mô hình**: `src/forecast.py` và `src/analytics/forecasting/forecaster.py`
  chia liên tục thành train → validation → test, xếp hạng theo **validation RMSE**, báo cáo test *sau* khi chốt mô hình.
  Điền kỳ thiếu numeric bằng **forward fill** (chỉ thông tin quá khứ); không nội suy xuyên tương lai.
- **Biểu đồ lương**: `views/salary.py` và `src/salary_model.py`: tab **Xu hướng lương** dùng trung vị USD/năm
  từng vị trí theo tháng/quý, hiển thị cả cỡ mẫu (n), ngưỡng tin tối thiểu. Dữ liệu lương chỉ tập trung Mar–Apr/2024;
  không diễn giải là xu hướng dài hạn.
- **Bảng 4.2 / 4.3**: `tools/run_experiments.py` chạy lại thí nghiệm, xuất các bảng vào `experiments/`
  và văn bản để chèn báo cáo tại `docs/KET_QUA_THUC_NGHIEM_4_2_4_3.md`.
- **Test mới**: `tests/test_itjobtrend.py` kiểm tra thay test không đổi model được chọn, kiểm tra tính nhân quả
  của xử lý chuỗi, tính đúng trung vị lương và kết quả Holt.
- **Tài liệu bảo vệ**: `docs/HUONG_DAN_GIAI_THICH_MODULE_NHAN.md` ghi rõ cơ chế các mô hình, luồng công cụ AI và câu hỏi phản biện.

Chạy lại kết quả trước khi nộp nếu bạn đổi dữ liệu hoặc model:

```powershell
python tools/run_experiments.py
python -m pytest -q
```

## 4b. Kiểm thử

```powershell
pytest                 # toàn bộ (~1–2 phút)
pytest -v tests/test_itjobtrend.py
```

| File | Nội dung |
|---|---|
| `test_itjobtrend.py` | Pipeline Kaggle, kỹ năng, backtest theo tháng, **ARIMA/SARIMA/Holt**, chống rò rỉ train/validation/test và điều kiện giữ nguyên mô hình chọn khi thay nhãn test, lương trung vị, chatbot |
| `test_data.py` | Upload CSV hợp lệ / rỗng / chỉ tiêu đề / nhị phân, UTF-8 BOM, `;` và tab, schema, thiếu, trùng lặp, sai kiểu, ngoại lệ, chuẩn hóa, undo, đổi dataset không dùng nhầm kết quả cũ |
| `test_jobs.py` | Chuẩn hóa lương VND/USD/theo tuần, kinh nghiệm, tỉnh/thành, lọc CNTT, kỹ năng tiếng Việt |
| `test_analytics.py` | EDA, K-Means/PCA, ML (hồi quy, phân loại mất cân bằng, rò rỉ), dự báo, Isolation Forest |
| `test_ai.py` | Router câu hỏi, công cụ dữ liệu, **AI không có API Key**, **AI trả lời câu hỏi dataset bằng số liệu thật**, lỗi 429/401, che key trong log |
| `test_ui.py` | **Test giao diện bằng `streamlit.testing.AppTest`**: mở lần lượt cả 20 trang (phiên mới và khi đã chọn dataset), không trang nào báo lỗi; `APP_PUBLIC=1` ẩn nhật ký dùng chung; giới hạn câu hỏi Gemini mỗi phiên (Gemini giả, không tốn quota) |

Tài liệu: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/ALGORITHMS.md](docs/ALGORITHMS.md).

## 5. Xử lý lỗi thường gặp
| Lỗi | Cách khắc phục |
|---|---|
| `'streamlit' is not recognized` | Chưa activate môi trường ảo: `.venv\Scripts\activate` |
| `python is not recognized` | Cài lại Python và tick "Add to PATH", hoặc dùng lệnh `py` |
| `No module named statsmodels` | Chạy lại `pip install -r requirements.txt` |
| App báo chưa có dữ liệu | Kiểm tra thư mục `data/processed/` còn 3 file; hoặc chạy pipeline ở trang Nhập dữ liệu |
| `API key không hợp lệ` | Kiểm tra key trong `.env` (không dấu cách, không dấu nháy), lưu rồi F5 |
| Lỗi 429 / quota | Key miễn phí bị giới hạn số lượt mỗi phút, đợi một lát rồi hỏi lại |
| Trang mở rộng báo "Chưa chọn bộ dữ liệu phân tích" | Vào **Bộ dữ liệu phân tích**, bấm *Dùng LinkedIn / Hacker News / dữ liệu Việt Nam* hoặc tải CSV |
| Trang Xu hướng & dự báo chạy chậm lần đầu | ARIMA/SARIMA cần vài giây để huấn luyện; kết quả được cache cho các lần sau |
