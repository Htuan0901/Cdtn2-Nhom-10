import streamlit as st

from src.skills import SKILL_NAMES
from src.state import get_data
from src.ui import fmt_int, hico, kpi_cards, page_header

page_header("trend", "ITJobTrend – Phân tích & dự báo xu hướng tuyển dụng ngành CNTT",
            "Đồ án Chuyên đề tốt nghiệp 2 · Chuyên ngành Khoa học dữ liệu · Dữ liệu: LinkedIn Job Postings 2023–2024 "
            "và Hacker News “Who is hiring” 2018–2023 (Kaggle)")

li, hn, _ = get_data(stop=False)
if li is not None:
    kpi_cards([
        ("Tin CNTT (LinkedIn)", fmt_int(len(li)), "tháng 3–4/2024"),
        ("Tin có lương", fmt_int(li["salary_usd"].notna().sum()), "đã chuẩn hóa USD/năm"),
        ("Tin Hacker News", fmt_int(len(hn)), f"{hn['month'].min():%m/%Y} – {hn['month'].max():%m/%Y}"),
        ("Số tháng (chuỗi thời gian)", hn["month"].nunique(), "dùng để dự báo"),
        ("Kỹ năng theo dõi", len(SKILL_NAMES), "từ điển chuẩn hóa"),
    ])

c1, c2 = st.columns([1.35, 1])
with c1:
    st.markdown(f"""
<div class="card">
<h4>{hico("briefcase")}Bài toán</h4>
Thị trường tuyển dụng CNTT thay đổi rất nhanh: công nghệ mới (Cloud, AI/LLM…) xuất hiện, công nghệ cũ giảm dần.
Sinh viên và nhà trường cần biết <b>kỹ năng nào đang được săn đón</b>, <b>mức lương theo từng vị trí</b> và
<b>xu hướng sắp tới</b>. Đề tài thu thập dữ liệu tin tuyển dụng công khai, làm sạch, trích xuất kỹ năng từ mô tả công việc,
phân tích và <b>dự báo xu hướng tuyển dụng theo thời gian</b>, trình bày qua dashboard trực quan.
</div>
<div class="card">
<h4>{hico("check")}Mục tiêu (theo đề cương)</h4>
<span class="pill">Thu thập & làm sạch minh bạch</span><span class="pill">EDA: kỹ năng hot, lương, khu vực</span>
<span class="pill">Trích xuất kỹ năng (text mining)</span><span class="pill">Dự báo xu hướng theo thời gian</span>
<span class="pill">So sánh với baseline (MAE/RMSE)</span><span class="pill">Dashboard tương tác</span><span class="pill">Trợ lý AI hỏi đáp</span>
</div>
""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
<div class="card">
<h4>{hico("sparkle")}Bắt đầu nhanh</h4>
<div class="step"><div class="num">1</div><div><b>Nhập dữ liệu</b><br><span style="color:#667085">Dữ liệu đã xử lý được nạp sẵn; có thể chạy lại pipeline từ file Kaggle</span></div></div>
<div class="step"><div class="num">2</div><div><b>Làm sạch</b><br><span style="color:#667085">Xem nhật ký từng bước, số dòng trước/sau</span></div></div>
<div class="step"><div class="num">3</div><div><b>EDA & trích xuất kỹ năng</b><br><span style="color:#667085">Kỹ năng hot, lương theo vị trí, khu vực</span></div></div>
<div class="step"><div class="num">4</div><div><b>Dự báo xu hướng</b><br><span style="color:#667085">9 mô hình (gồm ARIMA, SARIMA), so sánh với baseline bằng MAE/RMSE</span></div></div>
<div class="step"><div class="num">5</div><div><b>Dashboard & Trợ lý AI</b><br><span style="color:#667085">Lọc tương tác, hỏi đáp bằng tiếng Việt</span></div></div>
</div>
""", unsafe_allow_html=True)
    st.page_link("views/eda.py", label="Xem phân tích ngay", icon=":material/arrow_forward:")

st.markdown("#### :material/account_tree: Pipeline xử lý dữ liệu")
st.markdown(f"""
<div class="pipeline">
<span class="node">{hico("upload", 15)}Thu thập (Kaggle)</span><span class="arrow">→</span>
<span class="node">{hico("clean", 15)}Làm sạch & chuẩn hóa</span><span class="arrow">→</span>
<span class="node">{hico("code", 15)}Trích xuất kỹ năng</span><span class="arrow">→</span>
<span class="node">{hico("chart", 15)}EDA</span><span class="arrow">→</span>
<span class="node">{hico("trend", 15)}Chuỗi thời gian & dự báo</span><span class="arrow">→</span>
<span class="node">{hico("money", 15)}Mô hình lương</span><span class="arrow">→</span>
<span class="node">{hico("dashboard", 15)}Dashboard</span><span class="arrow">→</span>
<span class="node">{hico("chat", 15)}Trợ lý AI</span>
</div>
""", unsafe_allow_html=True)

t1, t2, t3 = st.tabs([":material/checklist: Yêu cầu chức năng", ":material/build: Công nghệ sử dụng", ":material/dataset: Về bộ dữ liệu"])
with t1:
    st.markdown("""
| Mã | Yêu cầu chức năng | Trang |
|---|---|---|
| FR-01 | Nạp dữ liệu Kaggle (file zip/thư mục), chạy pipeline xử lý | Nhập dữ liệu |
| FR-02 | Làm sạch: xóa trùng, xử lý thiếu, lọc tin CNTT, chuẩn hóa lương/địa điểm/kỹ năng; nhật ký trước/sau | Làm sạch dữ liệu |
| FR-03 | EDA: kỹ năng phổ biến, lương theo vị trí/kinh nghiệm, phân bố theo khu vực | EDA |
| FR-04 | Trích xuất kỹ năng/công nghệ từ mô tả công việc, bảng đặc trưng, thử với văn bản bất kỳ | Trích xuất kỹ năng |
| FR-05 | Phân tích xu hướng theo giai đoạn và dự báo theo tháng; chọn mô hình bằng rolling-origin trên validation, đánh giá test độc lập, so sánh baseline Naive (MAE/RMSE/MAPE) | Xu hướng & dự báo |
| FR-06 | Xu hướng lương trung vị theo năm × nhóm vị trí (Hacker News 2018–2023); mô hình dự đoán mức lương theo vị trí, cấp bậc, kỹ năng | Dự đoán mức lương |
| FR-07 | Dashboard tương tác có bộ lọc | Dashboard |
| FR-08 | Trợ lý AI (Gemini) trả lời câu hỏi về dữ liệu & câu hỏi chung | Trợ lý AI |
| FR-09 | Xuất dữ liệu sạch, bảng đặc trưng, kết quả dự báo (CSV/Excel), **báo cáo PDF** | Xuất kết quả & báo cáo |
| FR-10 | Chọn bộ dữ liệu phân tích: LinkedIn, Hacker News, **dữ liệu Việt Nam** hoặc **CSV/Parquet bất kỳ** (tự nhận diện encoding, dấu phân cách, schema) | Bộ dữ liệu phân tích |
| FR-11 | Dashboard KPI chất lượng dữ liệu (thiếu, trùng, tỷ lệ hợp lệ, số mô hình đã huấn luyện…) | Dashboard tổng quan dữ liệu |
| FR-12 | Làm sạch tương tác: Mean/Median/Mode, IQR/Z-score, StandardScaler/MinMaxScaler, chuyển kiểu, nhật ký, hoàn tác | Làm sạch tương tác |
| FR-13 | EDA tự động thích nghi kiểu dữ liệu + diễn giải ngôn ngữ tự nhiên | EDA tự động |
| FR-14 | Chuẩn hóa tin tuyển dụng Việt Nam (lương VND, kinh nghiệm, 63 tỉnh/thành) + trích xuất kỹ năng tiếng Việt | Thị trường tuyển dụng (VN) |
| FR-15 | Khai phá: K-Means, Elbow, Silhouette, PCA, hồ sơ cụm | Khai phá dữ liệu |
| FR-16 | Machine Learning tổng quát: hồi quy/phân loại, chống rò rỉ, cân bằng lớp, CV 5-fold, baseline, lưu mô hình | Machine Learning |
| FR-17 | Dự báo tổng quát + ARIMA/SARIMA (có điều kiện áp dụng) | Dự báo tổng quát · Xu hướng & dự báo |
| FR-18 | Phát hiện bất thường Isolation Forest, xuất CSV | Phát hiện bất thường |
| FR-19 | Data Analysis Agent: Python tính số liệu, Gemini diễn giải; chạy offline khi chưa có key | Trợ lý AI |
| FR-20 | Lưu lịch sử hoạt động, huấn luyện, lỗi vào SQLite; trang thông tin hệ thống | Thông tin hệ thống |

**Ngoài phạm vi** (theo đề cương): thu thập real-time liên tục, dự báo cá nhân hóa, mô hình deep learning phức tạp cho chuỗi thời gian.
""")
with t2:
    st.markdown("""
| Thành phần | Công nghệ | Lý do chọn |
|---|---|---|
| Xử lý & phân tích | **pandas**, **NumPy** | Chuẩn công nghiệp cho dữ liệu bảng |
| Trích xuất kỹ năng | **Regex** + từ điển ~80 kỹ năng chuẩn hóa | Minh bạch, dễ giải thích, chạy nhanh |
| Dự báo | **statsmodels** (Holt, Holt-Winters), **scikit-learn** (Linear, Ridge) | Mô hình thống kê/ML cơ bản, đúng phạm vi đề tài |
| Mô hình lương | Ridge, Random Forest, Gradient Boosting | So sánh tuyến tính và phi tuyến |
| Dashboard | **Streamlit**, **Plotly** | Web app dữ liệu bằng Python, biểu đồ tương tác |
| Trợ lý AI | **Google Gemini API** + function calling + Data Analysis Agent | Hỏi đáp tự nhiên, số liệu do Python tính |
| Khai phá / bất thường | **scikit-learn** (K-Means, PCA, Isolation Forest), **XGBoost** | Học không giám sát, phát hiện điểm bất thường |
| Lưu trữ & báo cáo | **SQLite**, **ReportLab**, **Matplotlib/Seaborn** | Nhật ký hoạt động, báo cáo PDF tiếng Việt |
| Kiểm thử | **pytest** | Kiểm thử tự động các module |
""")
with t3:
    st.markdown("""
**1. LinkedIn Job Postings (2023–2024)**: [kaggle.com/datasets/arshkon/linkedin-job-postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)
123.849 tin tuyển dụng tại Mỹ (chủ yếu đăng tháng 3–4/2024), gồm `postings.csv` (tiêu đề, mô tả, lương, địa điểm, cấp bậc…)
và các bảng phụ `job_skills`, `job_industries`, `salaries`, `companies`. Dùng cho **EDA lương, kỹ năng, khu vực** và **mô hình lương**.

**2. Hacker News “Who is hiring” 2018–2023**: [kaggle.com/datasets/kidrulit/hackernews-who-is-hiring-listings-from-2018](https://www.kaggle.com/datasets/kidrulit/hackernews-who-is-hiring-listings-from-2018)
47.206 tin tuyển dụng CNTT đăng hàng tháng trên Hacker News (01/2018 – 06/2023): `jobs_2018_2023.csv` (nội dung tin)
và `parsed_jobs_2018_2023.csv` (công ty, quốc gia, remote, vị trí, tech stack). Dùng cho **chuỗi thời gian và dự báo xu hướng**.

Vì sao cần 2 nguồn: LinkedIn có lương và cấu trúc rõ nhưng chỉ là ảnh chụp 1 thời điểm; Hacker News kéo dài 66 tháng nên dự báo được.
""")

st.markdown('<div class="footer-note">ITJobTrend · Đồ án Chuyên đề tốt nghiệp 2 · Khoa CNTT</div>', unsafe_allow_html=True)
