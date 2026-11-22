# Nhật ký deploy và kiểm thử bản web ITJobTrend

- Link app: https://itjobtrend.streamlit.app
- Repo: https://github.com/Htuan0901/Cdtn2-Nhom-10 (nhánh `main`, file chính `app.py`)
- Nền tảng: Streamlit Community Cloud, Python 3.11
- Người phụ trách: Nguyễn Hoàng Tuấn

> Quy ước: mục **A** là các lỗi đã gặp và đã sửa trong code (có commit/PR làm bằng chứng).
> Mục **C** là kết quả kiểm thử thật trên web, **người kiểm thử tự điền** sau khi bấm thử. Không điền trước khi đã thử.

## A. Lỗi đã gặp khi chuẩn bị deploy và cách sửa

| # | Ngày | Hiện tượng | Nguyên nhân | Cách sửa | Bằng chứng |
|---|---|---|---|---|---|
| 1 | 16/11/2026 | Đã nhập `GEMINI_API_KEY` trong Secrets nhưng app vẫn báo "Chưa có Gemini API key" | `load_dotenv(override=True)` cho file `.env` rỗng ghi đè biến môi trường của nền tảng | Đổi thành `override=False`, đọc thêm `st.secrets` làm nguồn cuối | `src/config.py`, hàm `get_gemini_key` |
| 2 | 16/11/2026 | Upload file lớn làm app trên gói free bị tắt vì hết RAM | `maxUploadSize = 1024` MB vượt RAM gói free | Giảm còn 200 MB, đồng bộ `MAX_UPLOAD_MB` | `.streamlit/config.toml`, `src/analytics/config.py` |
| 3 | 16/11/2026 | Render/Railway gói free (512 MB) khởi động xong vài phút thì bị kill | App dùng 0,8–2 GB RAM | Chọn Streamlit Community Cloud (~2,7 GB) hoặc Hugging Face Spaces | `DEPLOY.md` |
| 4 | 21/11/2026 | Nhánh của một thành viên làm Streamlit không đọc được `config.toml` | Còn dấu xung đột `<<<<<<<` / `>>>>>>>` sau khi merge hai lịch sử không liên quan | Không merge nhánh đó, đưa code sạch lên qua PR #4 | PR #3 (đóng), PR #4 |
| 5 | 22/11/2026 | Trên bản web, ai mở trang **Thông tin hệ thống** cũng xem được câu hỏi chat, tên file, traceback của người khác | SQLite `storage/app.db` dùng chung cho mọi phiên trên server | Thêm `APP_PUBLIC=1`: ẩn 3 tab nhật ký, ẩn đường dẫn CSDL và đuôi key | `src/analytics/ui/pages/system.py`, `tests/test_ui.py` |
| 6 | 22/11/2026 | Người lạ có link có thể hỏi Gemini không giới hạn bằng key của nhóm, hết quota trước buổi bảo vệ | Key của nhóm đặt trong Secrets, không có giới hạn | Giới hạn `GEMINI_MAX_QUESTIONS` câu mỗi phiên (mặc định 20 khi `APP_PUBLIC=1`); người dùng key riêng thì không giới hạn | `src/chat_quota.py`, `views/chat.py`, `tests/test_ui.py` |

Thêm dòng mới vào bảng này mỗi khi gặp lỗi trên bản web (ghi cả lỗi chưa sửa được, ghi "chưa sửa" ở cột Cách sửa).

## B. Cấu hình Secrets trên Streamlit Cloud

share.streamlit.io → app **itjobtrend** → **⋮ → Settings → Secrets**:

```toml
GEMINI_API_KEY = "khóa-của-nhóm"
GEMINI_MODEL = "gemini-flash-latest"
APP_PUBLIC = "1"
GEMINI_MAX_QUESTIONS = "20"
```

Bấm **Save**, chờ app khởi động lại (khoảng 1 phút). Sau đó làm phần C.

## C. Checklist kiểm thử bản web (người kiểm thử điền)

Ghi: **Đạt** / **Lỗi** (mô tả ngắn, rồi thêm vào bảng A) / **Bỏ qua**. Máy tính: ghi trình duyệt (Chrome, Edge...).
Điện thoại: ghi tên máy và trình duyệt (vd. iPhone 13 – Safari, Samsung A54 – Chrome).

Thiết bị đã dùng:
- Máy tính: ……………… Ngày thử: ……/……/2026
- Điện thoại: ……………… Ngày thử: ……/……/2026

| # | Bước kiểm thử | Kết quả mong đợi | Máy tính | Điện thoại |
|---|---|---|---|---|
| 1 | Mở link app (lần đầu có thể phải bấm "Yes, get this app back up") | Trang chủ hiện trong ≤ 2 phút, không có hộp lỗi đỏ | | |
| 2 | Thanh bên: xem "Trạng thái dữ liệu" | LinkedIn 12.099 tin, Hacker News có số tin | | |
| 3 | Điện thoại: bấm nút `>` góc trên trái để mở menu | Menu 20 trang mở/đóng được, chữ không bị cắt | | |
| 4 | Mở lần lượt 6 nhóm trang: Tổng quan, Dữ liệu, Phân tích, Mô hình, Trợ lý & Báo cáo, Hệ thống | Không trang nào hiện "Đã xảy ra lỗi" | | |
| 5 | Bộ dữ liệu phân tích → chọn dữ liệu Việt Nam mẫu | Hiện số dòng, sidebar cập nhật "Bộ phân tích" | | |
| 6 | EDA thị trường CNTT, Dashboard thị trường | Biểu đồ hiện đủ, phóng to/thu nhỏ được; điện thoại cuộn ngang bảng được | | |
| 7 | Xu hướng & dự báo: chạy backtest và dự báo | Có bảng MAE/RMSE và biểu đồ dự báo | | |
| 8 | Dự đoán mức lương: huấn luyện, nhập thử 1 hồ sơ | Ra mức lương dự đoán | | |
| 9 | Trợ lý AI: hỏi "Top 10 kỹ năng cho Data Engineer" | Có câu trả lời; dòng "Còn x/20 câu hỏi Gemini" giảm 1 | | |
| 10 | Trợ lý AI: hỏi liên tục tới khi hết lượt (hoặc tạm đặt `GEMINI_MAX_QUESTIONS = "2"`) | Hiện cảnh báo vàng "đã dùng hết … câu hỏi Gemini", không gọi Gemini nữa | | |
| 11 | Thông tin hệ thống | Chỉ có 2 tab *Trạng thái module*, *Thư viện*; có dòng "bản web công khai (APP_PUBLIC=1)" | | |
| 12 | Xuất kết quả & báo cáo: tạo và tải PDF | Tải được file PDF, mở được, tiếng Việt không lỗi font | | |
| 13 | Xoay ngang điện thoại, tải lại trang (F5) | Bố cục không vỡ; sau F5 phiên mới, bộ đếm câu hỏi về lại 0 | | |

Ảnh chụp màn hình khi thử nên lưu vào `docs/screenshots/web/` (vd. `dt_01_trang_chu.png`, `pc_09_tro_ly_ai.png`) để chèn vào báo cáo.

## D. Kiểm thử tự động (đã chạy, không cần điền)

`pytest` chạy trên GitHub Actions ở mỗi Pull Request vào `main` (workflow **Kiem thu tu dong**). Trong đó
`tests/test_ui.py` dùng `streamlit.testing.AppTest` mở lần lượt 20 trang ở hai trạng thái (phiên mới và đã chọn
bộ dữ liệu) và kiểm tra `APP_PUBLIC`, giới hạn câu hỏi Gemini. Test này chạy app trong tiến trình Python, không
thay được việc bấm thử trên trình duyệt thật ở mục C (giao diện điện thoại, tốc độ mạng, Secrets trên Cloud).
