# Hướng dẫn đưa ITJobTrend lên web

Ứng dụng đã chạy thử được trên Python 3.11–3.13, 117/117 test đạt. Lúc chạy, app dùng khoảng **0,8–2 GB RAM**
(tùy số trang đã mở và số người dùng cùng lúc), dữ liệu kèm theo khoảng 37 MB. Vì vậy:

| Nền tảng | Miễn phí | RAM | Phù hợp? |
|---|---|---|---|
| **Streamlit Community Cloud** | Có | ~2,7 GB | Tốt nhất cho demo, deploy nhanh nhất |
| **Hugging Face Spaces** (Docker) | Có | 16 GB | Tốt, mạnh hơn, cần Dockerfile (đã có sẵn) |
| Render / Railway gói free | Có | 512 MB | Không đủ RAM, dễ bị tắt app |

> Gói miễn phí sẽ "ngủ" khi không có ai truy cập một thời gian. Trước buổi bảo vệ nên mở link trước vài phút để app khởi động lại.

## 0. Bảo mật API Key (bắt buộc đọc)

- **Không** đưa file `.env` hay `.streamlit/secrets.toml` lên GitHub (đã có trong `.gitignore`).
- Trên web, nhập key ở phần **Secrets** của nền tảng (xem bên dưới). Code đọc key theo thứ tự:
  key nhập tạm trong trang Trợ lý AI → file `.env` → biến môi trường → `st.secrets`.
- Lưu ý: nếu bạn cài key của nhóm vào Secrets, **mọi người có link đều dùng chung quota key đó**.
  Nếu không muốn, để trống Secrets, người dùng tự dán key của họ ở trang Trợ lý AI (key chỉ sống trong phiên).
- Nếu lỡ commit key lên GitHub: vào Google AI Studio **xóa key đó và tạo key mới** (xóa commit là chưa đủ).

## 1. Streamlit Community Cloud (khuyến nghị)

1. Đẩy code lên GitHub (repo có thể public hoặc private), `app.py` nằm ở **thư mục gốc** của repo.
2. Vào https://share.streamlit.io → đăng nhập bằng GitHub → **Create app** → **Deploy a public app from GitHub**.
3. Chọn repo `Htuan0901/Cdtn2-Nhom-10`, branch `main`, Main file path `app.py`.
4. Bấm **Advanced settings**: Python version **3.11** hoặc 3.12; ô **Secrets** dán:
   ```toml
   GEMINI_API_KEY = "khóa-của-nhóm"
   GEMINI_MODEL = "gemini-2.5-flash"
   ```
5. Bấm **Deploy**. Lần đầu cài thư viện mất khoảng 3–5 phút. Link có dạng `https://<tên>.streamlit.app`.
6. Mỗi lần `git push` lên `main`, app tự cập nhật.

## 2. Hugging Face Spaces (Docker)

1. Tạo tài khoản https://huggingface.co → **New Space** → SDK: **Docker** (Blank) → Hardware: CPU basic (free).
2. Mở file `README.md` của Space, đảm bảo phần đầu có:
   ```yaml
   ---
   title: ITJobTrend
   sdk: docker
   app_port: 8501
   ---
   ```
3. Đưa toàn bộ code (gồm `Dockerfile` có sẵn) lên Space: tab **Files → Add file → Upload files**, hoặc `git push` vào repo của Space.
4. **Settings → Variables and secrets → New secret**: `GEMINI_API_KEY` = khóa của nhóm.
5. Space tự build Docker (~5–8 phút). Link: `https://huggingface.co/spaces/<user>/<space>`.

## 3. Chạy bằng Docker trên máy bất kỳ

```bash
docker build -t itjobtrend .
docker run -p 8501:8501 -e GEMINI_API_KEY=khóa-của-nhóm itjobtrend
```

## Giới hạn cần biết khi chạy trên web

- Upload file tối đa **200 MB** (`.streamlit/config.toml`). Chạy ở máy cá nhân muốn tăng thì sửa `maxUploadSize`.
- Thư mục `storage/` (SQLite nhật ký, mô hình đã huấn luyện, file xuất) **bị xóa mỗi lần app khởi động lại**
  và **dùng chung giữa mọi người truy cập** (trang Thông tin hệ thống hiển thị nhật ký chung). Đây là bản demo, không lưu dữ liệu lâu dài.
- Trang "Nhập dữ liệu Kaggle" cần upload `archive.zip` (~160 MB) và chạy pipeline vài phút, có thể chậm trên gói free.
  App đã có sẵn dữ liệu xử lý trong `data/processed/` nên không cần bước này khi demo.
