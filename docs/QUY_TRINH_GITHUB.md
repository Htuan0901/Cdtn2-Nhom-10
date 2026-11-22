# Quy trình làm việc trên GitHub của Nhóm 10

Repo: https://github.com/Htuan0901/Cdtn2-Nhom-10

## 1. Luồng làm việc

1. Mỗi việc trong backlog là một **Issue**, gắn vào milestone của sprint và người phụ trách.
2. Tạo nhánh từ `main` mới nhất, đặt tên `feature/<việc>` hoặc `fix/<lỗi>`:
   ```bash
   git checkout main && git pull
   git checkout -b feature/ten-viec
   ```
3. Commit nhỏ, message tiếng Việt nói rõ **làm gì và vì sao** (dòng đầu ≤ 72 ký tự, các dòng sau giải thích).
4. `git push -u origin feature/ten-viec`, mở Pull Request vào `main`. Mô tả PR điền theo mẫu
   `.github/pull_request_template.md`, ghi `Closes #số-issue` để issue tự đóng khi merge.
5. CI **Kiem thu tu dong** tự chạy `pytest` (gồm test giao diện 20 trang). Đỏ thì sửa và push tiếp lên cùng nhánh.
6. Một thành viên khác review và **Approve**. Khi CI xanh và có 1 approve thì bấm **Merge**.

## 2. Bảo vệ nhánh `main` (đã cấu hình)

Ruleset **bao-ve-main** (Settings → Rules → Rulesets) áp cho nhánh mặc định:

| Quy tắc | Trạng thái |
|---|---|
| Cấm xóa nhánh `main` | Đã bật |
| Cấm force push (`non_fast_forward`) | Đã bật |
| Mọi thay đổi phải qua Pull Request, cần **1 approve** | Đã bật |
| Bắt buộc CI `test` báo xanh mới được merge | Bật cùng PR thêm test giao diện (xem mục 2.1) |

### 2.1 Cách tự kiểm tra / bật lại bằng tay

1. Vào repo → **Settings** → menu trái **Rules → Rulesets** → bấm **bao-ve-main**.
2. **Enforcement status**: `Active`. **Target branches**: `Default` (tức `main`).
3. Trong **Branch rules** đánh dấu:
   - ☑ Restrict deletions
   - ☑ Block force pushes
   - ☑ Require a pull request before merging → *Required approvals*: `1`
   - ☑ Require status checks to pass → **Add checks** → gõ `test` → chọn check **test** (GitHub Actions)
4. Bấm **Save changes**.

> Check `test` chỉ hiện trong danh sách sau khi workflow đã chạy ít nhất một lần trên repo.

## 3. Bảng backlog theo sprint trên GitHub Projects

Issue và milestone đã tạo sẵn. Bảng Project phải tạo bằng giao diện web (một lần, khoảng 3 phút):

1. Mở https://github.com/Htuan0901?tab=projects → **New project** → chọn mẫu **Board** → tên `Nhóm 10 – Backlog ITJobTrend` → **Create**.
2. Góc phải → **⋯ → Settings**:
   - **Custom fields → + New field**: tên `Sprint`, kiểu **Iteration**, độ dài 1 tuần (hoặc kiểu Single select với 4 giá trị Sprint 1…Sprint 4).
   - **Manage access**: thêm 2 thành viên với quyền *Write*.
3. Quay lại bảng → **⋯ → Workflows → Auto-add to project**: chọn repo `Cdtn2-Nhom-10`, filter `is:issue,pr`, bật **On**.
   Các issue cũ không tự vào: ở ô cuối cột bấm **+ Add item → #** chọn repo, thêm các issue #5–#12.
4. Trong repo: tab **Projects → Link a project** → chọn bảng vừa tạo để bảng hiện ở trang repo.
5. Cột mặc định *Todo / In Progress / Done*: kéo thẻ khi bắt đầu làm; PR merge kèm `Closes #` thì thẻ tự sang *Done*.

### Nội dung backlog

| Sprint | Việc | Người làm | Theo dõi |
|---|---|---|---|
| 1 – Nền tảng | Khung app, cấu hình `.env`, giao diện dùng chung, SQLite, đọc CSV/Parquet, kết nối Gemini, CI ban đầu | Tuấn | PR #1, PR #4 |
| 2 – Dữ liệu & phân tích | Pipeline làm sạch LinkedIn/HN, dữ liệu VN, trích xuất kỹ năng, EDA, dashboard | Quỳnh | PR #2, PR #4 |
| 3 – Mô hình | Dự báo theo tháng (Naive/MA/Holt/ARIMA/SARIMA), chống rò rỉ, mô hình lương, thực nghiệm Bảng 4.2/4.3 | Nhân | PR #4 |
| 4 – Hoàn thiện & triển khai | CI chạy pytest mỗi PR | Tuấn | #5 |
| 4 | Bảo vệ main, mẫu PR, bảng backlog | Tuấn | #6 |
| 4 | Test giao diện 20 trang bằng AppTest | Tuấn | #7 |
| 4 | `APP_PUBLIC=1` ẩn nhật ký dùng chung | Tuấn | #8 |
| 4 | Giới hạn câu hỏi Gemini mỗi phiên | Tuấn | #9 |
| 4 | Deploy Streamlit Cloud, kiểm thử máy tính + điện thoại, nhật ký lỗi | Tuấn | #10 |
| 4 | Báo cáo mục 4.2 Phân tích hệ thống | Tuấn | #11 |
| 4 | Báo cáo Phụ lục B, C, D | Tuấn | #12 |

Sprint 1–3 đã hoàn thành qua các PR trên. Thành viên khác bổ sung việc của mình bằng cách tạo Issue mới, gắn milestone sprint tương ứng.
