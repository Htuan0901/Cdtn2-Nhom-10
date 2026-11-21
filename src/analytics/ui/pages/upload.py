"""MODULE 2 - Data Upload Center: tải, kiểm tra, nhận diện schema, chọn cột thời gian & biến mục tiêu."""
import streamlit as st

from src.analytics import state
from src.analytics.config import MAX_UPLOAD_MB, PREVIEW_ROWS, SAMPLE_DATA
from src.analytics.data import loader, schema
from src.analytics.db.database import log_activity, log_error, register_dataset
from src.analytics.domain.jobs import is_job_dataset
from src.analytics.ui import components as ui
from src.analytics.ui import nav
from src.analytics.ui.theme import BLUE, NAVY, TEAL


def _builtin_sources():
    """Ba bộ dữ liệu có sẵn của đồ án."""
    from src.state import get_data
    ss = st.session_state
    ui.section("Bộ dữ liệu có sẵn", "Bấm để dùng làm bộ dữ liệu phân tích")
    li, hn, _ = get_data(stop=False)
    c1, c2, c3 = st.columns(3)
    pick = None
    with c1:
        st.markdown(f"<div class='card'><h4>LinkedIn CNTT (Mỹ, 2024)</h4>{len(li):,} tin đã làm sạch · lương USD/năm · "
                    "vị trí, cấp bậc, bang, kỹ năng</div>" if li is not None else "<div class='card'><h4>LinkedIn</h4>Chưa có dữ liệu "
                    "đã xử lý</div>", unsafe_allow_html=True)
        if li is not None and ui.button("Dùng LinkedIn", key="ws_li"):
            pick = ("li",)
    with c2:
        st.markdown(f"<div class='card'><h4>Hacker News (2018–2023)</h4>{len(hn):,} tin theo <b>tháng</b> · chuỗi thời gian 66 kỳ "
                    "(phù hợp ARIMA/SARIMA)</div>" if hn is not None else "<div class='card'><h4>Hacker News</h4>Chưa có dữ liệu</div>",
                    unsafe_allow_html=True)
        if hn is not None and ui.button("Dùng Hacker News", key="ws_hn"):
            pick = ("hn",)
    with c3:
        st.markdown("<div class='card'><h4>Việt Nam (2022–2026)</h4>9.999 tin thật lấy mẫu phân tầng từ 606.878 tin · lương VND dạng "
                    "chuỗi, địa điểm, kinh nghiệm, mô tả tiếng Việt</div>", unsafe_allow_html=True)
        if SAMPLE_DATA.exists() and ui.button("Dùng dữ liệu Việt Nam", key="btn_sample"):
            pick = ("vn",)
    if pick:
        ok = False
        try:
            if pick[0] == "li":
                _load_frame(li, "LinkedIn CNTT (Kaggle, đã xử lý)", f"kaggle_li_{len(li)}", "listed_date")
            elif pick[0] == "hn":
                _load_frame(hn, "Hacker News Who is hiring (Kaggle, đã xử lý)", f"kaggle_hn_{len(hn)}", "month")
            else:
                _load(SAMPLE_DATA.read_bytes(), SAMPLE_DATA.name, None)
            ok = True
        except Exception as e:  # noqa: BLE001
            st.error(f"Không nạp được dữ liệu: {e}")
            log_error("upload", e)
        if ok:
            st.toast("Đã chọn bộ dữ liệu phân tích")
            st.rerun()


def _load(raw: bytes, name: str, max_rows: int | None):
    ss = st.session_state
    with st.spinner("Đang đọc và kiểm tra dữ liệu..."):
        res = loader.load_file(raw, name, max_rows=max_rows)
    state.set_dataset(ss, res.df, res.meta)
    ss["load_warnings"] = res.warnings
    tc = schema.time_candidates(res.df)
    ss["time_col"] = tc[0] if tc else None
    register_dataset(res.meta)
    log_activity("upload", "Tải dataset", f"{name}: {res.df.shape[0]} dòng x {res.df.shape[1]} cột, encoding={res.encoding}",
                 res.dataset_id)


def _load_frame(df, name: str, dataset_id: str, time_col: str | None = None):
    """Nạp một DataFrame có sẵn (LinkedIn/HN đã xử lý) làm bộ dữ liệu phân tích."""
    ss = st.session_state
    meta = {"dataset_id": dataset_id, "name": name, "size_bytes": int(df.memory_usage(deep=False).sum()),
            "encoding": "đã xử lý", "delimiter": "-", "n_rows": int(len(df)), "n_cols": int(df.shape[1])}
    state.set_dataset(ss, df.copy(), meta)
    ss["load_warnings"] = []
    ss["time_col"] = time_col
    register_dataset(meta)
    log_activity("upload", "Chọn dataset có sẵn", f"{name}: {len(df)} dòng x {df.shape[1]} cột", dataset_id)


def render():
    ss = st.session_state
    ui.page_header("Bộ dữ liệu phân tích", "Chọn bộ dữ liệu cho các module phân tích tổng quát (Dashboard, Làm sạch tương tác, EDA tự động, "
                   "Khai phá, Machine Learning, Dự báo tổng quát, Bất thường, Báo cáo PDF, Trợ lý AI): LinkedIn, Hacker News, "
                   "dữ liệu Việt Nam hoặc file CSV/Parquet bất kỳ.", "upload")
    _builtin_sources()

    c1, c2 = st.columns([3, 2])
    with c1:
        ui.section("Hoặc tải file của bạn", f"CSV (UTF-8 / UTF-8 BOM / CP1258), dấu phân cách , ; tab | - tối đa {MAX_UPLOAD_MB} MB")
        up = st.file_uploader("Chọn file", type=["csv", "txt", "tsv", "parquet"], label_visibility="collapsed")
        limit = st.number_input("Giới hạn số dòng đọc (0 = đọc toàn bộ)", min_value=0, value=0, step=10_000,
                                help="Dùng khi file rất lớn để thao tác nhanh hơn. Kết quả phân tích khi đó chỉ đại diện cho phần dữ liệu đã đọc.")
        if up is not None:
            raw = up.getvalue()
            new_id = loader.fingerprint(raw, up.name)
            st.caption(f"File: **{up.name}** · Kích thước: **{loader.human_size(len(raw))}**")
            same = new_id == ss.get("dataset_id")
            label = "Dataset này đang được sử dụng" if same else ("Thay thế dataset hiện tại" if state.has_data(ss) else "Nạp dataset")
            if state.has_data(ss) and not same:
                ui.callout("Nạp file mới sẽ <b>thay thế</b> dataset hiện tại và xóa toàn bộ kết quả phân tích cũ "
                           "(tránh dùng nhầm kết quả của dataset trước).", "warn")
            if ui.button(label, key="btn_load", primary=True, disabled=same):
                loaded = False
                try:
                    _load(raw, up.name, limit or None)
                    loaded = True
                except loader.DataLoadError as e:
                    st.error(f"File không hợp lệ: {e}")
                    log_error("upload", str(e))
                except Exception as e:  # noqa: BLE001
                    st.error(f"Không đọc được dữ liệu: {e}")
                    log_error("upload", e)
                if loaded:
                    st.toast("Đã nạp dataset thành công")
                    st.rerun()
    with c2:
        ui.section("Lưu ý", "")
        st.markdown("<div class='card'>Hệ thống nhận diện schema tự động, <b>không cố định tên cột</b>: file CSV về tuyển dụng, môi trường, "
                    "bán hàng… đều phân tích được bằng các module tổng quát. Module <b>Thị trường tuyển dụng (VN)</b> tự ánh xạ cột "
                    "tiêu đề/lương/địa điểm/kinh nghiệm cho dữ liệu tin tuyển dụng.</div>", unsafe_allow_html=True)

    if not state.has_data(ss):
        ui.empty_state("Chưa có dataset", "Kéo thả file CSV vào khung phía trên hoặc dùng dataset mẫu.", "upload", go_upload=False)
        return

    df, meta = ss["df"], ss["meta"]
    for w in ss.get("load_warnings") or []:
        ui.callout(w, "warn")
    ui.section("Thông tin file")
    ui.kpi_cards([
        {"label": "Tên file", "value": f"<span style='font-size:1rem'>{meta['name']}</span>", "icon": "report", "accent": NAVY},
        {"label": "Kích thước", "value": loader.human_size(meta["size_bytes"]), "icon": "upload", "accent": BLUE},
        {"label": "Số dòng × cột", "value": f"{len(df):,} × {df.shape[1]}", "icon": "rows", "accent": TEAL,
         "hint": "Dữ liệu hiện tại (sau làm sạch nếu có)"},
        {"label": "Encoding · Phân cách", "value": f"<span style='font-size:1rem'>{meta['encoding']} · {loader.delimiter_label(meta['delimiter'])}</span>",
         "icon": "code", "accent": NAVY},
    ])

    if is_job_dataset(df) and "is_it" not in df.columns:
        ui.callout("Hệ thống nhận diện đây là <b>dữ liệu tin tuyển dụng</b>. Sau khi làm sạch, hãy vào module "
                   "<b>Thị trường tuyển dụng (VN)</b> để chuẩn hóa lương, kinh nghiệm, địa điểm và trích xuất kỹ năng.", "ok")
        if nav.PAGES.get("jobs") and ui.button("Mở module Thị trường tuyển dụng (VN)", key="go_jobs", full=False):
            st.switch_page(nav.PAGES["jobs"])

    t1, t2, t3 = st.tabs([f"{PREVIEW_ROWS} dòng đầu tiên", "Schema & kiểu dữ liệu", "Cấu hình vai trò cột"])
    with t1:
        ui.show_df(df.head(PREVIEW_ROWS), height=520, hide_index=False)
    with t2:
        prof = schema.profile_frame(df).drop(columns="role")
        prof.columns = ["Cột", "Kiểu pandas", "Vai trò nhận diện", "Số ô thiếu", "Thiếu (%)", "Số giá trị khác nhau", "Ví dụ"]
        ui.show_df(prof, height=min(600, 40 + 35 * len(prof)))
        st.caption("Vai trò được suy luận tự động từ nội dung (kiểu, số giá trị khác nhau, độ dài chuỗi, khả năng chuyển sang thời gian) "
                   "kết hợp gợi ý từ tên cột - không cố định bất kỳ tên cột nào.")
    with t3:
        tc = schema.time_candidates(df)
        opts = ["(Không có)"] + list(df.columns)
        cur_t = ss.get("time_col")
        idx = opts.index(cur_t) if cur_t in opts else (opts.index(tc[0]) if tc else 0)
        a, b = st.columns(2)
        new_t = a.selectbox("Cột thời gian", opts, index=idx, help=f"Gợi ý tự động: {', '.join(tc) if tc else 'không phát hiện'}")
        targets = schema.target_candidates(df)
        topts = ["(Chưa chọn)"] + targets
        cur_y = ss.get("target_col")
        new_y = b.selectbox("Biến mục tiêu (cho Machine Learning)", topts, index=topts.index(cur_y) if cur_y in topts else 0)
        ss["time_col"] = None if new_t == "(Không có)" else new_t
        ss["target_col"] = None if new_y == "(Chưa chọn)" else new_y
        if ss["time_col"]:
            parsed = schema.to_time_index(df[ss["time_col"]])
            ok = parsed.notna().mean() * 100
            msg = f"Cột <b>{ss['time_col']}</b>: {ok:.1f}% giá trị chuyển được sang thời gian"
            if parsed.notna().any():
                msg += f" (từ {parsed.min():%d/%m/%Y} đến {parsed.max():%d/%m/%Y})."
            ui.callout(msg, "ok" if ok > 90 else "warn")
        ui.section("Danh sách cột theo vai trò")
        groups = schema.roles(df)
        for role, cols in groups.items():
            st.markdown(f"**{schema.ROLE_LABELS.get(role, role)}** ({len(cols)}): " + ", ".join(f"`{c}`" for c in cols))
