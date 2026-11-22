"""Kiểm thử giao diện tự động bằng streamlit.testing.AppTest.

AppTest chạy app.py ngay trong tiến trình pytest (không cần trình duyệt), rồi chuyển lần lượt tới
từng trang trong menu. Mỗi trang phải chạy hết mà không có ngoại lệ và không hiện hộp lỗi
"Đã xảy ra lỗi" (hộp này do src/analytics/ui/runner.py hiện ra khi một trang bị lỗi).

Có hai lượt:
1. Phiên mới, chưa chọn bộ dữ liệu phân tích (giống người vừa mở link).
2. Đã chọn sẵn bộ dữ liệu Việt Nam mẫu (data/vn/sample_it_jobs.csv), để các trang phân tích
   tổng quát vẽ đầy đủ thay vì chỉ báo "chưa chọn bộ dữ liệu".

Ngoài ra kiểm tra hai tính năng an toàn khi đưa lên web: APP_PUBLIC=1 và giới hạn câu hỏi Gemini.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")
TIMEOUT = 180  # giây cho mỗi lần chạy trang (trang Dự báo/Lương huấn luyện mô hình khi mở lần đầu)

# Lấy danh sách trang trực tiếp từ PAGE_DEFS trong app.py, để thêm/bớt trang thì test tự cập nhật
PAGES = re.findall(r'"(views/[a-z_]+\.py)"', (ROOT / "app.py").read_text(encoding="utf-8"))

# Biến môi trường có thể làm kết quả test khác nhau giữa các máy
_ENV_KEYS = ("APP_PUBLIC", "GEMINI_MAX_QUESTIONS")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


def _errors(at: AppTest) -> list[str]:
    """Các thông báo lỗi do app tự bắt được (runner.run_page) hoặc ngoại lệ chưa bắt."""
    msgs = [e.value for e in at.exception]
    msgs += [e.value for e in at.error if str(e.value).startswith("Đã xảy ra lỗi")]
    return msgs


def _new_app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not _errors(at), _errors(at)
    return at


def _load_sample_dataset(at: AppTest) -> None:
    from src.analytics import state
    from src.analytics.data import loader, schema

    path = ROOT / "data" / "vn" / "sample_it_jobs.csv"
    res = loader.load_file(path.read_bytes(), path.name, max_rows=3000)
    ss: dict = {}
    state.init_state(ss)
    state.set_dataset(ss, res.df, res.meta)
    tc = schema.time_candidates(res.df)
    ss["time_col"] = tc[0] if tc else None
    for k, v in ss.items():
        at.session_state[k] = v


@pytest.fixture(scope="module")
def app_fresh():
    return _new_app()


@pytest.fixture(scope="module")
def app_with_dataset():
    at = _new_app()
    _load_sample_dataset(at)
    return at


def test_menu_co_du_20_trang():
    assert len(PAGES) == 20
    assert len(set(PAGES)) == 20
    for p in PAGES:
        assert (ROOT / p).is_file(), p


@pytest.mark.parametrize("page", PAGES)
def test_mo_trang_khong_loi(app_fresh, page):
    at = app_fresh.switch_page(page).run()
    assert not _errors(at), f"{page}: {_errors(at)}"


@pytest.mark.parametrize("page", PAGES)
def test_mo_trang_khi_da_chon_dataset(app_with_dataset, page):
    at = app_with_dataset.switch_page(page).run()
    assert not _errors(at), f"{page}: {_errors(at)}"
    assert at.session_state["df"] is not None


# ------------------------------------------------------------------ APP_PUBLIC
def _tab_labels(at: AppTest) -> list[str]:
    return [t.label for t in at.tabs]


def test_trang_he_thong_ban_noi_bo_hien_nhat_ky():
    at = _new_app().switch_page("views/system.py").run()
    assert not _errors(at)
    labels = _tab_labels(at)
    assert "Lịch sử hoạt động" in labels and "Nhật ký lỗi" in labels


def test_trang_he_thong_ban_web_an_nhat_ky(monkeypatch):
    monkeypatch.setenv("APP_PUBLIC", "1")
    at = _new_app().switch_page("views/system.py").run()
    assert not _errors(at)
    labels = _tab_labels(at)
    assert labels == ["Trạng thái module", "Thư viện"]
    page_text = " ".join(str(m.value) for m in at.markdown)
    assert "APP_PUBLIC=1" in page_text


@pytest.mark.parametrize("value,expected", [("1", True), ("true", True), ("ON", True), ("0", False), ("", False)])
def test_is_public(monkeypatch, value, expected):
    from src.config import is_public
    monkeypatch.setenv("APP_PUBLIC", value)
    assert is_public() is expected


# ------------------------------------------------------------------ Giới hạn câu hỏi Gemini
def test_gioi_han_mac_dinh(monkeypatch):
    from src.config import DEFAULT_PUBLIC_CHAT_LIMIT, get_chat_limit
    assert get_chat_limit() == 0                      # chạy trên máy: không giới hạn
    monkeypatch.setenv("APP_PUBLIC", "1")
    assert get_chat_limit() == DEFAULT_PUBLIC_CHAT_LIMIT
    monkeypatch.setenv("GEMINI_MAX_QUESTIONS", "5")
    assert get_chat_limit() == 5
    monkeypatch.setenv("GEMINI_MAX_QUESTIONS", "abc")  # giá trị sai: quay về mặc định
    assert get_chat_limit() == DEFAULT_PUBLIC_CHAT_LIMIT


def test_bo_dem_cau_hoi():
    from src import chat_quota
    ss: dict = {}
    assert chat_quota.remaining(ss, 0) is None        # 0 = không giới hạn
    assert chat_quota.remaining(ss, 2) == 2
    chat_quota.register_question(ss)
    chat_quota.register_question(ss)
    assert chat_quota.remaining(ss, 2) == 0 and not chat_quota.can_ask(ss, 2)
    ss["gemini_key_override"] = "key-rieng-cua-nguoi-dung"   # dùng key riêng thì không bị giới hạn
    assert chat_quota.can_ask(ss, 2)
    chat_quota.register_question(ss)
    assert chat_quota.used(ss) == 2


def test_chat_dung_goi_gemini_khi_het_luot(monkeypatch):
    """Giới hạn 2 câu: 2 câu đầu gọi Gemini, câu thứ 3 không gọi nữa và trang báo hết lượt."""
    from src import chatbot

    calls = []

    def fake_ask(api_key, model, msgs, prompt, ctx, **kw):
        calls.append(prompt)
        return f"Trả lời giả cho: {prompt}"

    monkeypatch.setattr(chatbot, "ask_gemini", fake_ask)
    monkeypatch.setenv("GEMINI_API_KEY", "khoa-gia-de-test")
    monkeypatch.setenv("GEMINI_MAX_QUESTIONS", "2")

    at = _new_app().switch_page("views/chat.py").run()
    for q in ["câu 1", "câu 2", "câu 3"]:
        at.chat_input[0].set_value(q).run()
        assert not _errors(at), _errors(at)

    assert calls == ["câu 1", "câu 2"]
    assert at.session_state["gemini_questions_used"] == 2
    assert any("hết 2 câu hỏi Gemini" in str(w.value) for w in at.warning)


def test_chat_khong_tru_luot_khi_gemini_loi(monkeypatch):
    """Gemini lỗi (vd. model không tồn tại) thì người dùng không bị mất lượt hỏi."""
    from src import chatbot

    def failing_ask(*a, **kw):
        raise chatbot.GeminiError("Không tìm thấy model 'gemini-2.5-flash'")

    monkeypatch.setattr(chatbot, "ask_gemini", failing_ask)
    monkeypatch.setenv("GEMINI_API_KEY", "khoa-gia-de-test")
    monkeypatch.setenv("GEMINI_MAX_QUESTIONS", "2")

    at = _new_app().switch_page("views/chat.py").run()
    at.chat_input[0].set_value("câu lỗi").run()
    assert not at.exception
    used = at.session_state["gemini_questions_used"] if "gemini_questions_used" in at.session_state else 0
    assert used == 0
    assert any("Không tìm thấy model" in str(e.value) for e in at.error)
