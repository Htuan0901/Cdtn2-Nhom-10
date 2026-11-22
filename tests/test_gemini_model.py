"""Tự chọn model thay thế khi GEMINI_MODEL đã bị Google ngừng (API trả 404 NOT_FOUND).

Không gọi mạng: dùng client giả có models.list() và models.generate_content().
"""
from types import SimpleNamespace

import pytest

from src.analytics.ai import gemini_client as gc


class NotFound(Exception):
    code = 404


class FakeModels:
    def __init__(self, available):
        self.available = available
        self.calls = []

    def list(self):
        return [SimpleNamespace(name=f"models/{n}", supported_actions=["generateContent"]) for n in self.available]

    def generate_content(self, model, contents, config=None):
        self.calls.append(model)
        if model not in self.available:
            raise NotFound(f"404 NOT_FOUND. models/{model} is not found for API version v1beta")
        return SimpleNamespace(text=f"OK từ {model}", candidates=[], automatic_function_calling_history=[])


SLEPT: list[float] = []


@pytest.fixture(autouse=True)
def _clear_cache(monkeypatch):
    SLEPT.clear()
    monkeypatch.setattr(gc, "_sleep", SLEPT.append)   # không chờ thật khi test
    gc._RESOLVED.clear()
    yield
    gc._RESOLVED.clear()


@pytest.mark.parametrize("names,expected", [
    (["gemini-3.0-flash", "gemini-2.5-flash-lite", "gemini-3.0-pro", "text-embedding-004"], "gemini-3.0-flash"),
    (["gemini-flash-latest", "gemini-3.0-flash"], "gemini-flash-latest"),
    (["gemini-3.1-flash-preview", "gemini-3.0-flash"], "gemini-3.0-flash"),          # ưu tiên bản ổn định
    (["gemini-3.0-flash-lite", "gemini-3.0-pro"], "gemini-3.0-flash-lite"),
    (["gemini-3.0-flash-image", "gemini-embedding-001"], None),                       # không có model văn bản
])
def test_choose_fallback_model(names, expected):
    assert gc.choose_fallback_model([f"models/{n}" for n in names], wanted="gemini-2.5-flash") == expected


def test_generate_tu_chuyen_model_khi_404(monkeypatch):
    fake = SimpleNamespace(models=FakeModels(["gemini-3.0-flash", "gemini-3.0-pro"]))
    client = gc.GeminiClient(api_key="khoa-gia", model="gemini-2.5-flash")
    monkeypatch.setattr(client, "_get", lambda: fake)
    reply = client.generate([{"role": "user", "parts": [{"text": "xin chào"}]}], "system")
    assert reply.text == "OK từ gemini-3.0-flash"
    assert client.model == "gemini-3.0-flash"
    assert fake.models.calls == ["gemini-2.5-flash", "gemini-3.0-flash"]
    # Lần sau dùng luôn model đã chọn, không gọi thử model cũ nữa
    client2 = gc.GeminiClient(api_key="khoa-gia", model="gemini-2.5-flash")
    monkeypatch.setattr(client2, "_get", lambda: fake)
    client2.generate([{"role": "user", "parts": [{"text": "lần 2"}]}], "system")
    assert fake.models.calls[-1] == "gemini-3.0-flash"


def test_generate_bao_loi_ro_khi_khong_co_model_thay_the(monkeypatch):
    fake = SimpleNamespace(models=FakeModels([]))
    client = gc.GeminiClient(api_key="khoa-gia", model="gemini-2.5-flash")
    monkeypatch.setattr(client, "_get", lambda: fake)
    with pytest.raises(gc.GeminiError) as ei:
        client.generate([{"role": "user", "parts": [{"text": "x"}]}], "system")
    assert ei.value.kind == "model"
    assert "GEMINI_MODEL" in ei.value.user_message


def test_loi_khac_404_khong_doi_model():
    class Boom(Exception):
        code = 429

    def fn(m):
        raise Boom("RESOURCE_EXHAUSTED")

    fake = SimpleNamespace(models=FakeModels(["gemini-3.0-flash"]))
    with pytest.raises(Boom):
        gc.call_with_model_fallback(fake, "gemini-2.5-flash", fn)
    assert gc._RESOLVED == {}


# ---------------------------------------------------------------- 429 do giới hạn lượt gọi mỗi phút (RPM)
class RateLimited(Exception):
    code = 429


def test_429_theo_phut_cho_roi_thu_lai_cung_model():
    tried = []

    def fn(m):
        tried.append(m)
        if len(tried) == 1:
            raise RateLimited("429 RESOURCE_EXHAUSTED GenerateRequestsPerMinutePerProjectPerModel "
                              "{'retryDelay': '7s'}")
        return f"OK từ {m}"

    fake = SimpleNamespace(models=FakeModels(["gemini-flash-latest", "gemini-3.0-flash-lite"]))
    result, used = gc.call_with_model_fallback(fake, "gemini-flash-latest", fn)
    assert result == "OK từ gemini-flash-latest" and used == "gemini-flash-latest"
    assert tried == ["gemini-flash-latest", "gemini-flash-latest"]
    assert SLEPT == [7.0]


def test_429_van_loi_sau_khi_thu_lai_thi_bao_loi_chi_2_lan_goi():
    tried = []

    def fn(m):
        tried.append(m)
        raise RateLimited("429 RESOURCE_EXHAUSTED PerMinute")

    fake = SimpleNamespace(models=FakeModels(["gemini-flash-latest"]))
    with pytest.raises(RateLimited):
        gc.call_with_model_fallback(fake, "gemini-flash-latest", fn)
    assert len(tried) == 2 and SLEPT == [gc.RATE_LIMIT_DEFAULT_WAIT]
    assert "mỗi phút" in gc._map_error(RateLimited("429 RESOURCE_EXHAUSTED")).user_message


def test_429_het_quota_ngay_khong_cho():
    tried = []

    def fn(m):
        tried.append(m)
        raise RateLimited("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier")

    fake = SimpleNamespace(models=FakeModels(["gemini-flash-latest"]))
    with pytest.raises(RateLimited):
        gc.call_with_model_fallback(fake, "gemini-flash-latest", fn)
    assert len(tried) == 1 and SLEPT == []
    assert "trong ngày" in gc._map_error(RateLimited("PerDay quota")).user_message


@pytest.mark.parametrize("msg,expected", [("'retryDelay': '12s'", 12.0), ('"retryDelay": "3.5s"', 3.5),
                                          ("'retryDelay': '59s'", gc.RATE_LIMIT_MAX_WAIT), ("không có", 10.0)])
def test_doc_retry_delay(msg, expected):
    assert gc.retry_delay_seconds(Exception(msg)) == expected


def test_gioi_han_vong_goi_cong_cu():
    """Gói miễn phí 5 lượt/phút: mỗi câu hỏi tối đa 1 lượt hỏi + 3 vòng công cụ."""
    assert gc.MAX_FUNCTION_CALLS == 3


class Overloaded(Exception):
    code = 503


def test_qua_tai_503_thu_model_khac_nhung_khong_nho():
    """503 high demand: thử model dự phòng 1 lần; lần sau vẫn thử lại model cấu hình (quá tải là tạm thời)."""
    fake = SimpleNamespace(models=FakeModels(["gemini-2.5-flash", "gemini-2.5-flash-lite"]))
    tried = []

    def fn(m):
        tried.append(m)
        if m == "gemini-2.5-flash":
            raise Overloaded("503 UNAVAILABLE. This model is currently experiencing high demand.")
        return f"OK {m}"

    result, used = gc.call_with_model_fallback(fake, "gemini-2.5-flash", fn)
    assert (result, used) == ("OK gemini-2.5-flash-lite", "gemini-2.5-flash-lite")
    assert tried == ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
    assert gc._RESOLVED == {}


def test_qua_tai_ca_model_du_phong_bao_loi_than_thien(monkeypatch):
    class AllBusy(FakeModels):
        def generate_content(self, model, contents, config=None):
            self.calls.append(model)
            raise Overloaded("503 UNAVAILABLE high demand")

    fake = SimpleNamespace(models=AllBusy(["gemini-2.5-flash", "gemini-2.5-flash-lite"]))
    client = gc.GeminiClient(api_key="khoa-gia", model="gemini-2.5-flash")
    monkeypatch.setattr(client, "_get", lambda: fake)
    with pytest.raises(gc.GeminiError) as ei:
        client.generate([{"role": "user", "parts": [{"text": "x"}]}], "system")
    assert "quá tải" in ei.value.user_message
    assert len(fake.models.calls) == 2      # model cấu hình + đúng 1 model dự phòng, không lặp vô hạn


def test_model_mac_dinh_la_alias_moi_nhat(monkeypatch):
    """Không đặt GEMINI_MODEL thì dùng alias gemini-flash-latest (Google tự trỏ tới Flash mới nhất)."""
    from src.analytics.config import DEFAULT_GEMINI_MODEL
    from src.config import get_gemini_model

    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert get_gemini_model() == "gemini-flash-latest" == DEFAULT_GEMINI_MODEL
    monkeypatch.setenv("GEMINI_MODEL", "gemini-pro-latest")
    assert get_gemini_model() == "gemini-pro-latest"
