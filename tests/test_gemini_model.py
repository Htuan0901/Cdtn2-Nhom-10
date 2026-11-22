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


@pytest.fixture(autouse=True)
def _clear_cache():
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
