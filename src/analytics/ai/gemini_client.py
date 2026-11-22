"""Kết nối Google Gemini qua thư viện chính thức google-genai.

- API Key đọc từ biến môi trường / .env (không bao giờ ghi trong mã, không log, không hiển thị).
- Chuẩn hóa lỗi: chưa cấu hình key, key sai, vượt giới hạn (429), lỗi máy chủ, lỗi mạng, timeout.
- Tìm kiếm web dùng công cụ Google Search Grounding chính thức của Gemini và trả về danh sách nguồn.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.analytics.config import AI_TIMEOUT_SECONDS, get_gemini_key, get_gemini_model
from src.analytics.logger import get_logger, redact

log = get_logger("gemini")


class GeminiError(Exception):
    def __init__(self, user_message: str, kind: str = "error"):
        super().__init__(user_message)
        self.user_message = user_message
        self.kind = kind


@dataclass
class GeminiReply:
    text: str
    sources: list[dict] = field(default_factory=list)
    function_calls: list[str] = field(default_factory=list)


def sdk_available() -> bool:
    try:
        from google import genai  # noqa: F401
        return True
    except Exception:
        return False


def _map_error(exc: Exception) -> GeminiError:
    name = exc.__class__.__name__
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    msg = redact(str(exc))
    log.warning("Gemini error %s (%s): %s", name, code, msg[:300])
    if code in (401, 403) or "API_KEY_INVALID" in msg or "API key not valid" in msg:
        return GeminiError("API Key không hợp lệ hoặc không có quyền truy cập model. Hãy kiểm tra GEMINI_API_KEY trong file .env.", "auth")
    if code == 429 or "RESOURCE_EXHAUSTED" in msg:
        return GeminiError("Đã vượt giới hạn gọi API của Gemini (rate limit/quota). Vui lòng đợi ít phút rồi thử lại.", "rate_limit")
    if is_model_not_found(exc):
        return GeminiError("Không tìm thấy model đã cấu hình (GEMINI_MODEL) và không tự chọn được model thay thế. "
                           f"Hãy đặt GEMINI_MODEL bằng một model có trong Google AI Studio. Chi tiết: {msg[:160]}", "model")
    if is_overloaded(exc):
        return GeminiError("Gemini đang quá tải phía Google (503), đã thử cả model dự phòng. Vui lòng đợi 1–2 phút rồi hỏi lại.", "server")
    if code and int(code) >= 500:
        return GeminiError("Máy chủ Gemini đang gặp sự cố tạm thời. Vui lòng thử lại sau.", "server")
    if "Timeout" in name or "timeout" in msg.lower():
        return GeminiError("Hết thời gian chờ phản hồi từ Gemini. Vui lòng thử lại.", "timeout")
    if "Connect" in name or "Network" in name or "getaddrinfo" in msg or "ConnectionError" in name:
        return GeminiError("Không kết nối được tới Gemini. Hãy kiểm tra kết nối Internet.", "network")
    return GeminiError(f"Gemini trả về lỗi: {msg[:200]}", "error")


# ---------------------------------------------------------------- Tự chọn model khi model cấu hình không còn
# Google định kỳ ngừng các model cũ (vd. gemini-2.0-flash, rồi gemini-2.5-flash). Khi đó API trả 404 NOT_FOUND.
# Thay vì báo lỗi, app hỏi API danh sách model mà key này dùng được rồi chọn model "flash" mới nhất.
_RESOLVED: dict[str, str] = {}      # model cấu hình -> model thay thế đã chọn (dùng chung trong tiến trình)
_SKIP_WORDS = ("embedding", "image", "imagen", "tts", "audio", "live", "aqa", "vision", "robotics", "computer-use",
               "learnlm", "gemma", "veo", "native")


def is_model_not_found(exc: Exception) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    msg = str(exc)
    return code == 404 or "NOT_FOUND" in msg or "is not found" in msg


def is_overloaded(exc: Exception) -> bool:
    """Model đang quá tải phía Google (503 UNAVAILABLE / high demand): lỗi tạm thời, nên thử model khác."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    msg = str(exc)
    return code == 503 or "UNAVAILABLE" in msg or "high demand" in msg or "overloaded" in msg.lower()


def _version(name: str) -> float:
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else 0.0


def choose_fallback_model(names: list[str], wanted: str = "") -> str | None:
    """Chọn model thay thế từ danh sách tên model (bỏ tiền tố "models/").

    Thứ tự ưu tiên: alias gemini-flash-latest → flash bản ổn định mới nhất → flash-lite → pro → bản preview.
    """
    clean = [n.split("/", 1)[-1] for n in names if n]
    clean = [n for n in clean if n.startswith("gemini") and not any(w in n for w in _SKIP_WORDS) and n != wanted]
    if not clean:
        return None
    if "gemini-flash-latest" in clean:
        return "gemini-flash-latest"

    def score(n: str):
        unstable = any(w in n for w in ("preview", "exp", "thinking"))
        if "flash" in n and "lite" not in n:
            fam = 0
        elif "flash-lite" in n:
            fam = 1
        elif "pro" in n:
            fam = 2
        else:
            fam = 3
        return (unstable, fam, -_version(n), "latest" not in n, len(n))

    return sorted(clean, key=score)[0]


def list_generate_models(client) -> list[str]:
    """Tên các model dùng được với generateContent cho key hiện tại."""
    out = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", None) or []
        if not actions or "generateContent" in actions:
            out.append(getattr(m, "name", "") or "")
    return out


def call_with_model_fallback(client, model: str, fn):
    """Gọi fn(model). Nếu model không tồn tại (404) hoặc đang quá tải (503) thì tự chọn model thay thế
    và gọi lại đúng 1 lần.

    Trả về (kết quả, model_đã_dùng). Ngoại lệ gốc được ném lại nếu không tìm được model thay thế.
    Model thay thế chỉ được nhớ cho lần sau khi model cũ không còn (404); quá tải là tạm thời nên không nhớ.
    """
    use = _RESOLVED.get(model, model)
    try:
        return fn(use), use
    except Exception as exc:  # noqa: BLE001
        missing = is_model_not_found(exc)
        if not (missing or is_overloaded(exc)):
            raise
        try:
            alt = choose_fallback_model(list_generate_models(client), wanted=use)
        except Exception:  # noqa: BLE001
            alt = None
        if not alt:
            raise exc
        log.warning("Model %s %s, chuyển sang %s", use, "không còn" if missing else "quá tải", alt)
        result = fn(alt)
        if missing:
            _RESOLVED[model] = alt
        return result, alt


class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key if api_key is not None else get_gemini_key()
        self.model = model or get_gemini_model()
        self._client = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _get(self):
        if not self.api_key:
            raise GeminiError("Chưa cấu hình GEMINI_API_KEY. Hãy tạo file .env (xem .env.example) và nhập API Key.", "no_key")
        if self._client is None:
            try:
                from google import genai
                from google.genai import types
            except Exception:
                raise GeminiError("Chưa cài thư viện google-genai (pip install google-genai).", "sdk")
            try:
                self._client = genai.Client(api_key=self.api_key,
                                            http_options=types.HttpOptions(timeout=AI_TIMEOUT_SECONDS * 1000))
            except Exception as exc:
                raise _map_error(exc)
        return self._client

    def generate(self, contents: list[dict], system: str, functions: list | None = None,
                 temperature: float = 0.3, max_calls: int = 6) -> GeminiReply:
        client = self._get()
        from google.genai import types

        cfg = dict(system_instruction=system, temperature=temperature)
        if functions:
            cfg["tools"] = functions
            cfg["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(maximum_remote_calls=max_calls)
        try:
            resp, self.model = call_with_model_fallback(
                client, self.model,
                lambda m: client.models.generate_content(model=m, contents=contents, config=types.GenerateContentConfig(**cfg)))
        except GeminiError:
            raise
        except Exception as exc:
            raise _map_error(exc)
        calls = []
        for item in (getattr(resp, "automatic_function_calling_history", None) or []):
            for part in (getattr(item, "parts", None) or []):
                fc = getattr(part, "function_call", None)
                if fc is not None and getattr(fc, "name", None):
                    calls.append(fc.name)
        text = _safe_text(resp)
        if not text:
            raise GeminiError("Gemini không trả về nội dung (có thể bị chặn bởi bộ lọc an toàn). Hãy diễn đạt lại câu hỏi.", "empty")
        return GeminiReply(text=text, function_calls=calls)

    def generate_with_search(self, contents: list[dict], system: str, temperature: float = 0.3) -> GeminiReply:
        client = self._get()
        from google.genai import types

        cfg = types.GenerateContentConfig(system_instruction=system, temperature=temperature,
                                          tools=[types.Tool(google_search=types.GoogleSearch())])
        try:
            resp, self.model = call_with_model_fallback(
                client, self.model, lambda m: client.models.generate_content(model=m, contents=contents, config=cfg))
        except Exception as exc:
            raise _map_error(exc)
        text = _safe_text(resp)
        if not text:
            raise GeminiError("Gemini không trả về nội dung.", "empty")
        return GeminiReply(text=text, sources=extract_sources(resp))

    def ping(self) -> tuple[bool, str]:
        try:
            r = self.generate([{"role": "user", "parts": [{"text": "Trả lời đúng một từ: OK"}]}], "Bạn là trợ lý kiểm tra kết nối.")
            return True, f"Kết nối thành công (model: {self.model}). Phản hồi: {r.text.strip()[:30]}"
        except GeminiError as e:
            return False, e.user_message


def _safe_text(resp) -> str:
    try:
        t = resp.text
        if t:
            return t.strip()
    except Exception:
        pass
    try:
        parts = resp.candidates[0].content.parts
        return "".join(getattr(p, "text", "") or "" for p in parts).strip()
    except Exception:
        return ""


def extract_sources(resp) -> list[dict]:
    out, seen = [], set()
    try:
        for cand in resp.candidates or []:
            gm = getattr(cand, "grounding_metadata", None)
            for ch in (getattr(gm, "grounding_chunks", None) or []):
                web = getattr(ch, "web", None)
                uri = getattr(web, "uri", None)
                if uri and uri not in seen:
                    seen.add(uri)
                    out.append({"title": getattr(web, "title", None) or uri, "uri": uri})
    except Exception:
        pass
    return out
