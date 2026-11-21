"""Data Analysis Agent: kết hợp Router + công cụ Python + Gemini.

Luồng xử lý một câu hỏi:
  1. Router xác định loại câu hỏi (dữ liệu / kiến thức chung / thời sự).
  2. Dữ liệu: gọi các hàm Python tính toán chính xác → JSON kết quả (giới hạn kích thước).
  3. Gemini diễn giải kết quả (được phép gọi thêm công cụ qua function calling nếu cần).
     Nếu chưa có API Key ⇒ trả lời trực tiếp từ kết quả tính toán (chế độ offline).
  4. Kiến thức chung → Gemini trả lời. Thời sự → Gemini + Google Search grounding, kèm nguồn.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from src.analytics.ai.gemini_client import GeminiClient, GeminiError
from src.analytics.ai.router import Route, route
from src.analytics.ai.tools import DataTools
from src.analytics.config import AI_MAX_CONTEXT_CHARS, AI_MAX_HISTORY_TURNS, APP_NAME
from src.analytics.logger import get_logger

log = get_logger("agent")

SYSTEM_DATA = f"""Bạn là trợ lý phân tích dữ liệu của hệ thống {APP_NAME} (phân tích & dự báo tuyển dụng ngành CNTT).
QUY TẮC BẮT BUỘC:
- Chỉ dùng các con số có trong KẾT QUẢ TÍNH TOÁN (JSON) hoặc kết quả công cụ bạn gọi. TUYỆT ĐỐI không tự tính, ước lượng hay bịa số liệu.
- Nếu kết quả có "available": false hoặc không chứa thông tin được hỏi, nói rõ là dữ liệu không đủ để trả lời và nêu lý do.
- Có thể gọi thêm công cụ nếu cần thông tin khác về dataset.
- Trả lời bằng tiếng Việt, ngắn gọn, có cấu trúc (markdown), nêu đơn vị đo; làm tròn hợp lý.
- Khi nói về tương quan, nhắc rằng tương quan không phải nhân quả. Khi nói về mô hình, không kết luận chỉ dựa trên một chỉ số.
"""
SYSTEM_GENERAL = f"""Bạn là trợ lý AI của hệ thống {APP_NAME}, chuyên về Khoa học dữ liệu, Machine Learning, thống kê và thị trường lao động CNTT.
Trả lời bằng tiếng Việt, chính xác, có ví dụ khi phù hợp, dùng markdown.
Bạn KHÔNG có quyền truy cập Internet trong chế độ này: nếu câu hỏi cần số liệu thời sự/mới nhất, hãy nói rõ giới hạn đó thay vì đoán.
Không bịa số liệu thống kê cụ thể nếu không chắc chắn."""
SYSTEM_WEB = f"""Bạn là trợ lý AI của hệ thống {APP_NAME}. Bạn được cung cấp công cụ Google Search để tra cứu thông tin mới nhất.
Trả lời bằng tiếng Việt, chỉ dựa trên kết quả tìm kiếm, nêu rõ mốc thời gian của thông tin. Nếu nguồn mâu thuẫn, hãy nói rõ."""


@dataclass
class AgentResponse:
    text: str
    kind: str
    tool_calls: list[tuple[str, dict]] = field(default_factory=list)
    tool_results: dict = field(default_factory=dict)
    sources: list[dict] = field(default_factory=list)
    used_llm: bool = False
    error: str | None = None
    llm_function_calls: list[str] = field(default_factory=list)


def _history_contents(history: list[dict]) -> list[dict]:
    out = []
    for m in history[-2 * AI_MAX_HISTORY_TURNS:]:
        role = "user" if m.get("role") == "user" else "model"
        text = str(m.get("content", ""))[:3000]
        if text:
            out.append({"role": role, "parts": [{"text": text}]})
    return out


def _clip(payload: dict, limit: int = AI_MAX_CONTEXT_CHARS) -> str:
    text = json.dumps(payload, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + ' …"[ĐÃ CẮT BỚT do vượt giới hạn ngữ cảnh]"'


def format_offline(results: dict) -> str:
    """Trình bày kết quả tính toán khi không có Gemini (vẫn chính xác vì là số liệu thật)."""
    lines = []
    for name, res in results.items():
        title = name.split("#")[0].replace("_", " ")
        lines.append(f"**{title}**")
        if not res.get("available", True):
            lines.append(f"- Không đủ dữ liệu: {res.get('reason')}")
            continue
        for k, v in res.items():
            if k == "available":
                continue
            if isinstance(v, list):
                lines.append(f"- {k}:")
                for item in v[:10]:
                    lines.append(f"    - {json.dumps(item, ensure_ascii=False, default=str) if isinstance(item, dict) else item}")
            elif isinstance(v, dict):
                lines.append(f"- {k}: {json.dumps(v, ensure_ascii=False, default=str)[:600]}")
            else:
                lines.append(f"- {k}: {v}")
    return "\n".join(lines)


class DataAgent:
    def __init__(self, tools: DataTools, client: GeminiClient | None = None, web_enabled: bool = True):
        self.tools = tools
        self.client = client or GeminiClient()
        self.web_enabled = web_enabled

    def run_tools(self, r: Route) -> dict:
        results = {}
        for i, (name, kwargs) in enumerate(r.calls):
            key = name if name not in results else f"{name}#{i}"
            results[key] = self.tools.call(name, **kwargs)
        return results

    def answer(self, question: str, history: list[dict] | None = None) -> AgentResponse:
        question = (question or "").strip()
        if not question:
            return AgentResponse("Bạn hãy nhập câu hỏi.", "empty")
        history = history or []
        r = route(question, self.tools.df, web_enabled=self.web_enabled)
        log.info("route=%s calls=%s", r.kind, [c[0] for c in r.calls])

        if r.kind == "no_data":
            return AgentResponse("Câu hỏi này cần dữ liệu nhưng hệ thống **chưa có dataset**. Hãy tải file CSV ở mục *Data Upload Center* trước.",
                                 r.kind)

        if r.kind == "data":
            results = self.run_tools(r)
            if not self.client.configured:
                text = ("*Chưa cấu hình GEMINI_API_KEY - dưới đây là kết quả tính toán trực tiếp từ dữ liệu (chưa được AI diễn giải).* \n\n"
                        + format_offline(results))
                return AgentResponse(text, r.kind, r.calls, results, used_llm=False, error="no_key")
            prompt = (f"CÂU HỎI CỦA NGƯỜI DÙNG: {question}\n\n"
                      f"KẾT QUẢ TÍNH TOÁN (JSON, do Python tính trên dataset hiện tại):\n{_clip(results)}\n\n"
                      "Hãy trả lời câu hỏi dựa trên kết quả trên.")
            contents = _history_contents(history) + [{"role": "user", "parts": [{"text": prompt}]}]
            try:
                reply = self.client.generate(contents, SYSTEM_DATA, functions=self.tools.gemini_functions())
                return AgentResponse(reply.text, r.kind, r.calls, results, used_llm=True, llm_function_calls=reply.function_calls)
            except GeminiError as e:
                text = (f"**Không gọi được Gemini:** {e.user_message}\n\nKết quả tính toán trực tiếp từ dữ liệu:\n\n" + format_offline(results))
                return AgentResponse(text, r.kind, r.calls, results, error=e.kind)

        contents = _history_contents(history) + [{"role": "user", "parts": [{"text": question}]}]
        if not self.client.configured:
            return AgentResponse("Câu hỏi kiến thức chung cần Gemini để trả lời, nhưng **chưa cấu hình GEMINI_API_KEY**. "
                                 "Hãy thêm API Key vào file `.env` rồi tải lại trang. (Các câu hỏi về dataset vẫn trả lời được ở chế độ offline.)",
                                 r.kind, error="no_key")
        if r.kind == "web":
            try:
                reply = self.client.generate_with_search(contents, SYSTEM_WEB)
                text = reply.text
                if reply.sources:
                    text += "\n\n**Nguồn tham khảo:**\n" + "\n".join(f"{i + 1}. [{s['title']}]({s['uri']})" for i, s in enumerate(reply.sources[:8]))
                else:
                    text += "\n\n*Lưu ý: Gemini không trả về nguồn tham khảo cụ thể cho câu trả lời này.*"
                return AgentResponse(text, "web", sources=reply.sources, used_llm=True)
            except GeminiError as e:
                log.warning("web search failed: %s", e.kind)
                if e.kind in ("no_key", "auth", "rate_limit", "network"):
                    return AgentResponse(f"**Lỗi:** {e.user_message}", "web", error=e.kind)
                # tìm kiếm thất bại vì lý do khác ⇒ trả lời không có web, nói rõ giới hạn
        try:
            reply = self.client.generate(contents, SYSTEM_GENERAL)
            prefix = "*(Không tra cứu được web - câu trả lời dựa trên kiến thức của mô hình, có thể chưa cập nhật.)*\n\n" if r.kind == "web" else ""
            return AgentResponse(prefix + reply.text, "general", used_llm=True)
        except GeminiError as e:
            return AgentResponse(f"**Lỗi:** {e.user_message}", r.kind, error=e.kind)
