"""Giới hạn số câu hỏi Gemini trong mỗi phiên, để người lạ trên bản web không dùng hết quota API key của nhóm.

Chỉ tính các câu hỏi dùng key của nhóm (từ .env / Secrets). Người dùng tự dán key riêng ở trang Trợ lý AI
thì không bị giới hạn, vì họ dùng quota của chính họ.
"""
from __future__ import annotations

from collections.abc import MutableMapping

from src.config import get_chat_limit

COUNTER_KEY = "gemini_questions_used"


def uses_group_key(ss: MutableMapping) -> bool:
    """True nếu phiên đang dùng key của nhóm (không có key người dùng tự nhập)."""
    return not str(ss.get("gemini_key_override", "") or "").strip()


def used(ss: MutableMapping) -> int:
    return int(ss.get(COUNTER_KEY, 0) or 0)


def remaining(ss: MutableMapping, limit: int | None = None) -> int | None:
    """Số câu còn lại; None nghĩa là không giới hạn."""
    limit = get_chat_limit() if limit is None else limit
    if limit <= 0 or not uses_group_key(ss):
        return None
    return max(0, limit - used(ss))


def can_ask(ss: MutableMapping, limit: int | None = None) -> bool:
    r = remaining(ss, limit)
    return r is None or r > 0


def register_question(ss: MutableMapping) -> None:
    """Ghi nhận một câu hỏi đã gửi bằng key của nhóm."""
    if uses_group_key(ss):
        ss[COUNTER_KEY] = used(ss) + 1
