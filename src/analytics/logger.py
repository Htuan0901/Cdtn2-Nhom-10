"""Logging tập trung + bộ lọc che API Key để không bao giờ ghi khóa bí mật ra file log."""
from __future__ import annotations

import logging
import os
import re
from logging.handlers import RotatingFileHandler

from src.analytics.config import LOG_DIR

_KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z_\-]{20,}")
_CONFIGURED = False


class RedactSecretsFilter(logging.Filter):
    """Thay mọi chuỗi giống Google API Key (và giá trị GEMINI_API_KEY hiện tại) bằng ***."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        redacted = redact(msg)
        if redacted != msg:
            record.msg, record.args = redacted, ()
        return True


def redact(text: str) -> str:
    if not text:
        return text
    text = _KEY_PATTERN.sub("***", str(text))
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if key and len(key) >= 8:
        text = text.replace(key, "***")
    return text


def get_logger(name: str = "app") -> logging.Logger:
    global _CONFIGURED
    root = logging.getLogger("jobmarket")
    if not _CONFIGURED:
        root.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        fh = RotatingFileHandler(LOG_DIR / "app.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        fh.addFilter(RedactSecretsFilter())
        root.addHandler(fh)
        root.propagate = False
        _CONFIGURED = True
    return root.getChild(name)
