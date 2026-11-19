"""Thành phần giao diện dùng chung để mọi trang đồng nhất (header, KPI, section, callout, bảng, biểu đồ)."""
from __future__ import annotations

import functools
import html
import inspect

import pandas as pd
import streamlit as st

from src.analytics import state
from src.analytics.analysis.algorithms_doc import ALGORITHMS, FIELDS
from src.analytics.ui import nav
from src.analytics.ui.theme import BLUE, CSS, DANGER, LOGO_SVG, NAVY, TEAL, WARNING  # noqa: F401

# Biểu tượng SVG (nét mảnh, phong cách Lucide) - không dùng emoji thay icon
_P = {
    "dashboard": '<rect x="3" y="3" width="7" height="9" rx="1"/><rect x="14" y="3" width="7" height="5" rx="1"/><rect x="14" y="12" width="7" height="9" rx="1"/><rect x="3" y="16" width="7" height="5" rx="1"/>',
    "upload": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>',
    "clean": '<path d="m3 21 9-9"/><path d="M12.2 6.2 11 5"/><path d="m19 11-1.2-1.2"/><path d="M15 4V2"/><path d="M15 16v-2"/><path d="M8 9h2"/><path d="M20 9h2"/><path d="M17.8 11.8 19 13"/><path d="M15 9h.01"/><path d="M17.8 6.2 19 5"/>',
    "chart": '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>',
    "briefcase": '<rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/>',
    "layers": '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"/>',
    "trend": '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>',
    "alert": '<path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "chat": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "report": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>',
    "settings": '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.6 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.6a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>',
    "rows": '<line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/>',
    "columns": '<rect x="3" y="3" width="18" height="18" rx="2"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/>',
    "missing": '<circle cx="12" cy="12" r="10"/><line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/>',
    "check": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "copy": '<rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>',
    "hash": '<line x1="4" y1="9" x2="20" y2="9"/><line x1="4" y1="15" x2="20" y2="15"/><line x1="10" y1="3" x2="8" y2="21"/><line x1="16" y1="3" x2="14" y2="21"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "money": '<line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
    "map": '<path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/>',
    "code": '<polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>',
    "sparkle": '<path d="M12 3l1.9 5.8L20 10.7l-5.8 1.9L12 18.4l-1.9-5.8L4.3 10.7l5.8-1.9z"/>',
}


def icon(name: str, size: int = 20, color: str = "currentColor", stroke: float = 2) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" '
            f'stroke-linecap="round" stroke-linejoin="round">{_P.get(name, _P["chart"])}</svg>')


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ Tương thích phiên bản Streamlit
@functools.lru_cache(maxsize=None)
def stretch(fn_name: str) -> dict:
    """Tham số 'giãn hết chiều rộng' phù hợp với phiên bản Streamlit đang cài.

    Streamlit mới: width="stretch" (use_container_width bị khai tử). Streamlit cũ: use_container_width=True.
    """
    try:
        params = inspect.signature(getattr(st, fn_name)).parameters
    except (TypeError, ValueError, AttributeError):
        return {}
    w = params.get("width")
    if w is not None and isinstance(w.default, str):
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


def show_df(df: pd.DataFrame, height: int | None = None, hide_index: bool = True, **kw) -> None:
    args = dict(stretch("dataframe"))
    if hide_index:
        args["hide_index"] = True
    if height:
        args["height"] = height
    args.update(kw)
    st.dataframe(df, **args)


def plot(fig, key: str | None = None) -> None:
    st.plotly_chart(fig, key=key, config={"displaylogo": False, "responsive": True}, **stretch("plotly_chart"))


def button(label: str, key: str | None = None, primary: bool = False, full: bool = True, **kw) -> bool:
    extra = stretch("button") if full else {}
    return st.button(label, key=key, type="primary" if primary else "secondary", **extra, **kw)


def download(label: str, data, file_name: str, mime: str, key: str | None = None) -> None:
    st.download_button(label, data=data, file_name=file_name, mime=mime, key=key, **stretch("download_button"))


# ------------------------------------------------------------------ Bố cục
def page_header(title: str, subtitle: str, icon_name: str, badge: str | None = None) -> None:
    """Tiêu đề trang theo phong cách ITJobTrend (nền pastel), icon SVG thay cho emoji."""
    ss = st.session_state
    if badge is None and ss.get("meta") and ss.get("df") is not None:
        m = ss["meta"]
        badge = f"{html.escape(str(m.get('name', '')))[:40]} · {len(ss['df']):,} dòng"
    b = f'<div class="badge">{badge}</div>' if badge else ""
    st.markdown(f"""<div class="page-hero"><div class="hero-row"><div class="hero-ico">{icon(icon_name, 24, "#fff")}</div>
        <div><h1 style="margin:0">{html.escape(title)}</h1><p>{subtitle}</p></div>{b}</div></div>""", unsafe_allow_html=True)


def section(title: str, caption: str = "") -> None:
    cap = f"<span>{caption}</span>" if caption else ""
    st.markdown(f'<div class="section-title"><div class="bar"></div><h3>{html.escape(title)}</h3>{cap}</div>',
                unsafe_allow_html=True)


def kpi_cards(items: list[dict]) -> None:
    """items: [{label, value, hint, icon, accent}] - hiển thị bằng class .kpi của ITJobTrend + viền màu nhấn."""
    cards = []
    for it in items:
        accent = it.get("accent", BLUE)
        ico = icon(it.get("icon", "chart"), 16, accent)
        hint = f'<div class="note">{it["hint"]}</div>' if it.get("hint") else ""
        cards.append(f"""<div class="kpi accent" style="--accent:{accent}"><div class="top"><div class="label">{html.escape(str(it['label']))}</div>
            <div class="ico" style="background:{accent}1A">{ico}</div></div><div class="value">{it['value']}</div>{hint}</div>""")
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def callout(text: str, kind: str = "info") -> None:
    st.markdown(f'<div class="callout {kind}">{text}</div>', unsafe_allow_html=True)


def insights(items: list[str]) -> None:
    import re
    for t in items:
        safe = html.escape(t)
        safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
        st.markdown(f'<div class="insight">{safe}</div>', unsafe_allow_html=True)


def empty_state(title: str, text: str, icon_name: str = "upload", go_upload: bool = True) -> None:
    st.markdown(f"""<div class="empty">{icon(icon_name, 44, "#94A3B8", 1.6)}<h3>{html.escape(title)}</h3><p>{text}</p></div>""",
                unsafe_allow_html=True)
    if go_upload and nav.PAGES.get("dataset"):
        c = st.columns([2, 1, 2])[1]
        with c:
            if button("Chọn bộ dữ liệu phân tích", key=f"goto_upload_{title}", primary=True):
                st.switch_page(nav.PAGES["dataset"])


def require_data() -> pd.DataFrame | None:
    ss = st.session_state
    if not state.has_data(ss):
        empty_state("Chưa chọn bộ dữ liệu phân tích",
                    "Vào trang <b>Bộ dữ liệu phân tích</b> để chọn LinkedIn, Hacker News, dữ liệu Việt Nam hoặc tải lên file CSV của bạn.")
        return None
    return ss["df"]


def algorithm_card(key: str, expanded: bool = False) -> None:
    a = ALGORITHMS.get(key)
    if not a:
        return
    with st.expander(f"Cơ sở khoa học: {a['name']}", expanded=expanded):
        rows = "".join(f"<tr><td style='width:170px;color:#64748B;font-weight:600;padding:6px 8px;vertical-align:top'>{lab}</td>"
                       f"<td style='padding:6px 8px'>{html.escape(a.get(k, ''))}</td></tr>" for k, lab in FIELDS)
        st.markdown(f"<table style='width:100%;border-collapse:collapse;font-size:.88rem'>{rows}</table>", unsafe_allow_html=True)


def logo() -> None:
    """Logo hệ thống ở đầu sidebar (st.logo), dự phòng bằng HTML nếu phiên bản Streamlit chưa hỗ trợ."""
    from src.analytics.config import ASSETS_DIR
    if hasattr(st, "logo"):
        kw = {"icon_image": str(ASSETS_DIR / "logo.png")}
        try:
            if "size" in inspect.signature(st.logo).parameters:
                kw["size"] = "large"
        except (TypeError, ValueError):
            pass
        st.logo(str(ASSETS_DIR / "logo_full.png"), **kw)
    else:
        sidebar_brand()


def sidebar_brand() -> None:
    from src.analytics.config import APP_NAME, APP_VERSION
    st.sidebar.markdown(f"""<div class="sb-brand">{LOGO_SVG}<div><div class="name">{APP_NAME}</div>
        <div class="sub">IT Recruitment Intelligence · v{APP_VERSION}</div></div></div>""", unsafe_allow_html=True)


def sidebar_status() -> None:
    from src.analytics.config import get_gemini_key
    ss = st.session_state
    if state.has_data(ss):
        m = ss.get("meta") or {}
        ds = (f'<span class="dot" style="background:{TEAL}"></span><b>{html.escape(str(m.get("name", ""))[:28])}</b><br>'
              f'{len(ss["df"]):,} dòng · {ss["df"].shape[1]} cột<br>Cập nhật: {ss.get("updated_at", "")}')
    else:
        ds = f'<span class="dot" style="background:{WARNING}"></span>Chưa có dataset'
    ai = (f'<span class="dot" style="background:{TEAL}"></span>Gemini: đã cấu hình' if get_gemini_key()
          else f'<span class="dot" style="background:{DANGER}"></span>Gemini: chưa có API Key')
    st.sidebar.markdown(f'<div class="sb-card">{ds}</div><div class="sb-card">{ai}</div>', unsafe_allow_html=True)


def fmt_num(v, digits: int = 2) -> str:
    try:
        if v is None or pd.isna(v):
            return "—"
        if float(v).is_integer():
            return f"{int(v):,}".replace(",", ".")
        return f"{v:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except (TypeError, ValueError):
        return str(v)


__all__ = ["NAVY", "TEAL", "BLUE"]
