"""Các thành phần giao diện dùng chung: CSS, tiêu đề trang, thẻ KPI, bảng màu biểu đồ."""
import html

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ---------------- Bảng màu (đã kiểm định cho người mù màu) ----------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _register_template():
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family="Be Vietnam Pro, Segoe UI, sans-serif", size=13, color="#2b2b2b"),
        colorway=SERIES,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=86, b=10),
        title=dict(font=dict(size=16, color="#1b1b1b"), x=0.01, xanchor="left", y=0.98, yanchor="top"),
        xaxis=dict(gridcolor="rgba(0,0,0,0.06)", zeroline=False, linecolor="rgba(0,0,0,0.15)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0.06)", zeroline=False, linecolor="rgba(0,0,0,0.15)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, title_text="", font=dict(size=12)),
        hoverlabel=dict(bgcolor="white", font_size=13, bordercolor="#ddd"),
        bargap=0.25,
    )
    pio.templates["itjobtrend"] = t
    pio.templates.default = "plotly_white+itjobtrend"


_register_template()

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select {
  font-family: 'Be Vietnam Pro', 'Segoe UI', sans-serif !important;
}
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#f4f8ff 0%,#fbf7ff 100%); border-right: 1px solid #e8ecf5; }

.sidebar-brand .logo { width:38px; height:38px; border-radius:12px; display:flex; align-items:center; justify-content:center;
  background: linear-gradient(135deg,#2a78d6,#6250d6); color:#fff; font-weight:800; font-size:18px; }
.sidebar-brand .name { font-weight:800; font-size:1.1rem; color:#1b1b1b; line-height:1.1 }
.sidebar-brand .sub { font-size:.75rem; color:#6b6b6b }

.page-hero { border-radius: 20px; padding: 1.4rem 1.6rem; margin-bottom: 1.2rem;
  background: linear-gradient(120deg,#eaf2ff 0%,#f3eeff 55%,#fff4ee 100%); border: 1px solid #e6eaf3; }
.page-hero h1 { font-size: 1.75rem; font-weight: 800; margin: 0 0 .25rem 0; color:#14213d; }
.page-hero p { margin: 0; color: #4a4f5c; font-size: .98rem; }
.page-hero .icon { font-size: 1.8rem; margin-right: .4rem }

.kpi-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr)); gap: .8rem; margin: .4rem 0 1rem 0; }
.kpi { background:#fff; border:1px solid #e9ecf2; border-radius:16px; padding: .9rem 1rem; box-shadow: 0 1px 2px rgba(16,24,40,.04); }
.kpi .label { color:#667085; font-size:.8rem; font-weight:600; text-transform: uppercase; letter-spacing:.03em }
.kpi .value { font-size:1.4rem; line-height:1.25; font-weight:800; color:#101828; margin-top:.15rem; }
.kpi .note { font-size:.8rem; color:#667085; margin-top:.1rem }
.kpi .note.up { color:#127a52 } .kpi .note.down { color:#b42318 }

.card { background:#fff; border:1px solid #e9ecf2; border-radius:16px; padding:1rem 1.2rem; margin-bottom: .8rem; }
.card h4 { margin:.1rem 0 .4rem 0; font-weight:700 }
.pill { display:inline-block; padding:.15rem .6rem; border-radius:999px; font-size:.78rem; font-weight:600; margin:0 .25rem .25rem 0;
  background:#eef4ff; color:#1c5cab; border:1px solid #d6e4fb }
.pill.green { background:#e8f7f0; color:#0f7a52; border-color:#c7ecd9 }
.pill.red { background:#fdecec; color:#b42318; border-color:#f8d0cf }
.pill.gray { background:#f2f2f0; color:#52514e; border-color:#e2e1dc }

.step { display:flex; gap:.8rem; align-items:flex-start; margin-bottom:.6rem }
.step .num { flex: 0 0 30px; height:30px; border-radius:50%; background:#2a78d6; color:#fff; font-weight:700;
  display:flex; align-items:center; justify-content:center }
.pipeline { display:flex; flex-wrap:wrap; gap:.4rem; align-items:center; margin:.5rem 0 1rem 0 }
.pipeline .node { background:#fff; border:1px solid #d6e4fb; border-radius:12px; padding:.45rem .8rem; font-weight:600; font-size:.88rem; color:#1c5cab }
.pipeline .arrow { color:#98a2b3; font-weight:700 }

div[data-testid="stMetric"] { background:#fff; border:1px solid #e9ecf2; border-radius:14px; padding:.6rem .9rem; }
.stTabs [data-baseweb="tab-list"] { gap: .3rem; }
.stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: .4rem 1rem; }
.stDownloadButton button, .stButton button { border-radius: 10px; font-weight:600 }

/* ---------- Chat kiểu Gemini ---------- */
.gemini-hello { font-size: 2.6rem; font-weight: 800; line-height:1.15; margin: 1.2rem 0 .2rem 0;
  background: linear-gradient(90deg,#4285f4 0%,#9b72cb 45%,#d96570 80%); -webkit-background-clip:text; background-clip:text; color: transparent; }
.gemini-sub { font-size: 1.7rem; font-weight: 600; color:#c4c7c5; margin-bottom: 1.4rem }
.suggest-note { color:#667085; font-size:.85rem; margin-bottom:.4rem }
[data-testid="stChatMessage"] { border-radius: 18px; padding: .6rem .9rem; }
[data-testid="stChatInput"] textarea { font-size: 1rem }
.key-warning { border-radius:16px; padding:1rem 1.2rem; background:#fff8eb; border:1px solid #f7d79b; color:#6b4a00 }
.key-warning code { background:#fff; }
.footer-note { color:#98a2b3; font-size:.8rem; text-align:center; margin-top:2rem }
</style>
"""


def inject_css():
    from src.analytics.ui.theme import CSS as EXT_CSS   # CSS bổ sung cho các module phân tích mở rộng
    st.markdown(CSS + EXT_CSS, unsafe_allow_html=True)


# Icon chuyên nghiệp (SVG) thay cho emoji ở tiêu đề trang
_HERO_ICONS = {"🏠": "dashboard", "📥": "upload", "🧹": "clean", "🔎": "chart", "🧩": "code", "📊": "dashboard",
               "📈": "trend", "💵": "money", "📤": "report", "✨": "chat"}


def page_header(icon: str, title: str, subtitle: str = ""):
    from src.analytics.ui.components import icon as svg_icon
    name = _HERO_ICONS.get(icon, "chart")
    st.markdown(
        f"""<div class="page-hero"><div class="hero-row"><div class="hero-ico">{svg_icon(name, 24, "#fff")}</div>
        <div><h1 style="margin:0">{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div></div></div>""",
        unsafe_allow_html=True,
    )


def kpi_cards(items):
    """items: list of (label, value, note, note_class)"""
    parts = []
    for it in items:
        label, value = it[0], it[1]
        note = it[2] if len(it) > 2 else ""
        cls = it[3] if len(it) > 3 else ""
        parts.append(
            f'<div class="kpi"><div class="label">{html.escape(str(label))}</div>'
            f'<div class="value">{html.escape(str(value))}</div>'
            f'<div class="note {cls}">{html.escape(str(note))}</div></div>'
        )
    st.markdown(f'<div class="kpi-grid">{"".join(parts)}</div>', unsafe_allow_html=True)


def fmt_int(n) -> str:
    try:
        return f"{int(n):,}".replace(",", ".")
    except Exception:
        return str(n)


def fmt_pct(x, digits=1) -> str:
    try:
        return f"{x * 100:.{digits}f}%".replace(".", ",")
    except Exception:
        return str(x)


def style_fig(fig, height=380):
    fig.update_layout(height=height + 30, legend_title_text="")
    return fig


def fmt_usd(x) -> str:
    try:
        return f"${x/1000:,.0f}k"
    except Exception:
        return str(x)
