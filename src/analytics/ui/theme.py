"""Hệ thống thiết kế (design system): bảng màu, CSS toàn cục, template biểu đồ."""

# Bảng màu đồng bộ với giao diện ITJobTrend (src/ui.py)
NAVY = "#14213d"
NAVY_2 = "#1c2b4a"
BLUE = "#2a78d6"
BLUE_LIGHT = "#6da7ec"
TEAL = "#1baf7a"
TEAL_DARK = "#127a52"
BG = "#ffffff"
CARD = "#FFFFFF"
BORDER = "#e9ecf2"
TEXT = "#101828"
MUTED = "#667085"
DANGER = "#e34948"
WARNING = "#eda100"
SUCCESS = "#1baf7a"
VIOLET = "#6250d6"

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948", "#0EA5E9", "#64748B"]
SEQUENTIAL = [[0, "#cde2fb"], [0.5, "#3987e5"], [1, "#0d366b"]]
DIVERGING = [[0, "#e34948"], [0.5, "#f8fafc"], [1, "#2a78d6"]]

# CSS bổ sung cho các module phân tích mở rộng - được nối vào CSS gốc của ITJobTrend (src/ui.py)
CSS = f"""
<style>
.page-hero .hero-row {{ display:flex; align-items:center; gap:14px; }}
.page-hero .hero-ico {{ width:44px; height:44px; border-radius:12px; background: linear-gradient(135deg,{BLUE},{VIOLET});
    display:flex; align-items:center; justify-content:center; flex-shrink:0; }}
.page-hero .badge {{ margin-left:auto; background:#fff; border:1px solid #d6e4fb; color:#1c5cab;
    padding:5px 12px; border-radius:999px; font-size:.78rem; font-weight:600; white-space:nowrap; }}
.kpi {{ position:relative; overflow:hidden; transition: transform .15s ease, box-shadow .15s ease; }}
.kpi:hover {{ transform: translateY(-2px); box-shadow: 0 8px 20px rgba(16,24,40,.08); }}
.kpi.accent:before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background: var(--accent, {BLUE}); }}
.kpi .top {{ display:flex; align-items:center; justify-content:space-between; gap:6px; }}
.kpi .ico {{ width:30px; height:30px; border-radius:9px; display:flex; align-items:center; justify-content:center; flex-shrink:0; }}
.section-title {{ display:flex; align-items:center; gap:10px; margin: 16px 0 8px 0; }}
.section-title .bar {{ width:4px; height:20px; border-radius:4px; background: linear-gradient({BLUE}, {VIOLET}); }}
.section-title h3 {{ margin:0; padding:0; font-size:1.05rem; font-weight:700; }}
.section-title span {{ color:{MUTED}; font-size:.82rem; }}
.insight {{ background:#fff; border:1px solid {BORDER}; border-left:4px solid {BLUE}; border-radius:12px;
           padding:9px 13px; margin-bottom:7px; font-size:.9rem; }}
.callout {{ border-radius:12px; padding:11px 14px; margin: 8px 0; font-size:.9rem; border:1px solid; }}
.callout.info {{ background:#eef4ff; border-color:#d6e4fb; color:#1c3f75; }}
.callout.warn {{ background:#fff8eb; border-color:#f7d79b; color:#6b4a00; }}
.callout.ok {{ background:#e8f7f0; border-color:#c7ecd9; color:#0f5c3f; }}
.callout.err {{ background:#fdecec; border-color:#f8d0cf; color:#8a1c14; }}
.empty {{ text-align:center; padding: 42px 20px; background:#fff; border:1px dashed #cfd6e4; border-radius:16px; color:{MUTED}; }}
.empty h3 {{ color:{TEXT}; margin-top: 10px; }}
.pill.ok {{ background:#e8f7f0; color:#0f7a52; border-color:#c7ecd9 }}
.pill.no {{ background:#fdecec; color:#b42318; border-color:#f8d0cf }}
.pill.neutral {{ background:#eef4ff; color:#1c5cab; border-color:#d6e4fb }}
div[data-testid="stExpander"] {{ border-radius:12px; }}
</style>
"""

LOGO_SVG = f"""<svg width="40" height="40" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
<defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{BLUE}"/><stop offset="1" stop-color="{TEAL}"/></linearGradient></defs>
<rect width="40" height="40" rx="11" fill="url(#lg)"/>
<path d="M10 27 L16 20 L21 24 L30 13" stroke="white" stroke-width="2.6" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
<circle cx="30" cy="13" r="2.6" fill="white"/><circle cx="10" cy="27" r="2" fill="white" opacity=".8"/>
<rect x="9" y="30" width="22" height="2" rx="1" fill="white" opacity=".55"/></svg>"""
