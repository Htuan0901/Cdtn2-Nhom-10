import json
import re
import time

import streamlit as st

from src import chatbot
from src.analytics import state as ws_state
from src.analytics.ai.agent import DataAgent
from src.analytics.ai.gemini_client import GeminiClient
from src.analytics.ai.router import route
from src.analytics.ai.tools import DataTools
from src.analytics.db.database import log_activity
from src.config import get_gemini_key, get_gemini_model

MODELS = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite", "gemini-2.0-flash"]

from src.state import get_data

li, hn, _ = get_data(stop=False)
chatbot.set_data(li, hn, fc_result=st.session_state.get("fc_result"), salary_bundle=st.session_state.get("salary_bundle"))

# ---------------- Thanh cài đặt ----------------
with st.sidebar:
    st.divider()
    st.markdown("**✨ Cài đặt Trợ lý AI**")
    env_model = get_gemini_model()
    options = MODELS if env_model in MODELS else [env_model] + MODELS
    model = st.selectbox("Model Gemini", options, index=options.index(env_model))
    use_data = st.toggle("Kết nối với dữ liệu đã tải", True, disabled=li is None,
                         help="Cho phép AI đọc tóm tắt dữ liệu và gọi công cụ tính toán trên dữ liệu thật.")
    use_search = st.toggle("Tìm kiếm Google", False,
                           help="Dùng Google Search cho câu hỏi thời sự/ngoài dữ liệu. Khi bật, AI sẽ không gọi công cụ dữ liệu.")
    temp = st.slider("Độ sáng tạo (temperature)", 0.0, 1.5, 0.7, 0.1)
    with st.expander("🔑 Nhập API key tạm thời"):
        k = st.text_input("GEMINI_API_KEY", type="password", value=st.session_state.get("gemini_key_override", ""),
                          help="Chỉ lưu trong phiên hiện tại. Để lưu lâu dài, điền vào file .env")
        st.session_state["gemini_key_override"] = k
    if st.button("🗑️ Cuộc trò chuyện mới", width="stretch"):
        st.session_state["messages"] = []
        st.rerun()

api_key = get_gemini_key()
msgs = st.session_state.setdefault("messages", [])

# ---------------- Màn hình chào kiểu Gemini ----------------
if not msgs:
    st.markdown('<div class="gemini-hello">Xin chào, bạn 👋</div>', unsafe_allow_html=True)
    st.markdown('<div class="gemini-sub">Hôm nay tôi có thể giúp gì cho bạn?</div>', unsafe_allow_html=True)
    if li is not None:
        st.markdown('<div class="suggest-note">Gợi ý câu hỏi</div>', unsafe_allow_html=True)
        sugg = [
            ("📊", "Tóm tắt những insight quan trọng nhất về thị trường tuyển dụng CNTT trong dữ liệu"),
            ("🧩", "Top 10 kỹ năng cho vị trí Data Engineer là gì?"),
            ("💵", "So sánh lương trung vị của Python, Java, Go và JavaScript"),
            ("📈", "Kỹ năng nào tăng và giảm mạnh nhất từ 2018 đến 2023?"),
            ("🗺️", "Bang nào trả lương cao nhất cho Software Engineer?"),
            ("🎓", "Sinh viên năm cuối nên học kỹ năng gì để dễ xin việc nhất?"),
        ]
    else:
        st.info("Mẹo: tải và làm sạch dữ liệu trước để Trợ lý AI có thể trả lời câu hỏi về dữ liệu của bạn. "
                "Bạn vẫn có thể hỏi các câu hỏi chung ngay bây giờ.")
        sugg = [
            ("💡", "Dự báo chuỗi thời gian là gì? Giải thích Holt-Winters thật dễ hiểu"),
            ("🧪", "Gợi ý cấu trúc báo cáo đồ án khoa học dữ liệu"),
            ("🐍", "Viết code pandas đếm số tin tuyển dụng theo tháng"),
            ("🌏", "Kể cho tôi 3 xu hướng tuyển dụng CNTT nổi bật hiện nay"),
        ]
    cols = st.columns(2)
    for i, (ic, q) in enumerate(sugg):
        if cols[i % 2].button(f"{ic}  {q}", key=f"sg_{i}", width="stretch"):
            st.session_state["pending_prompt"] = q
            st.rerun()

# ---------------- Lịch sử hội thoại ----------------
for m in msgs:
    with st.chat_message(m["role"], avatar=":material/person:" if m["role"] == "user" else ":material/auto_awesome:"):
        st.markdown(m["content"])
        if m.get("calls"):
            with st.expander("Minh bạch tính toán: hàm Python đã gọi trên bộ dữ liệu phân tích"):
                st.caption("Số liệu do pandas tính trực tiếp; Gemini chỉ nhận kết quả này để diễn giải, không tự tính.")
                for c in m["calls"]:
                    st.code(f"{c['tool']}({', '.join(f'{k}={v!r}' for k, v in c['args'].items())})", language="python")
                st.json(m.get("results", {}), expanded=False)

ws = st.session_state
ws_df = ws.get("df")
if ws_df is not None and ws.get("meta"):
    st.caption(f"Bộ dữ liệu phân tích đang chọn: **{ws['meta'].get('name')}** ({len(ws_df):,} dòng). Câu hỏi định lượng về bộ dữ liệu này "
               "được Data Analysis Agent tính bằng Python trước, rồi mới nhờ Gemini diễn giải.")
if not api_key:
    st.markdown("""
<div class="key-warning">
<b>Chưa có Gemini API key.</b> Câu hỏi định lượng về bộ dữ liệu phân tích vẫn được trả lời (chế độ offline); các câu hỏi khác cần API key:<br>
1. Lấy key miễn phí tại <a href="https://aistudio.google.com/app/apikey" target="_blank">Google AI Studio</a>.<br>
2. Mở file <code>.env</code> trong thư mục dự án, điền: <code>GEMINI_API_KEY=key_cua_ban</code> rồi lưu file.<br>
3. Tải lại trang (F5). Hoặc dán key tạm thời vào ô <b>🔑 Nhập API key tạm thời</b> ở thanh bên.
</div>""", unsafe_allow_html=True)

prompt = st.chat_input("Hỏi ITJobTrend AI bất cứ điều gì…")
if not prompt and st.session_state.get("pending_prompt"):
    prompt = st.session_state.pop("pending_prompt")

KAGGLE_REF = re.compile(r"linkedin|hacker\s*news|\bhn\b|usd|\bbang\b|\bstate\b", re.I)


def _ws_snapshot() -> dict:
    res = {k: ws_state.get_result(ws, k) for k in ws_state.RESULT_KEYS}
    return {"df": ws.get("df"), "meta": ws.get("meta"), "time_col": ws.get("time_col"),
            "results": {k: v for k, v in res.items() if v is not None}, "ws_clean_log": ws.get("ws_clean_log")}

if prompt:
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(prompt)
    # --- Định tuyến: câu hỏi định lượng về BỘ DỮ LIỆU PHÂN TÍCH ⇒ Data Analysis Agent (Python tính, Gemini diễn giải);
    #     câu hỏi về LinkedIn/HN, kiến thức chung, thời sự ⇒ Gemini + công cụ Kaggle / Google Search.
    r = route(prompt, ws_df) if (ws_df is not None and use_data) else None
    use_agent = r is not None and r.kind == "data" and not KAGGLE_REF.search(prompt)
    calls, results, err = [], {}, False
    with st.chat_message("assistant", avatar=":material/auto_awesome:"):
        with st.spinner("Đang phân tích…"):
            if use_agent or not api_key:
                agent = DataAgent(DataTools(_ws_snapshot), GeminiClient(api_key=api_key or "", model=model), web_enabled=use_search)
                if use_agent or (ws_df is not None and r is not None and r.kind == "data"):
                    resp = agent.answer(prompt, msgs)
                    answer = resp.text
                    calls = [{"tool": n, "args": a} for n, a in resp.tool_calls]
                    results = json.loads(json.dumps(resp.tool_results, ensure_ascii=False, default=str))
                else:
                    answer = ("Câu hỏi này cần Gemini nhưng **chưa có API key**. Hãy điền `GEMINI_API_KEY` trong file `.env` "
                              "(hoặc nhập key tạm ở thanh bên) rồi thử lại. Câu hỏi định lượng về *bộ dữ liệu phân tích* vẫn được "
                              "trả lời bằng số liệu tính trực tiếp khi chưa có key.")
            else:
                ctx = chatbot.build_data_context(li, hn) if (use_data and li is not None) else None
                try:
                    answer = chatbot.ask_gemini(api_key, model, msgs, prompt, ctx, use_tools=use_data,
                                                use_search=use_search, temperature=temp)
                except chatbot.GeminiError as e:
                    answer, err = f"{e}", True

        def _typewriter(txt):
            for w in txt.split(" "):
                yield w + " "
                time.sleep(0.008)

        if err:
            st.error(answer)
        else:
            st.write_stream(_typewriter(answer))
    log_activity("chat", "Agent" if (use_agent or not api_key) else "Gemini", prompt[:200], ws.get("dataset_id"))
    msgs.append({"role": "user", "content": prompt})
    if not err:
        msgs.append({"role": "assistant", "content": answer, "calls": calls, "results": results})
    st.rerun()
